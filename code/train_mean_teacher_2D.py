import argparse
import logging
import os
import random
import shutil
import sys
import time
import numpy as np
import torch
import torchio as tio
import torch.backends.cudnn as cudnn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss,BCELoss
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from PIL import ImageFile
from dataloaders.dataset import (BaseDataSets, BaseFetaDataSets, RandomGenerator, ResizeTransform,TwoStreamBatchSampler)
from networks.net_factory import net_factory
from utils import losses, metrics, ramps
from val_2D import test_single_volume,test_batch
import segmentation_models_pytorch as smp
from configs.configsMeanTeacher import  *
ImageFile.LOAD_TRUNCATED_IMAGES = True

# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"



def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return configs.consistency * ramps.sigmoid_rampup(epoch, configs.consistency_rampup)

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    def create_model(ema=False):
        # Network definition
        # model = net_factory(net_type=args.model, in_chns=1,
        #                     class_num=num_classes)
        #

        model = smp.Unet(
            encoder_name=args.backbone,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=1,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=num_classes)  # model output channels (number of classes in your dataset)

        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model = create_model()
    ema_model = create_model(ema=True)

    #TODO push model to GPU
    model.cuda()
    ema_model.cuda()

    # def worker_init_fn(worker_id):
    #     random.seed(args.seed + worker_id)


    # db_train = BaseDataSets(base_dir=args.root_path, split="train", num=None, transform=transforms.Compose([
    #     RandomGenerator(args.patch_size)
    # ]))
    #
    #
    # db_val = BaseDataSets(base_dir=args.root_path, split="val")

    # trainTransform=transforms.Compose([RandomGenerator(args.patch_size)])

    # valTransform = transforms.Compose([ResizeTransform(args.patch_size,mode='val')])

    trainTransform = tio.Compose([
        tio.RandomMotion(p=0.2),
        tio.ZNormalization(masking_method=tio.ZNormalization.mean),
        tio.RandomBlur(),
        tio.RandomNoise(p=0.5),
        tio.RandomFlip(),
        tio.OneOf({tio.RandomAffine(): 0.8,
            tio.RandomElasticDeformation(): 0.2}),
        # ResizeTransform(args.patch_size, mode='val')
        tio.Resize((args.patch_size[0],args.patch_size[1],1))
    ])


    valTransform = transforms.Compose([ tio.ZNormalization(masking_method=tio.ZNormalization.mean),
                                        tio.Resize((args.patch_size[0], args.patch_size[1], 1)) ])




    db_train = BaseFetaDataSets(base_dir=args.root_path, split="train", num=None, transform=trainTransform)

    db_val = BaseFetaDataSets(base_dir=args.root_path, split="val",num=None,transform=valTransform)

    total_slices = len(db_train)
    #
    print("Total silices is: {}, labeled slices is: {}".format(
        total_slices, len(db_train.labeled_idxs)))

    # labeled_slice = patients_to_slices(args.root_path, args.labeled_num)
    # print("Total silices is: {}, labeled slices is: {}".format(
    #     total_slices, labeled_slice))
    #


    # labeled_idxs = list(range(0, 664))
    # unlabeled_idxs = list(range(664, 1000))


    labeled_idxs = db_train.labeled_idxs
    unlabeled_idxs = db_train.unlabeled_idxs


    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, batch_size, batch_size-args.labeled_bs)


    # trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
    #                          num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=args.num_workers, pin_memory=True)


    model.train()

    valloader = DataLoader(db_val, batch_size=args.val_batch_size, shuffle=False,
                           num_workers=args.num_workers,drop_last=True)


    # optimizer = optim.SGD(model.parameters(), lr=base_lr,
    #                       momentum=0.9, weight_decay=0.0001)
    #

    optimizer = optim.Adam(model.parameters(), lr=base_lr)

    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)

    writer = SummaryWriter(snapshot_path + '/log')
    writer_1 = SummaryWriter(snapshot_path + '/log_val')

    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0 #TODO add to the best performance metric or split
    train_loss_list=[]
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            unlabeled_volume_batch = volume_batch[args.labeled_bs:]

            noise = torch.clamp(torch.randn_like(
                unlabeled_volume_batch) * 0.1, -0.2, 0.2)
            ema_inputs = unlabeled_volume_batch + noise

            outputs = model(volume_batch)
            outputs_soft = torch.softmax(outputs, dim=1)
            with torch.no_grad():
                ema_output = ema_model(ema_inputs)
                ema_output_soft = torch.softmax(ema_output, dim=1)

            loss_ce = ce_loss(outputs[:args.labeled_bs],
                              label_batch[:][:args.labeled_bs].long())
            loss_dice = dice_loss(
                outputs_soft[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1))

            if epoch_num < args.meanTeacherEpoch:
                consistency_loss = 0.0
            else:
                consistency_loss = torch.mean(
                    (outputs_soft[args.labeled_bs:] - ema_output_soft) ** 2)


            supervised_loss = 0.5 * (loss_dice + loss_ce)
            consistency_weight = get_current_consistency_weight(iter_num//150)


            loss = supervised_loss + consistency_weight * consistency_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_ema_variables(model, ema_model, args.ema_decay, iter_num)

            # lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9

            # for param_group in optimizer.param_groups:
            #     param_group['lr'] = lr_

            iter_num = iter_num + 1
            # writer.add_scalar('info/lr', lr_, iter_num)

            train_loss_list.append(loss.detach().cpu().numpy())

            writer.add_scalar('info/consistency_loss',
                              consistency_loss, iter_num)
            writer.add_scalar('info/consistency_weight',
                              consistency_weight, iter_num)

            logging.info(
                'iteration %d : loss : %f, loss_ce: %f, loss_dice: %f' %
                (iter_num, loss.item(), loss_ce.item(), loss_dice.item()))

            if iter_num % 20 == 0:

                #TODO visualize image in the correct format
                # image=tio.Image(tensor=volume_batch[0][..., np.newaxis]).as_pil().convert('L')
                # image=transforms.ToTensor()(image)

                image = volume_batch[1, 0:1, :, :]
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(
                    outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction',
                                 outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)


        if iter_num >= max_iterations:
            break

        model.eval()
        metric_list = []
        val_loss_list=[]




        for i_batch, sampled_batch in enumerate(valloader):
            metric_i,val_ce,val_dice = test_batch(
                sampled_batch["image"].cuda(), sampled_batch["label"].cuda(), model, classes=args.num_classes,ce_loss=ce_loss,dice_loss=dice_loss)

            val_loss_list.append(np.array([val_ce.detach().cpu().numpy(),val_dice.detach().cpu().numpy()]))

            metric_list.append(np.array(metric_i))


        #val per batch
        val_ce_loss_mean =np.mean(val_loss_list, axis=0)[0]
        val_dice_loss_mean =np.mean(val_loss_list, axis=0)[1]

        val_dice_mean=np.mean(metric_list, axis=0)[0] / args.batch_size
        performance = val_dice_mean

        train_loss_mean=np.mean(train_loss_list)

        val_loss_mean=0.5*(val_ce_loss_mean+val_dice_loss_mean)


        writer.add_scalar('info/model_total_loss', train_loss_mean, epoch_num)
        writer_1.add_scalar('info/model_total_loss', val_loss_mean, epoch_num)

        # mean_hd95 = np.sum(metric_list, axis=0)[1] / len(db_val)


        # for class_i in range(num_classes - 1):
        #     writer.add_scalar('info/val_{}_dice'.format(class_i + 1),
        #                       metric_list[class_i, 0], iter_num)
        #     writer.add_scalar('info/val_{}_hd95'.format(class_i + 1),
        #                       metric_list[class_i, 1], iter_num)

        # performance = np.mean(metric_list, axis=0)[0]

        # mean_hd95 = np.mean(metric_list, axis=0)[1]

        writer_1.add_scalar('info/val_mean_dice', performance, epoch_num)
        # writer.add_scalar('info/val_mean_hd95', mean_hd95, epoch_num)

        if performance > best_performance:
            best_performance = performance
            save_mode_path = os.path.join(snapshot_path,
                                          'iter_{}_dice_{}.pth'.format(
                                              iter_num, round(best_performance, 4)))
            # TODO save to best performance path
            save_best = os.path.join(snapshot_path,
                                     '{}_best_model.pth'.format(args.model))
            torch.save(model.state_dict(), save_mode_path)
            torch.save(model.state_dict(), save_best)

        logging.info(
            'iteration %d : mean_dice : %f' % (iter_num, performance))
        model.train()

        print(epoch_num+1,' finished')
    writer.close()
    writer_1.close()
    return "Training Finished!"


if __name__ == "__main__":

    configs=Configs('./configs/meanTeacher.ini')

    if not configs.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(configs.seed)
    np.random.seed(configs.seed)
    torch.manual_seed(configs.seed)
    torch.cuda.manual_seed(configs.seed)

    log_time=int(time.time())

    snapshot_path = "../model/{}_labeled/{}/{}".format(
        configs.exp, configs.model,log_time)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(vars(configs)))
    train(configs, snapshot_path)
