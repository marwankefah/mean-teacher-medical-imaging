import argparse

import time
import segmentation_models_pytorch as smp
from PIL import Image

import numpy as np
import torch

from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms

from dataloaders.dataset import (BaseDataSets, BaseFetaDataSets, RandomGenerator,
                                 TwoStreamBatchSampler,ResizeTransform)
from networks.net_factory import net_factory
from utils import losses, metrics, ramps
from val_2D import test_single_volume,test_batch,calculate_metric_percase

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/FETA/', help='Name of Experiment')

parser.add_argument('--model_path', type=str,
                    default='../model/FETA/studyBuddy_labeled/unetResnet34/unetResnet34_best_model.pth', help='Name of Experiment')

parser.add_argument('--split_type', type=str,
                    default='test', help='model_name')
parser.add_argument('--backbone', type=str,
                    default='resnet34', help='backbone')


parser.add_argument('--model', type=str,
                    default='unet', help='model_name')

parser.add_argument('--batch_size', type=int, default=16,
                    help='batch_size per gpu')

parser.add_argument('--patch_size', type=list,  default=[256, 256],
                    help='patch size of network input')

parser.add_argument('--seed', type=int,  default=1337, help='random seed')

parser.add_argument('--num_classes', type=int,  default=2,
                    help='output channel of network')

args = parser.parse_args()


def test_batch(outputs, label_batch, classes, ce_loss, dice_loss, patch_size=[256, 256]):
    out_soft = torch.softmax(outputs, dim=1)

    out = torch.argmax(out_soft, dim=1, keepdim=True)

    loss_ce = ce_loss(outputs.detach(), label_batch.long().detach())
    loss_dice = dice_loss(out_soft.detach(), label_batch.unsqueeze(1).detach())

    out = out.squeeze().cpu().detach().numpy()
    labels = label_batch.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        for j in range(out.shape[0]):
            metric_list.append(calculate_metric_percase(
                out[j] == i, labels[j] == i))
    return np.sum(metric_list, axis=0), loss_ce, loss_dice


def create_model(ema=False):
    # Network definition
    # model = net_factory(net_type=args.model, in_chns=1,
    #                     class_num=num_classes)
    #
    # todo check the difference between net factory and smp unet
    model = smp.Unet(
        encoder_name=args.backbone,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
        in_channels=1,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=args.num_classes)  # model output channels (number of classes in your dataset)

    if ema:
        for param in model.parameters():
            param.detach_()
    return model

model = create_model()
ema_model = create_model(ema=True)


# db_val = BaseFetaDataSets(base_dir=args.root_path, split=args.split_type,num=None,transform=None)
# valloader = DataLoader(db_val, batch_size=1, shuffle=False,
#                            num_workers=0)

db_val = BaseFetaDataSets(base_dir=args.root_path, split=args.split_type,num=None,transform=transforms.Compose([ResizeTransform(args.patch_size,mode='val')]))

valloader = DataLoader(db_val, batch_size=args.batch_size, shuffle=False,
                           num_workers=0, drop_last=True)


model.load_state_dict(torch.load(args.model_path))
model.cuda()
model.eval()
metric_list =[]
ce_loss = CrossEntropyLoss()
dice_loss = losses.DiceLoss(args.num_classes)
log_time = int(time.time())

for i_batch, sampled_batch in enumerate(valloader):

    with torch.no_grad():
        outputs = model(sampled_batch["image"].cuda())
        metric_i,_,_ = test_batch(outputs
            , sampled_batch["label"].cuda(), classes=args.num_classes,ce_loss=ce_loss,dice_loss=dice_loss)
        metric_list.append(np.array(metric_i))

        for idx, i in enumerate(sampled_batch['idx'].numpy().tolist()):
            psuedoMaskPath = args.root_path + '/'+args.split_type+'_'+str(log_time)+'/'+db_val.sample_list[i][0].split("\\")[-1]

            psuedoMask = (torch.softmax(outputs, dim=1).detach().cpu().numpy()[idx][1])

            # TODO put the thrshold in argument
            im = Image.fromarray(psuedoMask >= 0.5).convert('RGB')
            im.save(psuedoMaskPath)



# for i_batch, sampled_batch in enumerate(valloader):
#     metric_i,_ = test_single_volume(
#         sampled_batch["image"], sampled_batch["label"], model, classes=args.num_classes)
#     metric_list.append(np.array(metric_i))

# metric_list = metric_list / len(db_val)

performanceMean = np.mean(metric_list,axis=0)[0]/args.batch_size
# performancestd=np.std(metric_list, axis=0).squeeze()[0]/16
mean_hd95 = np.sum(metric_list,axis=0)[1]/len(db_val)
print(performanceMean)