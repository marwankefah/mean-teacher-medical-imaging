import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    else:
        return 0, 0


def test_single_volume(image, label, net, classes, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    x, y = image.shape[0], image.shape[1]
    slice = zoom(image, (patch_size[0] / x, patch_size[1] / y), order=0)
    input = torch.from_numpy(slice).unsqueeze(
        0).unsqueeze(0).float().cuda()
    net.eval()
    with torch.no_grad():
        out = torch.argmax(torch.softmax(
            net(input), dim=1), dim=1).squeeze(0)
        out = out.cpu().detach().numpy()
        pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
        # prediction[ind] = pred
        prediction = pred
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list,prediction

def test_batch(volume_batch, label_batch, net, classes,ce_loss,dice_loss, patch_size=[256, 256]):
    net.eval()
    # prediction = np.zeros_like(label)
    with torch.no_grad():
        outputs=net(volume_batch)

        out_soft=torch.softmax(outputs, dim=1)

        out = torch.argmax(out_soft, dim=1,keepdim=True)

        loss_ce = ce_loss(outputs.detach(),label_batch.long().detach())
        loss_dice = dice_loss(out_soft.detach(), label_batch.unsqueeze(1).detach())


        out = out.squeeze().cpu().detach().numpy()
        labels=label_batch.cpu().detach().numpy()
        metric_list = []
        for i in range(1, classes):
            for j in range(out.shape[0]):
                metric_list.append(calculate_metric_percase(
                    out[j] == i, labels[j] == i))
    return np.sum(metric_list,axis=0),loss_ce,loss_dice

