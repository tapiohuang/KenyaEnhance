import cv2
import torch
from skimage.metrics import structural_similarity as ssim
from torch import nn
import torch.nn.functional as F

import lpips_loss


class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()

    def forward(self, original, enhance):
        return ssim_loss(original, enhance) + mse_loss(original, enhance)


def mse_loss(original, enhance):
    return torch.nn.MSELoss()(original, enhance)


def psnr_loss(original, enhance):
    mse = mse_loss(original, enhance)
    return 10 * torch.log10((255 ** 2) / mse)


def _ssim_loss(original, enhance):
    avg_light_ori = torch.sum(original, dim=(2, 3)) / (original.size(2) * original.size(3))
    avg_light_enh = torch.sum(enhance, dim=(2, 3)) / (enhance.size(2) * enhance.size(3))
    contrast_ori = torch.sqrt(torch.sum((original - avg_light_ori.unsqueeze(2).unsqueeze(3)) ** 2, dim=(2, 3)) / (
            original.size(2) * original.size(3) - 1))
    contrast_enh = torch.sqrt(torch.sum((enhance - avg_light_enh.unsqueeze(2).unsqueeze(3)) ** 2, dim=(2, 3)) / (
            enhance.size(2) * enhance.size(3) - 1))
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    c3 = c2 / 2
    l = (2 * avg_light_ori * avg_light_enh + c1) / (avg_light_ori ** 2 + avg_light_enh ** 2 + c1)
    c = (2 * contrast_ori * contrast_enh + c2) / (contrast_ori ** 2 + contrast_enh ** 2 + c2)
    xy = torch.sum(torch.sub(original, avg_light_ori.unsqueeze(2).unsqueeze(3)) * torch.sub(enhance,
                                                                                            avg_light_enh.unsqueeze(
                                                                                                2).unsqueeze(3)),
                   dim=(2, 3)) / (original.size(2) * original.size(3) - 1)
    s = (c3 + xy) / (contrast_ori * contrast_enh + c3)
    ssim = l * c * s
    # ssim barch,channel
    return torch.sum((ssim + 1) / 2) / 3


def ssim_loss(original, enhance):
    batch, channel, height, width = original.size()
    loss = 0
    for i in range(batch):
        loss += _ssim_loss(original[i].unsqueeze(0), enhance[i].unsqueeze(0))
    return loss / batch


if __name__ == '__main__':
    original = cv2.imread('E:\LOL-v2\Real_captured\Train\\Low\\low00001.png')
    enhance = cv2.imread('E:\LOL-v2\Real_captured\Train\\Normal\\normal00001.png')
    original = torch.tensor(original).permute(2, 0, 1).unsqueeze(0).cuda()
    enhance = torch.tensor(enhance).permute(2, 0, 1).unsqueeze(0).cuda()
    print(ssim_loss(original, enhance))
