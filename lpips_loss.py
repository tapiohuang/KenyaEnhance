import cv2
import torch

import lpips

lpips_model = lpips.LPIPS(net='alex')
lpips_model = lpips_model.cuda()


def lpips_loss(original, enhance):
    return 1 - lpips_model(original, enhance).mean()


if __name__ == '__main__':
    original = cv2.imread('E:\LOL-v2\Real_captured\Train\\Normal\\normal00001.png')
    enhance = cv2.imread('E:\LOL-v2\Real_captured\Train\\Normal\\normal00001.png')
    original = torch.tensor(original).permute(2, 0, 1).unsqueeze(0).float().cuda()
    enhance = torch.tensor(enhance).permute(2, 0, 1).unsqueeze(0).float().cuda()
    print(lpips_loss(original, enhance))
