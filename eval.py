import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

import dataset
from net import KenYaNet

net = KenYaNet()
net = net.cuda()

test_dataset = dataset.LOLPairImagesDataset(lol_path='E:\LOL-v2\Real_captured\Test')
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=1)

if __name__ == '__main__':
    net.load_state_dict(torch.load('checkpoint.pth'))
    net.eval()
    for iteration, batch in enumerate(test_dataloader, 1):
        original_img, normal_img = batch[0], batch[1]
        original_img = original_img.cuda()
        normal_img = normal_img.cuda()
        enhance_img = net(original_img)
        enhance_img = original_img + enhance_img
        enhance_img = enhance_img * 255
        enhance_img = enhance_img.cpu().detach().numpy()
        enhance_img = enhance_img.squeeze()
        enhance_img = enhance_img.transpose(1, 2, 0)
        enhance_img = cv2.cvtColor(enhance_img, cv2.COLOR_RGB2BGR)
        enhance_img = cv2.resize(enhance_img, (600, 400))
        new_img = np.zeros((400, 600 * 2, 3), dtype=np.uint8)
        new_img[:, :600, :] = cv2.resize(
            (normal_img * 255).to(torch.uint8).cpu().detach().numpy().squeeze().transpose(1, 2, 0), (600, 400))
        new_img[:, 600:, :] = enhance_img
        cv2.imwrite('E:\LOL-v2\Real_captured\Test\Enhance\\enhance{:05d}.png'.format(iteration), new_img)
