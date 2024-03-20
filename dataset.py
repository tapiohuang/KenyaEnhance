import sys

import cv2
import torch
import torch.utils.data as data
import numpy as np
import os.path


class PairImage:
    def __init__(self, original_path, normal_path):
        self.original_path = original_path
        self.normal_path = normal_path

    def load_imgs(self):
        original_img = cv2.imread(self.original_path, cv2.IMREAD_COLOR)
        normal_img = cv2.imread(self.normal_path, cv2.IMREAD_COLOR)
        original_img = cv2.resize(original_img, (512, 512))
        normal_img = cv2.resize(normal_img, (512, 512))
        return original_img, normal_img


class LOLPairImagesDataset(data.Dataset):
    def __init__(self, lol_path, transform=None):
        self.pair_images = []
        self.lol_path = lol_path
        self._load()
        self.transform = transform

    def _load(self):
        low_path = os.path.join(self.lol_path, 'Low')
        normal_path = os.path.join(self.lol_path, 'Normal')
        low_images_path = os.listdir(low_path)
        for low_image_path in low_images_path:
            normal_image_name = low_image_path.replace('low', 'normal')
            normal_image_path = os.path.join(normal_path, normal_image_name)
            if os.path.exists(normal_image_path):
                pair_image = PairImage(os.path.join(low_path, low_image_path), normal_image_path)
                self.pair_images.append(pair_image)

    def __len__(self):
        return len(self.pair_images)

    def __getitem__(self, idx):
        original_img, normal_img = self.pair_images[idx].load_imgs()
        original_tensor = torch.tensor(original_img).permute(2, 0, 1).float() / 255
        normal_tensor = torch.tensor(normal_img).permute(2, 0, 1).float() / 255
        return original_tensor, normal_tensor


if __name__ == '__main__':
    dataset = LOLPairImagesDataset('E:\LOL-v2\Real_captured\Train')
    dataloader = data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
    for original, normal in dataloader:
        print(original.shape)
        print(normal.shape)
        break
