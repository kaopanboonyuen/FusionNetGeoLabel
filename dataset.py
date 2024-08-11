"""
Author: Teerapong Panboonyuen (Kao Panboonyuen)
Project: Semantic Segmentation on Remotely Sensed Images Using Deep Convolutional Encoder-Decoder Neural Network
Description: This script defines the dataset class for loading and preprocessing remotely sensed images 
             for the FusionNetGeoLabel model.
License: MIT License
"""

import os
import torch
from torch.utils.data import Dataset
from PIL import Image

class RemoteSensingDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", "_mask.png"))

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask