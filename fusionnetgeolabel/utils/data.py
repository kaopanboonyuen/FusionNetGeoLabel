# -*- coding: utf-8 -*-
# ======================================================================
#  FusionNetGeoLabel: HR-GCN-FF-DA for Remote Sensing Semantic Segmentation
#  Author: Teerapong (Kao) Panboonyuen et al.
#  Repository: FusionNetGeoLabel
#  License: MIT
#
#  Citation:
#  @phdthesis{panboonyuen2019semantic,
#    title     = {Semantic segmentation on remotely sensed images using deep convolutional encoder-decoder neural network},
#    author    = {Teerapong Panboonyuen},
#    year      = {2019},
#    school    = {Chulalongkorn University},
#    type      = {Ph.D. thesis},
#    doi       = {10.58837/CHULA.THE.2019.158},
#    address   = {Faculty of Engineering},
#    note      = {Doctor of Philosophy}
#  }
# ======================================================================
import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from .augmentation import build_train_aug, build_val_aug

def _load_img(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def _load_mask(path):
    mask = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(path)
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    return mask.astype(np.int64)

class SegFolder(Dataset):
    """Folder with paired images/masks.
    Structure:
    root/
      images/*.png|jpg|tif
      masks/*.png|tif  (integer labels 0..K-1)
    """
    def __init__(self, root, split="train", img_size=512, mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)):
        self.root = root
        self.img_dir = os.path.join(root, "images")
        self.mask_dir = os.path.join(root, "masks")
        self.ids = sorted([os.path.splitext(f)[0] for f in os.listdir(self.img_dir)])
        self.split = split
        if split == "train":
            self.aug = build_train_aug(img_size, mean, std)
        else:
            self.aug = build_val_aug(img_size, mean, std)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        _id = self.ids[i]
        img = _load_img(os.path.join(self.img_dir, _id + ".png")) if os.path.exists(os.path.join(self.img_dir, _id+".png")) else _load_img(os.path.join(self.img_dir, _id + ".jpg")) if os.path.exists(os.path.join(self.img_dir, _id+".jpg")) else _load_img(os.path.join(self.img_dir, _id + ".tif"))
        mask = _load_mask(os.path.join(self.mask_dir, _id + ".png")) if os.path.exists(os.path.join(self.mask_dir, _id+".png")) else _load_mask(os.path.join(self.mask_dir, _id + ".tif"))
        augmented = self.aug(image=img, mask=mask)
        img = augmented["image"]
        mask = augmented["mask"]
        return img.float(), torch.from_numpy(mask).long(), _id
