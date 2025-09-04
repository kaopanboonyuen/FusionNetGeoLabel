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
import albumentations as A
from albumentations.pytorch import ToTensorV2

def build_train_aug(img_size=512, mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)):
    return A.Compose([
        A.RandomCrop(img_size, img_size, always_apply=True),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ColorJitter(p=0.3),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])

def build_val_aug(img_size=512, mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)):
    return A.Compose([
        A.CenterCrop(img_size, img_size, always_apply=True),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])
