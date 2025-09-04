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
import torch
import torch.nn as nn
import torch.nn.functional as F

class SegLoss(nn.Module):
    """Cross-entropy + optional dice."""
    def __init__(self, num_classes, dice_weight=0.2, ignore_index=255):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dice_weight = dice_weight
        self.ignore_index = ignore_index
        self.num_classes = num_classes
    def forward(self, logits, target):
        loss = self.ce(logits, target)
        if self.dice_weight > 0:
            loss = loss + self.dice_weight * self.dice_loss(logits, target)
        return loss
    def dice_loss(self, logits, target):
        # one-hot target
        N, C, H, W = logits.shape
        with torch.no_grad():
            mask = target != self.ignore_index
            tgt = torch.zeros((N, C, H, W), device=logits.device, dtype=torch.float32)
            valid = mask.sum()
            t = target.clone()
            t[~mask] = 0
            tgt.scatter_(1, t.unsqueeze(1), 1.0)
            tgt = tgt * mask.unsqueeze(1)
        prob = torch.softmax(logits, dim=1) * mask.unsqueeze(1)
        num = 2 * (prob * tgt).sum(dim=(0,2,3))
        den = (prob + tgt).sum(dim=(0,2,3)) + 1e-6
        dice = 1 - (num / den).mean()
        return dice
