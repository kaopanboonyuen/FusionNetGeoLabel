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

def conv1x1(in_c, out_c, stride=1):
    return nn.Conv2d(in_c, out_c, 1, stride, bias=False)

class GlobalConv(nn.Module):
    """Global Convolutional Network head with large-kernel factorization."""
    def __init__(self, in_c, out_c, k=7):
        super().__init__()
        pad = k // 2
        self.left = nn.Sequential(
            nn.Conv2d(in_c, out_c, (k,1), padding=(pad,0), bias=False),
            nn.Conv2d(out_c, out_c, (1,k), padding=(0,pad), bias=False),
        )
        self.right = nn.Sequential(
            nn.Conv2d(in_c, out_c, (1,k), padding=(0,pad), bias=False),
            nn.Conv2d(out_c, out_c, (k,1), padding=(pad,0), bias=False),
        )
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.left(x) + self.right(x)
        x = self.bn(x)
        return self.act(x)

class DepthwiseAtrousPyramid(nn.Module):
    """Depthwise Atrous (rates r in R) using depthwise separable convs."""
    def __init__(self, in_c, out_c, rates=(1,2,4,8)):
        super().__init__()
        self.branches = nn.ModuleList()
        for r in rates:
            self.branches.append(nn.Sequential(
                nn.Conv2d(in_c, in_c, 3, padding=r, dilation=r, groups=in_c, bias=False),
                nn.BatchNorm2d(in_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_c, out_c, 1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            ))
        self.proj = nn.Sequential(nn.Conv2d(out_c*len(rates), out_c, 1, bias=False),
                                  nn.BatchNorm2d(out_c), nn.ReLU(inplace=True))
    def forward(self, x):
        feats = [b(x) for b in self.branches]
        x = torch.cat(feats, dim=1)
        return self.proj(x)

class FusionBlock(nn.Module):
    """Simple additive feature fusion after aligning channels & size."""
    def __init__(self, in_low, in_high, out_c):
        super().__init__()
        self.low_proj = nn.Sequential(nn.Conv2d(in_low, out_c, 1, bias=False), nn.BatchNorm2d(out_c))
        self.high_proj = nn.Sequential(nn.Conv2d(in_high, out_c, 1, bias=False), nn.BatchNorm2d(out_c))
        self.act = nn.ReLU(inplace=True)
    def forward(self, low, high):
        # upsample high to low
        high = F.interpolate(self.high_proj(high), size=low.shape[-2:], mode="bilinear", align_corners=False)
        low = self.low_proj(low)
        out = self.act(low + high)
        return out
