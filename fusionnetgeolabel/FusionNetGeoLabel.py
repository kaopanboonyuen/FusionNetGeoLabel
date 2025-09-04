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
from .modules import HRBackbone, DepthwiseSeparableConv
from .heads import GlobalConv, DepthwiseAtrousPyramid, FusionBlock

class FusionNetGeoLabel(nn.Module):
    """HR-GCN-FF-DA for semantic segmentation."""
    def __init__(self, num_classes=6, in_channels=3, hr_channels=(32,64,128,256), gcn_kernel=7, da_rates=(1,2,4,8)):
        super().__init__()
        assert in_channels == 3, "Backbone stem is defined for 3 channels; adapt if needed."
        self.backbone = HRBackbone(channels=hr_channels)
        # Project each stage to a common emb dim for fusion
        C = hr_channels
        emb = 128
        self.proj = nn.ModuleList([nn.Conv2d(C[i], emb, 1, bias=False) for i in range(4)])
        self.bn = nn.ModuleList([nn.BatchNorm2d(emb) for _ in range(4)])

        # GCN heads at multiple scales
        self.gcn = nn.ModuleList([GlobalConv(emb, emb, k=gcn_kernel) for _ in range(4)])

        # Feature fusion pyramid (top-down)
        self.fuse32_16 = FusionBlock(emb, emb, emb)  # fuse 1/32 to 1/16
        self.fuse16_8  = FusionBlock(emb, emb, emb)  # fuse 1/16 to 1/8
        self.fuse8_4   = FusionBlock(emb, emb, emb)  # fuse 1/8 to 1/4

        # Depthwise Atrous pyramid on the finest resolution
        self.da = DepthwiseAtrousPyramid(emb, emb, rates=da_rates)

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Conv2d(emb, emb, 3, padding=1, bias=False),
            nn.BatchNorm2d(emb),
            nn.ReLU(inplace=True),
            nn.Conv2d(emb, num_classes, 1)
        )

    def forward(self, x):
        H, W = x.shape[-2:]
        feats = self.backbone(x)   # [1/4, 1/8, 1/16, 1/32]
        outs = []
        for i, f in enumerate(feats):
            p = self.bn[i](self.proj[i](f))
            outs.append(self.gcn[i](p))

        x4, x8, x16, x32 = outs  # resolutions: 1/4, 1/8, 1/16, 1/32

        y16 = self.fuse32_16(x16, x32)
        y8  = self.fuse16_8(x8, y16)
        y4  = self.fuse8_4(x4, y8)

        y  = self.da(y4)
        logits = self.classifier(y)
        logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
        return logits
