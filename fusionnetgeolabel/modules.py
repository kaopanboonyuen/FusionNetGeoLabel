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
from einops import rearrange

def conv3x3(in_c, out_c, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_c, out_c, 3, stride, padding=dilation, bias=False, groups=groups, dilation=dilation)

def conv1x1(in_c, out_c, stride=1):
    return nn.Conv2d(in_c, out_c, 1, stride, bias=False)

class BNAct(nn.Sequential):
    def __init__(self, c):
        super().__init__(nn.BatchNorm2d(c), nn.ReLU(inplace=True))

class DepthwiseSeparableConv(nn.Module):
    """depthwise 3x3 + pointwise 1x1"""
    def __init__(self, in_c, out_c, stride=1, dilation=1):
        super().__init__()
        self.dw = nn.Conv2d(in_c, in_c, 3, stride=stride, padding=dilation, dilation=dilation, groups=in_c, bias=False)
        self.pw = nn.Conv2d(in_c, out_c, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        return self.act(x)

# --------------------- HRNet-style backbone ---------------------
class ExchangeUnit(nn.Module):
    """Fuse features across branches by resizing and summing."""
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        C = channels
        self.proj = nn.ModuleList([
            nn.ModuleList([self._make_proj(C[i], C[j], i, j) for j in range(len(C))])
            for i in range(len(C))
        ])
    def _make_proj(self, c_in, c_out, i, j):
        if i == j:
            return nn.Identity()
        elif i < j:
            # downsample i->j
            ops = []
            for k in range(j - i):
                ops += [nn.Conv2d(c_in if k == 0 else c_out, c_out, 3, stride=2, padding=1, bias=False),
                        nn.BatchNorm2d(c_out), nn.ReLU(inplace=True)]
            return nn.Sequential(*ops)
        else:
            # upsample i->j
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, bias=False),
                nn.BatchNorm2d(c_out)
            )
    def forward(self, feats):
        # feats: list[Tensors] shallow->deep resolutions
        outs = []
        for j in range(len(self.channels)):
            y = 0
            for i, x in enumerate(feats):
                z = self.proj[i][j](x)
                if i > j:  # upsample
                    scale = 2 ** (i - j)
                    z = F.interpolate(z, scale_factor=scale, mode="bilinear", align_corners=False)
                y = y + z
            outs.append(F.relu(y, inplace=True))
        return outs

class HRStage(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks=2):
        super().__init__()
        self.branches = nn.ModuleList()
        for i, (ci, co) in enumerate(zip(in_channels, out_channels)):
            blocks = []
            for _ in range(num_blocks):
                blocks += [conv3x3(ci, co), nn.BatchNorm2d(co), nn.ReLU(inplace=True)]
                ci = co
            self.branches.append(nn.Sequential(*blocks))
        self.exchange = ExchangeUnit(out_channels)
    def forward(self, feats):
        feats = [b(x) for b, x in zip(self.branches, feats)]
        return self.exchange(feats)

class HRBackbone(nn.Module):
    def __init__(self, channels=(32, 64, 128, 256)):
        super().__init__()
        self.stem = nn.Sequential(
            conv3x3(3, 32, stride=2), BNAct(32),
            conv3x3(32, 32, stride=2), BNAct(32),
        )
        # stage1 single branch
        c1 = channels[0]
        self.layer1 = nn.Sequential(conv3x3(32, c1), BNAct(c1), conv3x3(c1, c1), BNAct(c1))
        # create 4-resolution branches
        self.transition1 = nn.ModuleList([
            nn.Sequential(conv3x3(c1, channels[0]), BNAct(channels[0])),
            nn.Sequential(conv3x3(c1, channels[1], stride=2), BNAct(channels[1]))
        ])
        self.stage2 = HRStage([channels[0], channels[1]], [channels[0], channels[1]], num_blocks=2)

        self.transition2 = nn.ModuleList([
            nn.Identity(),
            nn.Identity(),
            nn.Sequential(conv3x3(channels[1], channels[2], stride=2), BNAct(channels[2]))
        ])
        self.stage3 = HRStage([channels[0], channels[1], channels[2]], [channels[0], channels[1], channels[2]], num_blocks=2)

        self.transition3 = nn.ModuleList([
            nn.Identity(), nn.Identity(), nn.Identity(),
            nn.Sequential(conv3x3(channels[2], channels[3], stride=2), BNAct(channels[3]))
        ])
        self.stage4 = HRStage([channels[0], channels[1], channels[2], channels[3]],
                              [channels[0], channels[1], channels[2], channels[3]], num_blocks=2)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x0 = self.transition1[0](x)
        x1 = self.transition1[1](x)
        feats = self.stage2([x0, x1])
        x0, x1 = feats
        x2 = self.transition2[2](x1)
        feats = self.stage3([x0, x1, x2])
        x0, x1, x2 = feats
        x3 = self.transition3[3](x2)
        feats = self.stage4([x0, x1, x2, x3])
        return feats  # list of 4 feature maps
