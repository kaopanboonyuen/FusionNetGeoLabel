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
import numpy as np

def tile_image(img, tile, stride):
    H, W = img.shape[-2:]
    tiles = []
    coords = []
    for y in range(0, max(1, H - tile + 1), stride):
        for x in range(0, max(1, W - tile + 1), stride):
            tiles.append(img[..., y:y+tile, x:x+tile])
            coords.append((y, x))
    return tiles, coords, (H, W)

def merge_tiles(prob_tiles, coords, full_shape, tile, stride, num_classes):
    H, W = full_shape
    prob = np.zeros((num_classes, H, W), dtype=np.float32)
    count = np.zeros((1, H, W), dtype=np.float32)
    for p, (y, x) in zip(prob_tiles, coords):
        h, w = p.shape[-2:]
        prob[:, y:y+h, x:x+w] += p
        count[:, y:y+h, x:x+w] += 1
    prob = prob / np.maximum(count, 1e-6)
    return prob
