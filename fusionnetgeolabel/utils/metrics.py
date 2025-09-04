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
import numpy as np

def confusion_matrix(pred, target, num_classes):
    with torch.no_grad():
        k = (target >= 0) & (target < num_classes)
        return torch.bincount(
            num_classes * target[k].view(-1) + pred[k].view(-1),
            minlength=num_classes ** 2
        ).reshape(num_classes, num_classes)

def f1_per_class(cm):
    # cm: CxC
    tp = np.diag(cm)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    f1 = 2 * tp / (2 * tp + fp + fn + 1e-9)
    return f1

class RunningScore:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()
    def reset(self):
        self.cm = torch.zeros(self.num_classes, self.num_classes, dtype=torch.int64)
    def update(self, pred, gt):
        self.cm += confusion_matrix(pred, gt, self.num_classes)
    def get_scores(self):
        cm = self.cm.cpu().numpy()
        acc = np.diag(cm).sum() / (cm.sum() + 1e-9)
        iu = np.diag(cm) / (cm.sum(1) + cm.sum(0) - np.diag(cm) + 1e-9)
        miou = np.nanmean(iu)
        f1 = f1_per_class(cm)
        return {"acc": float(acc), "miou": float(np.nanmean(iu)), "f1": f1.tolist()}
