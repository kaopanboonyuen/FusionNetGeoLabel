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
import math, random, os
import torch
from torch.optim.lr_scheduler import _LRScheduler

def seed_everything(seed=1337):
    import numpy as np, random, torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.ema = type(model)().to(next(model.parameters()).device)
        self.ema.load_state_dict(model.state_dict())
        self.decay = decay
        for p in self.ema.parameters():
            p.requires_grad_(False)
    @torch.no_grad()
    def update(self, model):
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.copy_(v * self.decay + msd[k] * (1.0 - self.decay))
    def state_dict(self):
        return self.ema.state_dict()

class CosineLR(_LRScheduler):
    def __init__(self, optimizer, T_max, eta_min=1e-6, last_epoch=-1, warmup_epochs=0):
        self.T_max = T_max
        self.eta_min = eta_min
        self.warmup_epochs = warmup_epochs
        super().__init__(optimizer, last_epoch)
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # linear warmup
            return [base_lr * (self.last_epoch + 1) / max(1, self.warmup_epochs) for base_lr in self.base_lrs]
        # cosine
        e = self.last_epoch - self.warmup_epochs
        T = max(1, self.T_max - self.warmup_epochs)
        return [self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * e / T)) / 2 for base_lr in self.base_lrs]

def cosine_lr(optimizer, epochs, eta_min=1e-6, warmup_epochs=0):
    return CosineLR(optimizer, T_max=epochs, eta_min=eta_min, warmup_epochs=warmup_epochs)
