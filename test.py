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
import os, argparse, torch
from torch.utils.data import DataLoader
from fusionnetgeolabel import FusionNetGeoLabel
from fusionnetgeolabel.utils import SegFolder, RunningScore

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', type=str, required=True)
    ap.add_argument('--data_dir', type=str, required=True)
    ap.add_argument('--num_classes', type=int, default=6)
    return ap.parse_args()

def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    cfg = ckpt.get('cfg', {})
    model = FusionNetGeoLabel(num_classes=args.num_classes,
                              in_channels=cfg.get('in_channels', 3),
                              hr_channels=tuple(cfg.get('model', {}).get('hr_channels', [32,64,128,256])),
                              gcn_kernel=cfg.get('model', {}).get('gcn_kernel', 7),
                              da_rates=tuple(cfg.get('model', {}).get('da_rates', [1,2,4,8]))).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    ds = SegFolder(args.data_dir, split='val', img_size=cfg.get('img_size', 512),
                   mean=tuple(cfg.get('data', {}).get('mean', [0.485,0.456,0.406])),
                   std=tuple(cfg.get('data', {}).get('std', [0.229,0.224,0.225])))
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)
    metric = RunningScore(args.num_classes)
    with torch.no_grad():
        for img, mask, _ in dl:
            img = img.to(device)
            logits = model(img)
            pred = torch.argmax(logits, dim=1).cpu()
            metric.update(pred, mask)
    scores = metric.get_scores()
    print("Accuracy:", scores['acc'])
    print("mIoU:", scores['miou'])
    print("F1 per class:", scores['f1'])

if __name__ == '__main__':
    main()
