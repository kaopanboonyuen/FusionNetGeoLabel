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
import os, argparse, yaml, time
import torch
from torch.utils.data import DataLoader
from fusionnetgeolabel import FusionNetGeoLabel
from fusionnetgeolabel.utils import SegFolder, SegLoss, RunningScore, cosine_lr, ModelEMA, seed_everything

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, default='configs/default.yaml')
    ap.add_argument('--save_dir', type=str, default=None)
    return ap.parse_args()

def main():
    args = parse_args()
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    seed_everything(cfg.get('seed', 1337))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_dir = args.save_dir or cfg.get('save_dir', 'runs')
    os.makedirs(save_dir, exist_ok=True)

    # Data
    train_ds = SegFolder(cfg['data']['train_dir'], split='train', img_size=cfg['img_size'], mean=tuple(cfg['data']['mean']), std=tuple(cfg['data']['std']))
    val_ds   = SegFolder(cfg['data']['val_dir'],   split='val',   img_size=cfg['img_size'], mean=tuple(cfg['data']['mean']), std=tuple(cfg['data']['std']))
    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['workers'], pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg['batch_size'], shuffle=False, num_workers=cfg['workers'], pin_memory=True)

    # Model
    model = FusionNetGeoLabel(num_classes=cfg['num_classes'],
                              in_channels=cfg['in_channels'],
                              hr_channels=tuple(cfg['model']['hr_channels']),
                              gcn_kernel=cfg['model']['gcn_kernel'],
                              da_rates=tuple(cfg['model']['da_rates'])).to(device)

    criterion = SegLoss(cfg['num_classes'])
    optim = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    sched = cosine_lr(optim, epochs=cfg['epochs'], eta_min=cfg['cosine_final_lr'], warmup_epochs=cfg['warmup_epochs'])
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.get('amp', True) and device=='cuda')
    ema = ModelEMA(model, decay=cfg.get('ema_decay', 0.999))

    best_miou = 0.0
    global_step = 0

    for epoch in range(cfg['epochs']):
        model.train()
        t0 = time.time()
        for imgs, masks, _ in train_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            optim.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=cfg.get('amp', True) and device=='cuda'):
                logits = model(imgs)
                loss = criterion(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            ema.update(model)
            global_step += 1
        sched.step()

        # Validation
        model.eval()
        metric = RunningScore(cfg['num_classes'])
        with torch.no_grad():
            for imgs, masks, _ in val_loader:
                imgs = imgs.to(device)
                masks = masks.to(device)
                logits = ema.ema(imgs)
                pred = torch.argmax(logits, dim=1)
                metric.update(pred.cpu(), masks.cpu())
        scores = metric.get_scores()
        miou = scores['miou']
        dt = time.time()-t0
        print(f"Epoch {epoch+1}/{cfg['epochs']} - loss {loss.item():.4f} - mIoU {miou:.4f} - time {dt:.1f}s")

        torch.save({'model': model.state_dict(),
                    'ema': ema.state_dict(),
                    'cfg': cfg}, os.path.join(save_dir, 'last.ckpt'))
        if miou > best_miou:
            best_miou = miou
            torch.save({'model': model.state_dict(),
                        'ema': ema.state_dict(),
                        'cfg': cfg}, os.path.join(save_dir, 'best.ckpt'))
    print("Training complete. Best mIoU =", best_miou)

if __name__ == '__main__':
    main()
