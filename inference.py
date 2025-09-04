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
import os, argparse, torch, rasterio
import numpy as np
from fusionnetgeolabel import FusionNetGeoLabel
from fusionnetgeolabel.utils import tile_image, merge_tiles

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True, type=str)
    ap.add_argument('--image', required=True, type=str, help='Input RGB image (png/jpg/tif) or GeoTIFF')
    ap.add_argument('--out', required=True, type=str, help='Output mask (GeoTIFF if input is GeoTIFF)')
    ap.add_argument('--tile', type=int, default=1024)
    ap.add_argument('--stride', type=int, default=768)
    ap.add_argument('--num_classes', type=int, default=6)
    return ap.parse_args()

def read_image(path):
    if path.lower().endswith('.tif') or path.lower().endswith('.tiff'):
        ds = rasterio.open(path)
        img = ds.read([1,2,3]).astype(np.float32)
        # simple 0-1 normalize per band
        for c in range(3):
            band = img[c]
            m, s = band.mean(), band.std() + 1e-6
            img[c] = (band - m) / s
        return img, ds
    else:
        import cv2
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
        img = (img - np.array([0.485,0.456,0.406]))/np.array([0.229,0.224,0.225])
        return img.transpose(2,0,1), None

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

    img, rio = read_image(args.image)
    import math
    img_t = torch.from_numpy(img).unsqueeze(0).to(device)
    tiles, coords, full = tile_image(img_t, args.tile, args.stride)
    probs = []
    with torch.no_grad():
        for t in tiles:
            logits = model(t)
            pr = torch.softmax(logits, dim=1)[0].cpu().numpy()
            probs.append(pr)
    prob = merge_tiles(probs, coords, img.shape[-2:], args.tile, args.stride, args.num_classes)
    pred = prob.argmax(0).astype(np.uint8)

    if rio is not None:
        meta = rio.meta
        meta.update(count=1, dtype='uint8')
        with rasterio.open(args.out, 'w', **meta) as dst:
            dst.write(pred, 1)
    else:
        import cv2
        cv2.imwrite(args.out, pred)
    print('Saved:', args.out)

if __name__ == '__main__':
    main()
