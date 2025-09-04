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
import os, argparse, textwrap, sys, shutil

MSG = """
ISPRS Vaihingen requires registration and acceptance of terms.
1) Register at the official ISPRS benchmark website and obtain credentials/download links.
2) Provide the downloaded zip files (Top, IRRG, labels) or a cookie/token to scripted download.
3) Place the archives under <out>/raw and run this script again to auto-extract & prepare.
"""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=str, required=True, help='Output directory, e.g., data/isprs_vaihingen')
    ap.add_argument('--token', type=str, default=None, help='(Optional) auth token/cookie string')
    args = ap.parse_args()
    out = args.out
    raw = os.path.join(out, 'raw')
    os.makedirs(raw, exist_ok=True)
    print(MSG)
    print('Created folder:', raw)
    print('If you already downloaded archives, put them under raw and re-run.')
    # Placeholder: extracting if files exist.
    archives = [f for f in os.listdir(raw) if f.lower().endswith(('.zip','.7z','.tar','.tar.gz'))]
    if archives:
        print('Found archives:', archives)
        # Extraction is left to user environment (7z/unzip). Keep instructions concise.
        print('Please extract the archives manually here, then organize as: out/images, out/masks')
    else:
        print('No archives found yet. Exiting gracefully.')
if __name__ == '__main__':
    main()
