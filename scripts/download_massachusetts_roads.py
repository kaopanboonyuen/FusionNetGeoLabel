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
import os, argparse, urllib.request, zipfile, shutil

URLS = [
    # Primary (may change over time). If unavailable, user can update this list.
    'https://www.cs.toronto.edu/~vmnih/data/mass_roads/train.tar.gz',
    'https://www.cs.toronto.edu/~vmnih/data/mass_roads/valid.tar.gz',
    'https://www.cs.toronto.edu/~vmnih/data/mass_roads/test.tar.gz',
]

def download(url, dst):
    try:
        print('Downloading', url)
        urllib.request.urlretrieve(url, dst)
        print('Saved to', dst)
        return True
    except Exception as e:
        print('Failed to download', url, '->', e)
        return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=str, required=True)
    args = ap.parse_args()
    raw = os.path.join(args.out, 'raw')
    os.makedirs(raw, exist_ok=True)
    for url in URLS:
        fname = os.path.join(raw, os.path.basename(url))
        if not os.path.exists(fname):
            ok = download(url, fname)
            if not ok:
                print('Please update the URL or download manually.')
    print('Please extract tar.gz files and prepare folder structure: images/, masks/ under', args.out)

if __name__ == '__main__':
    main()
