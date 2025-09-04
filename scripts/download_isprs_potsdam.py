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
import os, argparse
MSG = """
ISPRS Potsdam requires registration and acceptance of terms.
Steps:
  1) Register and download the data manually from the official ISPRS Potsdam benchmark.
  2) Place archives under <out>/raw and re-run this script to proceed with extraction/tiling.
"""
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=str, required=True)
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, 'raw'), exist_ok=True)
    print(MSG)
    print('Created:', os.path.join(args.out, 'raw'))
if __name__ == '__main__':
    main()
