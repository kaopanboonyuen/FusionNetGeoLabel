# FusionNetGeoLabel

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI: 10.58837/CHULA.THE.2019.158](https://img.shields.io/badge/DOI-10.58837/CHULA.THE.2019.158-blue.svg)](https://digital.car.chula.ac.th/chulaetd/8534/)

## Overview

**FusionNetGeoLabel** is an advanced deep learning framework designed to address the challenges of semantic segmentation in remotely sensed imagery. This repository contains the code, models, and documentation related to my Ph.D. work on enhancing the accuracy and efficiency of semantic labeling in satellite and aerial images.

## Author

**Teerapong Panboonyuen**  
(also known as Kao Panboonyuen)  
Ph.D. in Computer Engineering, Chulalongkorn University

## Abstract

Semantic segmentation is a critical task in remote sensing, impacting fields like agriculture planning, map updates, route optimization, and navigation. While Deep Convolutional Encoder-Decoder (DCED) networks are the state-of-the-art for this task, they often struggle with recovering low-level features (e.g., rivers, low vegetation) in remote sensing images due to limitations in their architecture and the scarcity of domain-specific training data.

This dissertation proposes an enhanced semantic segmentation architecture tailored to remote sensing. The key contributions are:

1. **Global Convolutional Network (GCN):** A modern CNN designed to improve the segmentation accuracy of remote sensing images.
2. **Channel Attention:** A mechanism to prioritize the most discriminative features.
3. **Domain-Specific Transfer Learning:** An approach to mitigate the challenge of limited training data in the remote sensing domain.
4. **Feature Fusion (FF):** A technique to capture and integrate low-level features effectively.
5. **Depthwise Atrous Convolution (DA):** A method to refine extracted features for better segmentation.

Our experiments, conducted on two private Landsat-8 datasets and the public "ISPRS Vaihingen" benchmark, demonstrate that the proposed architecture significantly outperforms baseline models.

## Publications & Resources

- **Ph.D. Thesis:** [Semantic Segmentation on Remotely Sensed Images Using Deep Convolutional Encoder-Decoder Neural Network](https://digital.car.chula.ac.th/chulaetd/8534/)
- **Code Repository:** [GitHub - FusionNetGeoLabel](https://github.com/kaopanboonyuen/FusionNetGeoLabel)
- **Pretrained Models:** [Download Pretrained Models](https://github.com/kaopanboonyuen/FusionNetGeoLabel)
- **ISPRS Vaihingen Dataset:** [Download Dataset](https://paperswithcode.com/dataset/isprs-vaihingen)
- **Scoreboard of ISPRS Vaihingen:** [Semantic Segmentation Leaderboard](https://paperswithcode.com/sota/semantic-segmentation-on-isprs-vaihingen)

![](img/GraphicalAbstract.png)
![](img/p2_method_2.png)
![](img/p3_method_3.png)

## How to Use

### Training

To train the model, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/kaopanboonyuen/FusionNetGeoLabel.git
   cd FusionNetGeoLabel
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Prepare your dataset and adjust the configuration settings in `config.json`.
4. Start training:
   ```bash
   python train.py --config config.json
   ```

### Inference

To perform inference using a pre-trained model:

1. Download the pretrained model from the [link](https://github.com/kaopanboonyuen/FusionNetGeoLabel).
2. Run the inference script:
   ```bash
   python inference.py --model path_to_pretrained_model --image path_to_image
   ```

## Citation

If you find this work useful in your research, please consider citing:

```bibtex
@phdthesis{panboonyuen2019semantic,
  title     = {Semantic segmentation on remotely sensed images using deep convolutional encoder-decoder neural network},
  author    = {Teerapong Panboonyuen},
  year      = {2019},
  school    = {Chulalongkorn University},
  type      = {Ph.D. thesis},
  doi       = {10.58837/CHULA.THE.2019.158},
  address   = {Faculty of Engineering},
  note      = {Doctor of Philosophy}
}
```
![](img/out5.png)
![](img/out1.png)
![](img/out3.png)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.