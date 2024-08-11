# **FusionNetGeoLabel** 🌍🛰️

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI: 10.58837/CHULA.THE.2019.158](https://img.shields.io/badge/DOI-10.58837/CHULA.THE.2019.158-blue.svg)](https://digital.car.chula.ac.th/chulaetd/8534/)

## **Overview** 📚

**FusionNetGeoLabel** is a cutting-edge deep learning framework tailored for semantic segmentation in remotely sensed imagery. This repository is a culmination of my Ph.D. research, aimed at enhancing the accuracy and efficiency of semantic labeling in satellite and aerial images. 🚀

## **Author** ✍️

**Teerapong Panboonyuen**  
*(also known as Kao Panboonyuen)*  
Ph.D. in Computer Engineering, Chulalongkorn University 🎓

## **Abstract** 📄

Semantic segmentation is a cornerstone in remote sensing, impacting various domains like agriculture 🌾, map updates 🗺️, and navigation 🚗. Despite the prominence of Deep Convolutional Encoder-Decoder (DCED) networks, they often falter in capturing low-level features such as rivers and low vegetation due to architectural constraints and limited domain-specific data.

This dissertation presents an advanced semantic segmentation framework designed for remote sensing, featuring five key innovations:

- **Global Convolutional Network (GCN):** 🧠 Enhances segmentation accuracy for remote sensing images.
- **Channel Attention Mechanism:** 🎯 Focuses on the most critical features for better performance.
- **Domain-Specific Transfer Learning:** 🛠️ Tackles the challenge of limited training data in remote sensing.
- **Feature Fusion (FF):** 🔄 Effectively integrates low-level features into the model.
- **Depthwise Atrous Convolution (DA):** 🔍 Refines feature extraction for improved segmentation.

Our experiments on private Landsat-8 datasets and the public "ISPRS Vaihingen" benchmark show that the proposed architecture significantly outperforms baseline models. 📊

## **Publications & Resources** 📚

- **Ph.D. Thesis:** [Semantic Segmentation on Remotely Sensed Images Using Deep Convolutional Encoder-Decoder Neural Network](https://digital.car.chula.ac.th/chulaetd/8534/) 📜
- **Code Repository:** [GitHub - FusionNetGeoLabel](https://github.com/kaopanboonyuen/FusionNetGeoLabel) 💻
- **Pretrained Models:** [Download Pretrained Models](https://github.com/kaopanboonyuen/FusionNetGeoLabel) 📥
- **ISPRS Vaihingen Dataset:** [Download Dataset](https://paperswithcode.com/dataset/isprs-vaihingen) 🗂️
- **ISPRS Vaihingen Leaderboard:** [Semantic Segmentation Leaderboard](https://paperswithcode.com/sota/semantic-segmentation-on-isprs-vaihingen) 🏆

<p align="center">
  <img src="img/GraphicalAbstract.png" alt="Graphical Abstract" width="600"/>
  <img src="img/p2_method_2.png" alt="Method 2" width="600"/>
  <img src="img/p3_method_3.png" alt="Method 3" width="600"/>
</p>

## **How to Use** 🔧

### **Training** 🏋️

To train the model, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/kaopanboonyuen/FusionNetGeoLabel.git
   cd FusionNetGeoLabel
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Prepare your dataset and adjust the configuration:**
   - Modify settings in `config.json` as needed.
4. **Start training:**
   ```bash
   python train.py --config config.json
   ```

### **Inference** 🔍

To perform inference using a pretrained model:

1. **Download the pretrained model:**
   - Available [here](https://github.com/kaopanboonyuen/FusionNetGeoLabel).
2. **Run the inference script:**
   ```bash
   python inference.py --model path_to_pretrained_model --image path_to_image
   ```

## **Citation** 📝

If this work contributes to your research, please cite it as follows:

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

<p align="center">
  <img src="img/out5.png" alt="Sample Output 1" width="400"/>
  <img src="img/out1.png" alt="Sample Output 2" width="400"/>
  <img src="img/out3.png" alt="Sample Output 3" width="400"/>
</p>

## **License** ⚖️

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.