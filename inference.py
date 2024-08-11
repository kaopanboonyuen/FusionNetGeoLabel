"""
Author: Teerapong Panboonyuen (Kao Panboonyuen)
Project: Semantic Segmentation on Remotely Sensed Images Using Deep Convolutional Encoder-Decoder Neural Network
Description: This script handles inference using a pretrained FusionNetGeoLabel model. It loads the model and
             performs semantic segmentation on a provided image.
License: MIT License
"""

import torch
from PIL import Image
from torchvision import transforms
from model import FusionNetGeoLabel

def load_model(model_path, device):
    model = FusionNetGeoLabel()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def preprocess_image(image_path, device):
    image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    image = transform(image).unsqueeze(0).to(device)
    return image

def predict(model, image):
    with torch.no_grad():
        output = model(image)
        prediction = torch.argmax(output, dim=1)
    return prediction

def save_prediction(prediction, output_path):
    prediction = prediction.squeeze().cpu().numpy()
    pred_image = Image.fromarray(prediction.astype('uint8'))
    pred_image.save(output_path)

if __name__ == "__main__":
    model_path = 'path_to_pretrained_model.pth'
    image_path = 'path_to_input_image.jpg'
    output_path = 'path_to_save_prediction.png'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = load_model(model_path, device)
    image = preprocess_image(image_path, device)
    prediction = predict(model, image)
    save_prediction(prediction, output_path)