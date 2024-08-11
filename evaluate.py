"""
Author: Teerapong Panboonyuen (Kao Panboonyuen)
Project: Semantic Segmentation on Remotely Sensed Images using Deep Convolutional Encoder-Decoder Neural Network
Description: This script evaluates the trained semantic segmentation model on the test dataset.
             It computes metrics such as accuracy, Intersection over Union (IoU), and F1-score.
License: MIT License
"""

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from tqdm import tqdm

from dataset import RemoteSensingDataset
from model import SemanticSegmentationModel
from utils import compute_iou, load_checkpoint

def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data, labels in tqdm(dataloader):
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    accuracy = accuracy_score(all_labels.flatten(), all_preds.flatten())
    f1 = f1_score(all_labels.flatten(), all_preds.flatten(), average='weighted')
    iou = compute_iou(all_labels, all_preds)

    return accuracy, f1, iou

def main():
    # Load configuration
    data_cfg = {
        'test_dataset': 'path_to_test_dataset',
        'batch_size': 16,
        'num_workers': 4,
        'checkpoint_path': 'path_to_trained_model_checkpoint.pth',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    # Load test dataset
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
    test_dataset = RemoteSensingDataset(data_cfg['test_dataset'], transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=data_cfg['batch_size'], shuffle=False, num_workers=data_cfg['num_workers'])

    # Load model
    model = SemanticSegmentationModel()
    load_checkpoint(model, data_cfg['checkpoint_path'])
    model.to(data_cfg['device'])

    # Evaluate model
    accuracy, f1, iou = evaluate(model, test_loader, data_cfg['device'])

    # Print results
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test F1-Score: {f1:.4f}")
    print(f"Test IoU: {iou:.4f}")

if __name__ == "__main__":
    main()