"""
Author: Teerapong Panboonyuen (Kao Panboonyuen)
Project: Semantic Segmentation on Remotely Sensed Images Using Deep Convolutional Encoder-Decoder Neural Network
Description: This script contains utility functions for data loading, checkpoint management, and other 
             helper functions for the FusionNetGeoLabel project.
License: MIT License
"""

import torch
from torch.utils.data import DataLoader
from dataset import RemoteSensingDataset
from torchvision import transforms

def load_data(data_cfg):
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])

    train_dataset = RemoteSensingDataset(data_cfg['train_dataset'], data_cfg['train_dataset'], transform)
    val_dataset = RemoteSensingDataset(data_cfg['val_dataset'], data_cfg['val_dataset'], transform)

    train_loader = DataLoader(train_dataset, batch_size=data_cfg['batch_size'], shuffle=True, num_workers=data_cfg['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=data_cfg['batch_size'], shuffle=False, num_workers=data_cfg['num_workers'])

    return train_loader, val_loader

def save_checkpoint(model, optimizer, epoch, checkpoint_path):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(checkpoint, checkpoint_path)