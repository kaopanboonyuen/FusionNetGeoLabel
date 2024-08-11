"""
Author: Teerapong Panboonyuen (Kao Panboonyuen)
Project: Semantic Segmentation on Remotely Sensed Images Using Deep Convolutional Encoder-Decoder Neural Network
Description: This script handles the training loop for the FusionNetGeoLabel model. It reads the configuration 
             from a JSON file, loads the data, initializes the model, and trains it on the specified dataset.
License: MIT License
"""

import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import RemoteSensingDataset
from model import FusionNetGeoLabel
from utils import load_data, save_checkpoint

def train(config):
    # Load configuration
    with open(config, 'r') as f:
        cfg = json.load(f)

    # Load data
    train_loader, val_loader = load_data(cfg['data'])

    # Initialize model
    model = FusionNetGeoLabel(cfg['model'])
    model = model.to(cfg['device'])

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg['training']['lr'])

    # Training loop
    for epoch in range(cfg['training']['epochs']):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(cfg['device']), labels.to(cfg['device'])

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{cfg['training']['epochs']}], Loss: {running_loss/len(train_loader):.4f}")

        # Validation
        if (epoch + 1) % cfg['training']['val_interval'] == 0:
            validate(model, val_loader, criterion, cfg['device'])

        # Save checkpoint
        if (epoch + 1) % cfg['training']['save_interval'] == 0:
            save_checkpoint(model, optimizer, epoch, cfg['training']['checkpoint_path'])

def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    print(f'Validation Loss: {val_loss/len(val_loader):.4f}')

if __name__ == "__main__":
    train('config.json')