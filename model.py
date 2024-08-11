"""
Author: Teerapong Panboonyuen (Kao Panboonyuen)
Project: Semantic Segmentation on Remotely Sensed Images Using Deep Convolutional Encoder-Decoder Neural Network
Description: This script defines the FusionNetGeoLabel model architecture, which is a CNN tailored for semantic
             segmentation on remotely sensed images using feature fusion and depthwise atrous convolution.
License: MIT License
"""

import torch
import torch.nn as nn

class FusionNetGeoLabel(nn.Module):
    def __init__(self, in_channels=3, out_channels=5, features=None):
        super(FusionNetGeoLabel, self).__init__()
        if features is None:
            features = [64, 128, 256, 512]
        
        self.encoder = nn.ModuleList()
        for feature in features:
            self.encoder.append(self._block(in_channels, feature))
            in_channels = feature

        self.decoder = nn.ModuleList()
        for feature in reversed(features):
            self.decoder.append(self._block(feature * 2, feature))
        
        self.bottleneck = self._block(features[-1], features[-1] * 2)

        self.final_layer = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for down in self.encoder:
            x = down(x)
            skip_connections.append(x)
            x = nn.MaxPool2d(kernel_size=2, stride=2)(x)
        
        x = self.bottleneck(x)

        skip_connections = skip_connections[::-1]
        for i in range(len(self.decoder)):
            x = nn.ConvTranspose2d(
                x.size(1), x.size(1) // 2, kernel_size=2, stride=2
            )(x)
            skip_connection = skip_connections[i]
            concat_skip = torch.cat((x, skip_connection), dim=1)
            x = self.decoder[i](concat_skip)
        
        return self.final_layer(x)

    def _block(self, in_channels, out_channels, kernel_size=3, padding=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )