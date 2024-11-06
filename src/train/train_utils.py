import os
from typing import Tuple

import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms.v2 as transforms_v2
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image


ROOT_DIR_EXPERIMENTS = os.path.expanduser('~/cv-building-timelapse/data/experiments')
BATCH_SIZE = 64


class CustomImageDataset(Dataset):
    def __init__(self, annotations_filename: str, img_rootdir: str, keypoint_label: str, input_transform=None, target_transform=None):
        self.img_labels = pd.read_csv(os.path.join(img_rootdir, annotations_filename))
        self.img_rootdir = img_rootdir
        self.keypoint_label = keypoint_label
        self.input_transform = input_transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        input_img_path = os.path.join(self.img_rootdir, 'input', self.img_labels.iloc[idx, 0])
        input_image = read_image(input_img_path)
        target_img_path = os.path.join(self.img_rootdir, 'target', self.keypoint_label, self.img_labels.iloc[idx, 1])
        target_image = read_image(target_img_path)
        if self.input_transform:
            input_image = self.input_transform(input_image)
        if self.target_transform:
            target_image = self.target_transform(target_image)
        return input_image, target_image


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # Encoder: contracting path
        self.enc1 = self.double_conv(in_channels, 64)
        self.enc2 = self.double_conv(64, 128)
        self.enc3 = self.double_conv(128, 256)
        self.enc4 = self.double_conv(256, 512)

        # Bottleneck
        self.bottleneck = self.double_conv(512, 512)

        # Decoder: expansive path
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec4 = self.double_conv(1024, 256)
        self.dec3 = self.double_conv(512, 128)
        self.dec2 = self.double_conv(256, 64)
        self.dec1 = self.double_conv(128, 32)

        # Final output layer
        self.out_conv = nn.Conv2d(32, out_channels, kernel_size=1)
        self.out_sigmoid = nn.Sigmoid()

    def double_conv(self, in_channels, out_channels):
        # Double convolutional layers followed by ReLU activation
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Contracting path
        enc1 = self.enc1(x)
        enc2 = self.enc2(nn.functional.max_pool2d(enc1, 2))
        enc3 = self.enc3(nn.functional.max_pool2d(enc2, 2))
        enc4 = self.enc4(nn.functional.max_pool2d(enc3, 2))

        # Bottleneck
        bottleneck = self.bottleneck(nn.functional.max_pool2d(enc4, 2))

        # Expansive path
        up4 = self.upsample(bottleneck)
        dec4 = self.dec4(torch.cat([up4, enc4], dim=1))
        up3 = self.upsample(dec4)
        dec3 = self.dec3(torch.cat([up3, enc3], dim=1))
        up2 = self.upsample(dec3)
        dec2 = self.dec2(torch.cat([up2, enc2], dim=1))
        up1 = self.upsample(dec2)
        dec1 = self.dec1(torch.cat([up1, enc1], dim=1))

        # Output layer
        out = self.out_conv(dec1)
        out = self.out_sigmoid(out)
        return out


def get_image_transforms() -> Tuple[transforms_v2.Transform, transforms_v2.Transform]:
    input_transform = transforms_v2.Compose([
        transforms_v2.ToDtype(torch.float32, scale=True),
        transforms_v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    target_transform = transforms_v2.Compose([
        transforms_v2.ToDtype(torch.float32, scale=True),  # scaling takes it into [0, 1] range
    ])

    return input_transform, target_transform
