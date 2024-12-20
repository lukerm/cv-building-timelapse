#  Copyright (C) 2024 lukerm of www.zl-labs.tech
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import os
from typing import Tuple

import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms.v2 as transforms_v2
from torch.utils.data import Dataset
from torchvision.io import read_image


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


class CustomImageMultiOutputDataset(Dataset):
    def __init__(self, annotations_filename: str, img_rootdir: str, input_transform=None, target_transform=None):
        self.img_labels = pd.read_csv(os.path.join(img_rootdir, annotations_filename))
        self.img_rootdir = img_rootdir
        self.input_transform = input_transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        input_img_path = os.path.join(self.img_rootdir, 'input', self.img_labels.iloc[idx, 0].split('/')[-1])
        input_image = read_image(input_img_path)
        if self.input_transform:
            input_image = self.input_transform(input_image)

        target_imgs_to_concat = []
        target_columns = self.img_labels.columns[1:]
        for column in target_columns:
            kp = column.split('_')[0]
            target_img_path = os.path.join(self.img_rootdir, 'target', kp, self.img_labels[column].iloc[idx].split('/')[-1])
            my_target_image = read_image(target_img_path)

            if self.target_transform:
                my_target_image = self.target_transform(my_target_image)

            target_imgs_to_concat.append(my_target_image)
        target_image = torch.cat(target_imgs_to_concat, dim=0)

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
        # self.bottleneck = self.double_conv(256, 256)

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
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
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
        up3 = self.upsample(dec4)  # use bottleneck when 3-layered
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
