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

import json
import os
import shutil
import time

import numpy as np
import pandas as pd
from PIL import Image
from torchvision.transforms import v2 as transforms_v2

from src.utils import extract_date_from_filename, hash_date


LABELS_FNAME = os.path.expanduser('~/cv-building-timelapse/data/labels/project-2-at-2024-10-11-16-11-19bfb516.json')

IMG_LOAD_DIR = os.path.expanduser('~/mydata/media/')
CROP_SIZE = (512, 512)
SAVE_ROOT_DIR = os.path.expanduser(f'~/cv-building-timelapse/data/experiments/{CROP_SIZE[0]}')
os.makedirs(SAVE_ROOT_DIR, exist_ok=True)

TEST_SET_DIGITS = ['0', '1']
VAL_SET_DIGITS = ['2', '3']
TRAIN_SET_DIGITS = ['4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f']

TARGET_LABELS = ['L1', 'L2', 'L3', 'R1', 'R2', 'R3', 'R4', 'D1', 'D2', 'D3', 'C1', 'C2',]

N_TRAIN_CROPS = 5  # note: this is now effectively ignored
SIGMA = 2.


def create_gaussian_output_image(
    width_label: float, height_label: float, img_width: int, img_height: int, sigma: float = SIGMA
) -> Image:
    """
    Create an Image of white / gray Gaussian distribution centered at the labelled point, black elsewhere.

    :param width_label: float, the labelled point's x-coordinate as a proportion of the image width (i.e. in [0, 1])
    :param height_label: float, the labelled point's y-coordinate as a proportion of the image height (i.e. in [0, 1])
    :param img_width: int, the image's width in pixels
    :param img_height: int, the image's height in pixels
    :param sigma: float, the standard deviation of the Gaussian distribution
    :return: Image, the Gaussian distribution as a grayscale Image
    """
    x_lab = int(width_label / 100 * img_width)
    y_lab = int(height_label / 100 * img_height)

    # Create a grid of x, y values
    x = np.arange(0, img_width, 1)
    y = np.arange(0, img_height, 1)
    xx, yy = np.meshgrid(x, y)

    # Create a Gaussian distribution
    zz = np.exp(-((xx - x_lab) ** 2 + (yy - y_lab) ** 2) / (2 * sigma ** 2))

    # Normalize the distribution in [0, 255] for grayscale
    zz = 255 * (zz - zz.min()) / (zz.max() - zz.min())

    return Image.fromarray(zz).convert('L')


if __name__ == "__main__":

    # Create storage folders for cropped images
    for loc in ['train', 'val', 'test']:
        for loc2 in ['input', 'target']:
            os.makedirs(os.path.join(SAVE_ROOT_DIR, loc, loc2), exist_ok=True)
            if CROP_SIZE[0] == 512:
                if loc2 == 'target':
                    for label in TARGET_LABELS:
                        os.makedirs(os.path.join(SAVE_ROOT_DIR, loc, loc2, label), exist_ok=True)
            elif CROP_SIZE[0] == 256:
                for label in TARGET_LABELS:
                    os.makedirs(os.path.join(SAVE_ROOT_DIR, loc, loc2, label), exist_ok=True)

    with open(LABELS_FNAME) as j:
        labels = json.load(j)

    labels = [label for label in labels if label['annotations'][0]['result']]
    dates = [extract_date_from_filename(label['data']['img']) for label in labels]
    hashes = [hash_date(d) for d in dates]

    train_transforms_512 = transforms_v2.Compose([
        transforms_v2.FiveCrop(size=CROP_SIZE),
    ])
    train_transforms_256 = transforms_v2.Compose([
        transforms_v2.FiveCrop(size=CROP_SIZE),
        transforms_v2.RandomHorizontalFlip(p=0.5),
    ])
    val_transforms = transforms_v2.Compose([
        transforms_v2.FiveCrop(size=CROP_SIZE),
    ])

    # Create CSV files for each of the folds and key-point labels
    dfs = {loc: {label: pd.DataFrame(None) for label in TARGET_LABELS} for loc in ['train', 'val', 'test']}


    for i, (label, h) in enumerate(zip(labels, hashes)):
        if i > 0 and i % 50 == 0:
            print(f'Processed {i} / {len(labels)} labels')

        img_loc = label['data']['img'].replace('/data/', IMG_LOAD_DIR)
        img_fname = os.path.split(img_loc)[1]
        fname_split_ext = os.path.splitext(img_fname)

        with Image.open(img_loc) as img:
            outputs = {}
            # Generate an output image (Gaussian distribution) for each keypoint label
            for annotation in label['annotations'][0]['result']:
                kp_label = annotation['value']['keypointlabels'][0]
                outputs[kp_label] = create_gaussian_output_image(
                    width_label=annotation['value']['x'], height_label=annotation['value']['y'],
                    img_width=img.width, img_height=img.height,
                    sigma=SIGMA,
                )

            if CROP_SIZE[0] == 512:

                # Take random (train) / controlled (val/test) crops of the original image, and the transformed Gaussian output
                # Note: we use md5-hash shuffling (last digit) to determine the train/val/test folds
                # Note: for val/test, we take two crops (bottom left and bottom right) as these are the most interesting regions
                if False: #h[-1] in TRAIN_SET_DIGITS:
                    fold = 'train'

                    for c in range(N_TRAIN_CROPS):
                        input_crop, target_crop = train_transforms_512(img, outputs)
                        crop_fname = f'{fname_split_ext[0]}.{c}{fname_split_ext[1]}'

                        input_save_path = os.path.join(SAVE_ROOT_DIR, 'train', 'input', crop_fname)
                        input_crop.save(input_save_path)
                        for kp, target_img in target_crop.items():
                            target_save_path = os.path.join(SAVE_ROOT_DIR, 'train', 'target', kp, crop_fname)
                            target_img.save(target_save_path)

                            # record the crop locations in the relevant dataframe
                            dfs['train'][kp] = pd.concat([
                                dfs['train'][kp],
                                pd.DataFrame({
                                    'input_crop_loc': [input_save_path],
                                    'target_crop_loc': [target_save_path]})
                            ])

                elif h[-1] in TRAIN_SET_DIGITS + VAL_SET_DIGITS + TEST_SET_DIGITS:
                    if h[-1] in TRAIN_SET_DIGITS:
                        fold = 'train'
                    elif h[-1] in VAL_SET_DIGITS:
                        fold = 'val'
                    else:
                        fold = 'train'  # although this looks odd, we're not really using test fold, so add it into training dataset

                    input_crops, target_crops = val_transforms(img, outputs)
                    _, _, bl_crop, br_crop, _ = input_crops
                    bl_input_save_path = os.path.join(SAVE_ROOT_DIR, fold, 'input', f'{fname_split_ext[0]}.0{fname_split_ext[1]}')
                    br_input_save_path = os.path.join(SAVE_ROOT_DIR, fold, 'input', f'{fname_split_ext[0]}.1{fname_split_ext[1]}')
                    bl_crop.save(bl_input_save_path)
                    time.sleep(0.05)
                    br_crop.save(br_input_save_path)
                    time.sleep(0.05)
                    for kp, target_imgs in target_crops.items():
                        _, _, bl_target_crop, br_target_crop, _ = target_imgs
                        bl_target_save_path = os.path.join(SAVE_ROOT_DIR, fold, 'target', kp, f'{fname_split_ext[0]}.0{fname_split_ext[1]}')
                        br_target_save_path = os.path.join(SAVE_ROOT_DIR, fold, 'target', kp, f'{fname_split_ext[0]}.1{fname_split_ext[1]}')
                        bl_target_crop.save(bl_target_save_path)
                        time.sleep(0.01)
                        br_target_crop.save(br_target_save_path)
                        time.sleep(0.01)

                        # record the crop locations in the relevant dataframe
                        dfs[fold][kp] = pd.concat([
                            dfs[fold][kp],
                            pd.DataFrame({
                                'input_crop_loc': [bl_input_save_path, br_input_save_path],
                                'target_crop_loc': [bl_target_save_path, br_target_save_path]})
                        ])

                else:
                    raise ValueError(f'Unexpected final hash digit: {h[-1]}')

            elif CROP_SIZE[0] == 256:
                if h[-1] in TRAIN_SET_DIGITS:
                    fold = 'train'
                elif h[-1] in VAL_SET_DIGITS:
                    fold = 'val'
                elif h[-1] in TEST_SET_DIGITS:
                    fold = 'test'
                else:
                    raise ValueError(f'Unexpected final hash digit: {h[-1]}')

                # Take the 512x512 five-crop, keeping only bottom-left and bottom-right corners
                input_crops, target_crops = transforms_v2.FiveCrop(size=(512, 512))(img, outputs)
                input_crops_bl = input_crops[2]
                input_crops_br = input_crops[3]
                target_crops_bl = {kp: crops[2] for kp, crops in target_crops.items()}
                target_crops_br = {kp: crops[3] for kp, crops in target_crops.items()}

                for kp in TARGET_LABELS:
                    if kp not in target_crops:
                        continue
                    # Exploit the fact that D/L labels always appear in the bottom-left corner of the original image
                    # (C/R appear in bottom right)
                    if kp.startswith('L') or kp.startswith('D'):
                        my_input_crops = input_crops_bl
                        my_target_crops = target_crops_bl.copy()
                    elif kp.startswith('C') or kp.startswith('R'):
                        my_input_crops = input_crops_br
                        my_target_crops = target_crops_br.copy()

                    my_input_crops, my_target_crops = train_transforms_256(my_input_crops, my_target_crops) if fold == 'train' else \
                                                val_transforms(my_input_crops, my_target_crops)
                    for c in range(5):
                        crop_fname = f'{fname_split_ext[0]}.{c}{fname_split_ext[1]}'
                        input_save_path = os.path.join(SAVE_ROOT_DIR, fold, 'input', kp, crop_fname)
                        my_input_crops[c].save(input_save_path)
                        target_save_path = os.path.join(SAVE_ROOT_DIR, fold, 'target', kp, crop_fname)
                        my_target_crops[kp][c].save(target_save_path)

                        # record the crop locations in the relevant dataframe
                        dfs[fold][kp] = pd.concat([
                            dfs[fold][kp],
                            pd.DataFrame({
                                'input_crop_loc': [input_save_path],
                                'target_crop_loc': [target_save_path]})
                        ])

            else:
                raise ValueError(f"Don't know how to handle CROP_SIZE: {CROP_SIZE}")

    for loc in ['train', 'val', 'test']:
        for label in TARGET_LABELS:
            dfs[loc][label].to_csv(os.path.join(SAVE_ROOT_DIR, loc, f'image_paths_{label}.csv'), index=False)


if __name__ == "__postfix__":
    # This is needed as some of the input crops failed to save correctly during the __main__ logic

    EXPERIMENTS_DIR = os.path.expanduser('~/cv-building-timelapse/data/experiments')
    CSV_FILE = os.path.join(EXPERIMENTS_DIR, 'test/image_paths_R1.csv')

    df_orig = pd.read_csv(CSV_FILE)
    new_rows = []

    for i, row in df_orig.iterrows():
        if os.path.exists(row['input_crop_loc']) and os.path.exists(row['target_crop_loc']):
            new_rows.append(row)
        else:
            print(f"Dropping missing image(s): {row['input_crop_loc']} or {row['target_crop_loc']}")

    df_fix = pd.DataFrame(new_rows)
    new_filename = os.path.splitext(CSV_FILE)[0] + '_clean.csv'
    df_fix.to_csv(new_filename, index=False)


if __name__ == "__postmerge__":

    EXPERIMENTS_DIR = os.path.expanduser('~/cv-building-timelapse/data/experiments/512')
    keypoints_to_group = ['R1', 'R2', 'R3', 'R4']
    crop_type_to_accept = '.1.'
    known_zero_image = os.path.join(EXPERIMENTS_DIR, 'train/target/R4/00acd4db-PXL_20210925_212647637.0.jpg')
    CSV_FILE_IN = os.path.join(EXPERIMENTS_DIR, 'val/image_paths_<kp>.csv')
    CSV_FILE_OUT = os.path.join(EXPERIMENTS_DIR, 'val/image_paths_R_group.csv')

    # first collect all known input images
    known_input_images = set()
    dfs = {}
    for kp in keypoints_to_group:
        df = pd.read_csv(CSV_FILE_IN.replace('<kp>', kp))
        known_input_images.update(df['input_crop_loc'].values)
        dfs[kp] = df

    # filter to only acceptable crop type; put them in order
    known_input_images = {img for img in known_input_images if crop_type_to_accept in img}
    known_input_images = sorted(list(known_input_images), key=lambda f: f.split('/')[-1].split('-PXL_')[1])

    # then iterate over all known input images and check if they are present in all dataframes
    rows = []
    for img in known_input_images:
        my_row = (img.replace(os.path.expanduser('~'), '~'),)
        for kp, df in dfs.items():
            if img in df['input_crop_loc'].values:
                target_img = df[df['input_crop_loc'] == img]['target_crop_loc'].values[0]
                my_row += (target_img.replace(os.path.expanduser('~'), '~'),)
            else:
                # for some reason, this target image doesn't exist for this keypoint
                # for simplicity, we'll just copy the known zero matrix, i.e. a completely black image
                new_target_img_name = img.replace('input', os.path.join('target', kp))
                shutil.copyfile(known_zero_image, new_target_img_name)  # copy image
                my_row += (new_target_img_name.replace(os.path.expanduser('~'), '~'),)

        rows.append(my_row)

    df_out = pd.DataFrame(rows, columns=['input_crop_loc'] + [f'{kp}_target_crop_loc' for kp in keypoints_to_group])
    df_out.to_csv(CSV_FILE_OUT, index=False)
