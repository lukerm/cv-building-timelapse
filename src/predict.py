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

"""Run keypoint detection model for ALL images in the LO_RES_FOLDER directory"""

import os
import time
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.v2 as transforms_v2
from torchvision.io import read_image

from src.config import CROP_SIZE, LO_RES_FOLDER, MODEL_MAP
from src.train.train_utils import get_image_transforms, UNet

# These are the offsets within the 512x512 crop
MICRO_OFFSET_MAP = {
    0: (0, 0),  # top left
    1: (256, 0),  # top right
    2: (0, 256),  # bottom left
    3: (256, 256),  # bottom right
    4: (128, 128),  # centre
}

PREDS_SAVE_ROOT = os.path.expanduser('~/cv-building-timelapse/data/predictions')
os.makedirs(PREDS_SAVE_ROOT, exist_ok=True)


def load_unet_model(model_path: str, out_channels: int = 1) -> torch.nn.Module:
    model = UNet(in_channels=3, out_channels=out_channels).to(device=torch.device('cpu'))
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=torch.device('cpu')))
    model.eval()
    return model


def get_centroids(
    prediction_tensor: torch.Tensor, macro_offset: tuple, offset_map: dict = MICRO_OFFSET_MAP,
    eps: float = 1e-5, p_cutoff: float = 0.5, restrict_idx: List[int] = None,
) -> List[tuple]:

    img_maxima = torch.amax(prediction_tensor, dim=(1, 2, 3))
    idx_todo = np.where(img_maxima > p_cutoff)[0]

    if restrict_idx is not None:
        idx_todo = sorted(list(set(restrict_idx).intersection(set(idx_todo))))

    coord_predictions = []
    for idx in idx_todo:
        pred_t = prediction_tensor[idx][0]
        max_points = np.where(pred_t > torch.max(pred_t) - eps)

        if len(max_points[0]) > 1 or len(max_points[1]) > 1:
            max_coords = np.array((int(np.mean(max_points[1])), int(np.mean(max_points[0]))))  # transpose from matrix space to image space
        else:
            max_coords = np.array((max_points[1][0], max_points[0][0]))  # transpose from matrix space to image space

        coord_predictions.append((max_coords + macro_offset + offset_map[idx], img_maxima[idx].detach().numpy(), idx))

    return coord_predictions


def get_centroids_512(
    prediction_tensor: torch.Tensor, macro_offset: tuple, eps: float = 1e-5,
) -> List[tuple]:

    img_maxima = torch.amax(prediction_tensor, dim=(0, 2, 3))  # note: assumes one in batch

    coord_predictions = []
    for idx in range(len(img_maxima)):
        pred_t = prediction_tensor[0][idx]
        max_points = np.where(pred_t > torch.max(pred_t) - eps)

        if len(max_points[0]) > 1 or len(max_points[1]) > 1:
            max_coords = np.array((int(np.mean(max_points[1])), int(np.mean(max_points[0]))))  # transpose from matrix space to image space
        else:
            max_coords = np.array((max_points[1][0], max_points[0][0]))  # transpose from matrix space to image space

        coord_predictions.append((max_coords + macro_offset, img_maxima[idx].detach().numpy(), idx))

    return coord_predictions


def make_single_prediction_crop256(model: torch.nn.Module, img_fpath: str, img_fname: str, keypoint: str) -> torch.Tensor:
    img_fullpath = os.path.join(img_fpath, img_fname)

    normalize_transform, _ = get_image_transforms()
    model_transform = transforms_v2.Compose([transforms_v2.FiveCrop(size=CROP_SIZE), normalize_transform])

    input_image = read_image(img_fullpath)
    _, _, bl, br, _ = transforms_v2.FiveCrop(size=(512, 512))(input_image)
    if keypoint.startswith('L') or keypoint.startswith('D'):
        my_input_crop = bl
    elif keypoint.startswith('C') or keypoint.startswith('R'):
        my_input_crop = br

    img_transformed = model_transform(my_input_crop)
    img_batch = torch.cat([crop.unsqueeze(0) for crop in img_transformed])

    return model(img_batch)


def make_single_prediction_crop512(model: torch.nn.Module, img_fpath: str, img_fname: str, keypoint: str) -> torch.Tensor:
    img_fullpath = os.path.join(img_fpath, img_fname)
    input_image = read_image(img_fullpath).unsqueeze(0)
    _, _, bl, br, _ = transforms_v2.FiveCrop(size=(512, 512))(input_image)
    if keypoint.startswith('L') or keypoint.startswith('D'):
        my_input_crop = bl
    elif keypoint.startswith('C') or keypoint.startswith('R'):
        my_input_crop = br

    normalize_transform, _ = get_image_transforms()
    img_transformed = normalize_transform(my_input_crop)  # only normalization, no need for second FiveCrop op

    return model(img_transformed)


if __name__ == "__main__":
    keypoint = 'R_group'  # TODO: configure
    out_channels = 4
    p_cutoff = 0.01  # cut-off values can be surprisingly low for some models

    model = load_unet_model(model_path=MODEL_MAP[keypoint], out_channels=out_channels)
    experiment_id = MODEL_MAP[keypoint].split('/')[-2]

    save_dir = os.path.join(PREDS_SAVE_ROOT, keypoint, experiment_id)
    for dir in ['tensors', 'coords', 'images']:
        os.makedirs(os.path.join(save_dir, dir), exist_ok=True)

    img_fnames = sorted(os.listdir(LO_RES_FOLDER))
    no_pred_list = []

    MAX_IDX = 5_000_000
    for i, img_fname in enumerate(img_fnames[:MAX_IDX]):
        if i % 100 == 0:
            t = datetime.now()
            print(f'({t}): Processed {i} / {len(img_fnames)} images')

        try:
            # save the raw prediction tensor
            tensor_save_fpath = os.path.join(save_dir, 'tensors', img_fname.replace('.jpg', '.pt'))
            if not os.path.exists(tensor_save_fpath):
                if CROP_SIZE[0] == 256:
                    prediction_tensor = make_single_prediction_crop256(model=model, img_fpath=LO_RES_FOLDER, img_fname=img_fname, keypoint=keypoint)
                elif CROP_SIZE[0] == 512:
                    prediction_tensor = make_single_prediction_crop512(model=model, img_fpath=LO_RES_FOLDER, img_fname=img_fname, keypoint=keypoint)
                torch.save(prediction_tensor, tensor_save_fpath)
            else:
                prediction_tensor = torch.load(tensor_save_fpath, weights_only=True)

            # calculate & save centroid coordinates as simple DataFrames
            if keypoint.startswith('L') or keypoint.startswith('D'):
                orig_offset = (0, 256)  # offset from original lo-res image (for 512x512 crop)
            elif keypoint.startswith('C') or keypoint.startswith('R'):
                orig_offset = (512, 256)

            if CROP_SIZE[0] == 256:
                restrict_idx = [0] if keypoint == 'D2' else None
                centroids = get_centroids(prediction_tensor, macro_offset=orig_offset, p_cutoff=p_cutoff, restrict_idx=restrict_idx)
            elif CROP_SIZE[0] == 512:
                centroids = get_centroids_512(prediction_tensor, macro_offset=orig_offset)

            centroids = [(int(x), int(y), float(p), idx) for (x, y), p, idx in centroids]  # unpack
            pd.DataFrame(
                centroids, columns=['x', 'y', 'p', 'idx']
            ).to_csv(
                os.path.join(save_dir, 'coords', img_fname.replace('.jpg', '.csv')), index=False,
            )
            centroids = [(x, y, p, idx) for x, y, p, idx in centroids if p > p_cutoff]  # remove low-confidence predictions
            if len(centroids) == 0:
                no_pred_list.append(img_fname)

            if keypoint in ['D2']:
                # ad-hoc analysis showed that only the 0th (top-left) crop gave non-spurious results for D2
                # Note: they are still recorded in the CSV in case you want to refer back later
                centroids = centroids[:1]

            # save visualization with centroids marked
            # looks like a small 'x' (adjust x_radius for larger/smaller 'x')
            x_radius = 4
            pixel_offsets = [[
                k * np.array([-1, -1]),
                k * np.array([1, -1]),
                k * np.array([-1, 1]),
                k * np.array([1, 1]),
            ] for k in range(x_radius + 1)]
            pixel_offsets = list(set([tuple(coord.tolist()) for sublist in pixel_offsets for coord in sublist]))  # flatten & uniquify

            input_image = read_image(os.path.join(LO_RES_FOLDER, img_fname))
            img_orig = transforms_v2.ToPILImage()(input_image)

            for coord in centroids:
                for pixel_offset in pixel_offsets:
                    try:
                        img_orig.putpixel(
                            tuple(np.array(coord[:2]) + pixel_offset),
                            (0, 0, 255) if coord[3] == 0 else (255, 0, 0)
                        )
                    # likely due to being near image's edge
                    except IndexError:
                        continue

            # save image only if there is a reasonable prediction
            if len(centroids) > 0:
                img_orig.save(os.path.join(save_dir, 'images', img_fname))
                time.sleep(0.05)

            # # 512x512
            # img512 = transforms_v2.ToPILImage()(my_input_crop)
            # for coord in get_centroids(prediction_tensor, macro_offset=(0,0), p_cutoff=p_cutoff):
            #     for pixel_offset in pixel_offsets:
            #         img512.putpixel(tuple(coord + pixel_offset), (255, 0, 0))
            # img512.show()

        except Exception as e:
            print(f'Unexpected Error for image {img_fname}: {e}')
            raise

    pd.DataFrame(no_pred_list).to_csv(os.path.join(save_dir, 'coords', 'no_predictions.csv'), index=False)
