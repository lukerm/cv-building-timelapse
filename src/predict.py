import os
from datetime import datetime
from typing import List, Tuple

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


def load_unet_model(model_path: str) -> torch.nn.Module:
    model = UNet(in_channels=3, out_channels=1).to(device=torch.device('cpu'))
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=torch.device('cpu')))
    model.eval()
    return model


def get_centroids(prediction_batch: torch.Tensor, macro_offset: tuple, offset_map: dict = MICRO_OFFSET_MAP, eps: float = 1e-5, p_cutoff: float = 0.5) -> List[tuple]:
    img_maxima = torch.amax(prediction_batch, dim=(1, 2, 3))
    idx_todo = np.where(img_maxima > p_cutoff)[0]

    coord_predictions = []
    for idx in idx_todo:
        pred_t = prediction_batch[idx][0]
        max_points = np.where(pred_t > torch.max(pred_t) - eps)

        if len(max_points[0]) > 1 or len(max_points[1]) > 1:
            raise ValueError(f'WARNING: multiple maxima detected: {max_points}')
        else:
            max_coords = np.array((max_points[1][0], max_points[0][0]))  # transpose from matrix space to image space

        coord_predictions.append(max_coords + macro_offset + offset_map[idx])

    return coord_predictions


def make_single_prediction_crop256(model: torch.nn.Module, img_fpath: str, img_fname: str, keypoint: str) -> Tuple[torch.Tensor, Tuple[int]]:
    img_fullpath = os.path.join(img_fpath, img_fname)

    normalize_transform, _ = get_image_transforms()
    model_transform = transforms_v2.Compose([transforms_v2.FiveCrop(size=CROP_SIZE), normalize_transform])

    input_image = read_image(img_fullpath)
    _, _, bl, br, _ = transforms_v2.FiveCrop(size=(512, 512))(input_image)
    if keypoint.startswith('L') or keypoint.startswith('D'):
        my_input_crop = bl
        orig_offset = (0, 256)  # offset from original lo-res image
    elif keypoint.startswith('C') or keypoint.startswith('R'):
        my_input_crop = br
        orig_offset = (512, 256)

    img_transformed = model_transform(my_input_crop)
    img_batch = torch.cat([crop.unsqueeze(0) for crop in img_transformed])

    return model(img_batch), orig_offset


if __name__ == "__main__":
    p_cutoff = 0.01
    keypoint = 'D2'
    model = load_unet_model(model_path=MODEL_MAP[keypoint])
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
            prediction_tensor, orig_offset = make_single_prediction_crop256(model=model, img_fpath=LO_RES_FOLDER, img_fname=img_fname, keypoint=keypoint)
            torch.save(prediction_tensor, os.path.join(save_dir, 'tensors', img_fname.replace('.jpg', '.pt')))

            # save centroid coordinates as simple DataFrames
            centroids = get_centroids(prediction_tensor, macro_offset=orig_offset, p_cutoff=p_cutoff)
            pd.DataFrame(
                centroids, columns=['x', 'y']
            ).to_csv(
                os.path.join(save_dir, 'coords', img_fname.replace('.jpg', '.csv')), index=False,
            )
            if len(centroids) == 0:
                no_pred_list.append(img_fname)

            # save visualization with centroids marked
            pixel_offsets = [(0, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]  # looks like a small 'x'
            input_image = read_image(os.path.join(LO_RES_FOLDER, img_fname))
            img_orig = transforms_v2.ToPILImage()(input_image)
            for coord in get_centroids(prediction_tensor, macro_offset=orig_offset, p_cutoff=p_cutoff):
                for pixel_offset in pixel_offsets:
                    img_orig.putpixel(tuple(coord + pixel_offset), (255, 0, 0))
            img_orig.save(os.path.join(save_dir, 'images', img_fname))


            # # 512x512
            # img512 = transforms_v2.ToPILImage()(my_input_crop)
            # for coord in get_centroids(prediction, macro_offset=(0,0), p_cutoff=p_cutoff):
            #     for pixel_offset in pixel_offsets:
            #         img512.putpixel(tuple(coord + pixel_offset), (255, 0, 0))
            # img512.show()

        except Exception as e:
            print(f'Unexpected Error for image {img_fname}: {e}')
            continue



    pd.DataFrame(no_pred_list).to_csv(os.path.join(save_dir, 'coords', 'no_predictions.csv'), index=False)
