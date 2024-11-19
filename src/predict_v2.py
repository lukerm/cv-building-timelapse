"""
Run keypoint detection both grouped models for ALL images in the HI_RES_FOLDER directory. The saved images will plot
predicted centroids from both models on the original hi-res images.
Note: this runs off the back off outputs from predict.py (version 1) to avoid expensive re-calculation of prediction
        tensors.
"""


import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torchvision.transforms.v2 as transforms_v2
from torchvision.io import read_image

from src.config import HI_RES_FOLDER, HI_RES_SIZE, LO_RES_SIZE

# We make use of existing coordinate predictions in these directories
COORD_LOAD_DIRS = {
    'left': 'DL_group/530_gpu/coords/',
    'right': 'R_group/507_gpu/coords/',
}

PREDS_SAVE_ROOT = os.path.expanduser('~/cv-building-timelapse/data/predictions')


def lo_to_hi_res_translation(lo_res_coords: tuple, lo_res_shape: tuple, hi_res_shape: tuple) -> tuple:
    lo_res_x, lo_res_y = lo_res_coords
    lo_res_w, lo_res_h = lo_res_shape
    hi_res_w, hi_res_h = hi_res_shape

    hi_res_x = lo_res_x * hi_res_w / lo_res_w
    hi_res_y = lo_res_y * hi_res_h / lo_res_h

    return int(hi_res_x), int(hi_res_y)


if __name__ == "__main__":
    keypoint_group_left = ['DL_group']
    keypoint_group_right = ['R_group']

    out_channels = 4
    p_cutoff = 0.05

    save_dir = os.path.join(PREDS_SAVE_ROOT, 'left_right_530_507')
    for dir in ['coords', 'images']:
        os.makedirs(os.path.join(save_dir, dir), exist_ok=True)

    img_fnames = sorted(os.listdir(HI_RES_FOLDER))
    no_pred_list = []

    MAX_IDX = 5_000_000
    for i, img_fname in enumerate(img_fnames[:MAX_IDX]):
        if i % 100 == 0:
            t = datetime.now()
            print(f'({t}): Processed {i} / {len(img_fnames)} images')

        left_pred_fname = os.path.join(PREDS_SAVE_ROOT, COORD_LOAD_DIRS['left'], img_fname.replace('.jpg', '.csv'))
        right_pred_fname = os.path.join(PREDS_SAVE_ROOT, COORD_LOAD_DIRS['right'], img_fname.replace('.jpg', '.csv'))

        if not os.path.exists(left_pred_fname) or not os.path.exists(right_pred_fname):
            no_pred_list.append(img_fname)
            continue

        df_preds = {
            'left': pd.read_csv(left_pred_fname).sort_values(by='p', ascending=False).reset_index(drop=True),
            'right': pd.read_csv(right_pred_fname).sort_values(by='p', ascending=False).reset_index(drop=True),
        }
        df_preds['left'] = df_preds['left'][df_preds['left']['p'] > p_cutoff]
        df_preds['right'] = df_preds['right'][df_preds['right']['p'] > p_cutoff]
        df_preds['left']['side'] = 'left'
        df_preds['right']['side'] = 'right'
        df_preds_all = pd.concat([df_preds['left'], df_preds['right']], ignore_index=True)
        df_preds_all.to_csv(os.path.join(save_dir, 'coords', img_fname.replace('.jpg', '.csv')), index=False)

        input_image = read_image(os.path.join(HI_RES_FOLDER, img_fname))
        img_orig = transforms_v2.ToPILImage()(input_image)

        # save visualization with centroids marked
        # looks like a small 'x' (adjust x_radius for larger/smaller 'x')
        x_radius = 14
        pixel_offsets = [[
            k * np.array([-1, -1]),
            k * np.array([1, -1]),
            k * np.array([-1, 1]),
            k * np.array([1, 1]),
        ] for k in range(x_radius + 1)]
        pixel_offsets = list(set([tuple(coord.tolist()) for sublist in pixel_offsets for coord in sublist]))  # flatten & uniquify
        pixel_offsets_surroundings = [np.array([k, 0]) for k in range(-2, 3)]  # extra width

        for side in ['left', 'right']:
            for r, row in df_preds[side].iterrows():
                x, y, p, idx = row['x'], row['y'], row['p'], row['idx']
                x, y = lo_to_hi_res_translation((x, y), LO_RES_SIZE, HI_RES_SIZE)
                for pixel_offset in pixel_offsets:
                    for pixel_offsets_surrounding in pixel_offsets_surroundings:
                        try:
                            img_orig.putpixel(
                                tuple(np.array([x, y]) + pixel_offset + pixel_offsets_surrounding),
                                (31, 198, 0) if r == 0 else (255, 0, 0)
                            )
                        # likely due to being near image's edge
                        except IndexError:
                            continue

        # save image only if there is a reasonable prediction
        if len(df_preds_all) > 0:
            img_orig.save(os.path.join(save_dir, 'images', img_fname))
            time.sleep(0.05)

    pd.DataFrame(no_pred_list).to_csv(os.path.join(save_dir, 'coords', 'no_predictions.csv'), index=False)
    print('Finished')
