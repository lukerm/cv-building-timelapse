import json
import os
from math import atan2, cos, sin, pi
from typing import List

from PIL import Image, ImageDraw


LABELS_FNAME = os.path.expanduser('~/cv-building-timelapse/data/predictions/labels_pretty_good_DL_R_groups_2024-11-15.json')
LABELS_CHOSEN_FNAME = os.path.expanduser('~/cv-building-timelapse/data/predictions/chosen_predictions_2024-11-20.json')

IMG_LOAD_DIR = os.path.expanduser('~/Pictures/London/.../balcony/construction')
IMG_SAVE_DIR = os.path.expanduser('~/cv-building-timelapse/data/adjust_translated_rotated/v5_preds_v2/')
os.makedirs(IMG_SAVE_DIR, exist_ok=True)

# Whether to create a parallel version of the image with a D1-R1 guideline and the date
ANNOTATED_COPY = True

# inferred from labels of: PXL_20220621_052524018.jpg
TARGET_LOCS = {
    'L2': (140, 2148),
    'D1': (1007, 1957),
    'D2': (914, 2049),
    'D3': (897, 2147),
    'R1': (2669, 1885),
    'R2': (2933, 2192),
    'R3': (3207, 1978),
    'R4': (3522, 1917),
}


def parse_img_filename(path: str) -> str:
    old_fname = path.split('/')[-1]
    sep = 'PXL_'
    new_fname = sep + old_fname.split(sep)[1]
    return new_fname


def get_restricted_photo_list(photos_json_fname: str) -> List[str]:
    with open(photos_json_fname) as j:
        photos = json.load(j)

    return photos


if __name__ == '__main__':

    if ANNOTATED_COPY:
        os.makedirs(IMG_SAVE_DIR[:-1] + 'a', exist_ok=True)

    with open(LABELS_FNAME) as j:
        labels = json.load(j)

    labels = [label for label in labels if label.get('annotations', label['predictions'])[0]['result']]

    photos_json_fname = LABELS_CHOSEN_FNAME
    if photos_json_fname is not None:
        photos = get_restricted_photo_list(photos_json_fname)
        labels = [label for label in labels if label['data']['img'] in photos]

    rerun_list = []  # None or [] to ignore this
    if rerun_list:
        labels = [label for label in labels if label['data']['img'] in rerun_list]

    for i, label in enumerate(labels):
        if i > 0 and i % 100 == 0:
            print(f'Processed {i} / {len(labels)} labels')

        img_loc = label['data']['img']

        img_orig_path = os.path.join(IMG_LOAD_DIR, parse_img_filename(img_loc))
        with Image.open(img_orig_path) as img:

            # translation: most confidently predicted point from right-hand group
            right_results = [res for res in label.get('annotations', label['predictions'])[0]['result'] if not res['value']['is_left']]
            right_confident_result = sorted(right_results, key=lambda x: x['value']['p'], reverse=True)[0]
            kp_label_right = right_confident_result['value']['keypointlabels'][0]

            label_right_actual_loc = (
                int(img.width * right_confident_result['value']['x'] / 100),
                int(img.height * right_confident_result['value']['y'] / 100)
            )
            x_trans = label_right_actual_loc[0] - TARGET_LOCS[kp_label_right][0]
            y_trans = label_right_actual_loc[1] - TARGET_LOCS[kp_label_right][1]

            # rotation: rotate according to most confidently predicted point from left-hand group
            left_results = [res for res in label.get('annotations', label['predictions'])[0]['result'] if res['value']['is_left']]
            left_confident_result = sorted(left_results, key=lambda x: x['value']['p'], reverse=True)[0]
            kp_label_left = left_confident_result['value']['keypointlabels'][0]

            label_left_actual_loc = (
                int(img.width * left_confident_result['value']['x'] / 100),
                int(img.height * left_confident_result['value']['y'] / 100)
            )
            target_angle = atan2( (TARGET_LOCS[kp_label_right][1] - TARGET_LOCS[kp_label_left][1]) , (TARGET_LOCS[kp_label_right][0] - TARGET_LOCS[kp_label_left][0]))
            actual_angle = atan2( (label_right_actual_loc[1] - label_left_actual_loc[1]) , (label_right_actual_loc[0] - label_left_actual_loc[0]))
            rotation_angle = actual_angle - target_angle
            rotation_angle = rotation_angle * 180 / pi  # convert to degrees
            img_transformed = img.transform(
                img.size, Image.AFFINE, (
                    1, 0, x_trans,
                    0, 1, y_trans,
                )
            ).rotate(
                rotation_angle, center=TARGET_LOCS[kp_label_right],
            )
            img_transformed.save(os.path.join(IMG_SAVE_DIR, parse_img_filename(img_loc)))

            # truth annotations
            if ANNOTATED_COPY:
                draw = ImageDraw.Draw(img_transformed)
                draw.line([TARGET_LOCS['D1'], TARGET_LOCS['R1']], width=5, fill='red')
                draw.text((50, 50), parse_img_filename(img_loc).split('_')[1], fill='red', font_size=50)

                img_transformed.save(os.path.join(IMG_SAVE_DIR[:-1] + 'a', parse_img_filename(img_loc)))
