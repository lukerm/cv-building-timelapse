import json
import os
from math import atan2, cos, sin, pi

from PIL import Image


LABELS_FNAME = os.path.expanduser('~/cv-building-timelapse/data/labels/project-2-at-2024-10-09-09-02-e4d48c49.json')

IMG_LOAD_DIR = os.path.expanduser('~/Pictures/London/.../balcony/construction')
IMG_SAVE_DIR = os.path.expanduser('~/cv-building-timelapse/data/adjust_translated_rotated/v3')
os.makedirs(IMG_SAVE_DIR, exist_ok=True)

TARGET_LOCS = {
    'D2': (907, 2045),
    'R1': (2672, 1896),
}


def parse_img_filename(path: str) -> str:
    old_fname = path.split('/')[-1]
    sep = 'PXL_'
    new_fname = sep + old_fname.split(sep)[1]
    return new_fname






if __name__ == '__main__':

    with open(LABELS_FNAME) as j:
        labels = json.load(j)

    labels = [label for label in labels if label['annotations'][0]['result']]

    for i, label in enumerate(labels):
        if i > 0 and i % 100 == 0:
            print(f'Processed {i} / {len(labels)} labels')

        img_loc = label['data']['img']

        img_orig_path = os.path.join(IMG_LOAD_DIR, parse_img_filename(img_loc))
        with Image.open(img_orig_path) as img:

            # translation
            kp_label = 'R1'
            r1_result = [res for res in label['annotations'][0]['result'] if kp_label in res['value']['keypointlabels']]
            label_r1_actual_loc = (
                int(img.width * r1_result[0]['value']['x'] / 100),
                int(img.height * r1_result[0]['value']['y'] / 100)
            )
            x_trans = label_r1_actual_loc[0] - TARGET_LOCS[kp_label][0]
            y_trans = label_r1_actual_loc[1] - TARGET_LOCS[kp_label][1]

            # rotation
            kp_label = 'D2'
            d2_result = [res for res in label['annotations'][0]['result'] if kp_label in res['value']['keypointlabels']]
            label_d2_actual_loc = (
                int(img.width * d2_result[0]['value']['x'] / 100),
                int(img.height * d2_result[0]['value']['y'] / 100)
            )
            target_angle = atan2( (TARGET_LOCS['R1'][1] - TARGET_LOCS['D2'][1]) , (TARGET_LOCS['R1'][0] - TARGET_LOCS['D2'][0]))
            actual_angle = atan2( (label_r1_actual_loc[1] - label_d2_actual_loc[1]) , (label_r1_actual_loc[0] - label_d2_actual_loc[0]))
            rotation_angle = actual_angle - target_angle
            rotation_angle = rotation_angle * 180 / pi  # convert to degrees

            img.transform(
                img.size, Image.AFFINE, (
                    1, 0, x_trans,
                    0, 1, y_trans,
                )
            ).rotate(
                rotation_angle, center=TARGET_LOCS['R1'],
            ).save(
                os.path.join(IMG_SAVE_DIR, parse_img_filename(img_loc))
            )
