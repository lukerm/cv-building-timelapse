"""
Find images with multiple keypoints labelled and create a JSON file that is similar in structure
to Label Studio exports. These can be read by adjust_imgs.py and potentially Label Studio at a later stage.
"""

import json
import os
from datetime import date
from typing import Dict

import pandas as pd

from src.utils import extract_date_from_filename

PREDICTIONS_ROOT = '/home/luke/cv-building-timelapse/data/predictions/'
PREDICTIONS_PATH_MAP = {
    'R_group': 'R_group/507_gpu/coords/',
    'DL_group': 'DL_group/530_gpu/coords/',
}

DL_GROUP_KEYPOINTS = {0:'D1', 1:'D2', 2:'D3', 3:'L2'}
R_GROUP_KEYPOINTS = {0:'R1', 1:'R2', 2:'R3', 3:'R4'}

IMG_SIZE = (1024, 768)  # lo resolution


def get_label_dataframe(keypoints_left: Dict[int, str], keypoints_right: Dict[int, str]) -> pd.DataFrame:

    label_rows = []
    for kp_group in PREDICTIONS_PATH_MAP.keys():
        is_left = kp_group == 'DL_group'  # False means right
        coord_files = sorted(os.listdir(os.path.join(PREDICTIONS_ROOT, PREDICTIONS_PATH_MAP[kp_group])))
        coord_files = [f for f in coord_files if f.startswith('PXL_')]
        for f in coord_files:
            df = pd.read_csv(os.path.join(PREDICTIONS_ROOT, PREDICTIONS_PATH_MAP[kp_group], f))
            df = df.sort_values('p', ascending=False)
            for _, row in df.iterrows():
                f_jpg = f.replace('csv', 'jpg')
                width = 100 * row['x'] / IMG_SIZE[0]
                height = 100 * row['y'] / IMG_SIZE[1]
                prob = row['p']
                kp = keypoints_left[row['idx']] if is_left else keypoints_right[row['idx']]
                label_rows.append((f_jpg, width, height, prob, kp, is_left))

    return pd.DataFrame(label_rows, columns=['img_filename', 'x_pc', 'y_pc', 'prob', 'kp', 'is_left'])


if __name__ == "__main__":
    df = get_label_dataframe(keypoints_left=DL_GROUP_KEYPOINTS, keypoints_right=R_GROUP_KEYPOINTS)

    # ensure that we only take images that appear twice
    dups = df.groupby('img_filename')['kp'].count()
    df_dups = df[df['img_filename'].isin(dups[dups > 1].index)]

    # filter on date (newer than 1st September 2021)
    df_dups['date'] = df_dups['img_filename'].apply(extract_date_from_filename)
    df_dups = df_dups[df_dups['date'] >= date(2021, 9, 1)]

    out_labels = []
    for img_filename in df_dups['img_filename'].unique():
        my_results = []
        my_df = df_dups[df_dups['img_filename'] == img_filename]
        for _, row in my_df.iterrows():
            my_results.append({
                'value': {
                    'x': float(row['x_pc']),
                    'y': float(row['y_pc']),
                    'p': float(row['prob']),
                    'keypointlabels': [row['kp']],
                    'is_left': row['is_left'],
                }
            })

        out_labels.append({
            'data': {
                'img': img_filename,
            },
            'predictions': [{'result': my_results,}],
        })

    save_fname = 'labels_pretty_good_DL_R_groups_2024-11-15.json'
    with open(os.path.join(PREDICTIONS_ROOT, save_fname), 'w') as j:
        json.dump(out_labels, j)
