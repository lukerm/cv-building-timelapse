"""
Find images with multiple keypoints labelled and create a JSON file that is similar in structure
to Label Studio exports. These can be read by adjust_imgs.py and potentially Label Studio at a later stage.
"""

import json
import os
from datetime import date
from typing import List

import pandas as pd

from src.utils import extract_date_from_filename

PREDICTIONS_ROOT = '/home/luke/cv-building-timelapse/data/predictions/'
PREDICTIONS_PATH_MAP = {
    'R1': 'R1/004_gpu/coords/',
    'D2': 'D2/025_gpu/coords/',
}

KEYPOINTS = ['D2', 'R1']

IMG_SIZE = (1024, 768)  # lo resolution


def get_label_dataframe(keypoints: List[str] = KEYPOINTS) -> pd.DataFrame:

    label_rows = []
    for kp in keypoints:
        coord_files = sorted(os.listdir(os.path.join(PREDICTIONS_ROOT, PREDICTIONS_PATH_MAP[kp])))
        coord_files = [f for f in coord_files if f.startswith('PXL_')]
        for f in coord_files:
            df = pd.read_csv(os.path.join(PREDICTIONS_ROOT, PREDICTIONS_PATH_MAP[kp], f))
            if len(df) == 0:
                continue
            row = df.iloc[0]
            width = 100 * row['x'] / IMG_SIZE[0]
            height = 100 * row['y'] / IMG_SIZE[1]
            f_jpg = f.replace('csv', 'jpg')
            label_rows.append((f_jpg, width, height, kp))

    return pd.DataFrame(label_rows, columns=['img_filename', 'x_pc', 'y_pc', 'kp'])


if __name__ == "__main__":
    df = get_label_dataframe(keypoints=KEYPOINTS)

    # ensure that we only take images that appear twice
    dups = df.groupby('img_filename')['kp'].count()
    df_dups = df[df['img_filename'].isin(dups[dups > 1].index)]

    # filter on date (newer than 1st September 2021)
    df_dups['date'] = df_dups['img_filename'].apply(extract_date_from_filename)
    df_dups = df_dups[df_dups['date'] >= date(2021, 9, 1)]

    # TODO: remove this logic - it's only for creating test videos. In reality, we want all labels available
    df_dups = df_dups.groupby(['date', 'kp']).first().reset_index()

    out_labels = []
    for img_filename in df_dups['img_filename'].unique():
        my_results = []
        for kp in KEYPOINTS:
            row = df_dups[(df_dups['img_filename'] == img_filename) & (df_dups['kp'] == kp)].iloc[0]
            my_results.append({
                'value': {
                    'x': float(row['x_pc']),
                    'y': float(row['y_pc']),
                    'keypointlabels': [row['kp']],
                }
            })

        out_labels.append({
            'data': {
                'img': img_filename,
            },
            'predictions': [{'result': my_results,}],
        })

    # out_data = [{'predictions': out_labels}]
    save_fname = 'labels_some_errors_D2_R1.json'
    with open(os.path.join(PREDICTIONS_ROOT, save_fname), 'w') as j:
        json.dump(out_labels, j)
