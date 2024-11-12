import json
import os

import numpy as np
from PIL import Image

from .video.adjust_imgs import parse_img_filename

LABELS_FNAME = os.path.expanduser('~/cv-building-timelapse/data/labels/project-2-at-2024-10-11-16-11-19bfb516.json')

IMG_LOAD_DIR = os.path.expanduser('~/Pictures/London/.../balcony/construction')
IMG_SAVE_DIR = os.path.expanduser('~/cv-building-timelapse/data/gauss_viz/v1')
os.makedirs(IMG_SAVE_DIR, exist_ok=True)


if __name__ == "__main__":

    with open(LABELS_FNAME) as j:
        labels = json.load(j)

    labels = [label for label in labels if label['annotations'][0]['result']]

    for i, label in enumerate(labels):
        if i % 51 != 0:
            continue
        print(f'Processed {i} / {len(labels)} labels')

        img_loc = label['data']['img']
        img_orig_path = os.path.join(IMG_LOAD_DIR, parse_img_filename(img_loc))

        with (Image.open(img_orig_path) as img):

            gauss_matrix = np.zeros((img.height, img.width))
            for annotation in label['annotations'][0]['result']:
                x_lab = int(annotation['value']['x'] / 100 * img.width)
                y_lab = int(annotation['value']['y'] / 100 * img.height)
                sigma = 25  # larger than it would be in training for visualization purposes

                # Create a grid of x, y values
                x = np.arange(0, img.width, 1)
                y = np.arange(0, img.height, 1)
                xx, yy = np.meshgrid(x, y)

                # Create a Gaussian distribution
                zz = np.exp(-((xx - x_lab) ** 2 + (yy - y_lab) ** 2) / (2 * sigma ** 2))

                # Normalize the distribution
                zz = (zz - zz.min()) / (zz.max() - zz.min())
                zz = zz * 255

                gauss_matrix += zz

            # Normalize the distribution
            gauss_matrix = (gauss_matrix - gauss_matrix.min()) / (gauss_matrix.max() - gauss_matrix.min())
            gauss_matrix = gauss_matrix * 255
            # Ensure that all pixels have min opacity to be visible
            gauss_matrix = np.maximum(50, gauss_matrix)

            alpha_mask = Image.fromarray(gauss_matrix).convert('L')  # has to be in grayscale / L mode
            img.putalpha(alpha_mask)
            # save as PNG to preserve alpha channel
            img.save(
                os.path.join(IMG_SAVE_DIR, parse_img_filename(img_loc).replace('.jpg', '.png'))
            )
