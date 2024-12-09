import os

from PIL import Image


IMG_LOAD_DIR = os.path.expanduser('~/cv-building-timelapse/data/adjust_translated_rotated/v6_preds_v3')
IMG_SAVE_DIR = os.path.expanduser('~/cv-building-timelapse/data/adjust_translated_rotated_cropped/v2/')
os.makedirs(IMG_SAVE_DIR, exist_ok=True)

# (left, upper, right, lower)
# see: scratch file to see how to get these values (v4)
CROP_DIMS = (205, 271, 3725, 2911)


if __name__ == "__main__":

    MAX_ID = 42_000_000
    img_filenames_all = sorted([f for f in os.listdir(IMG_LOAD_DIR) if f.endswith('.jpg')])[:MAX_ID]
    for i, img_filename in enumerate(img_filenames_all):
        if i > 0 and i % 100 == 0:
            print(f'Processed {i} / {len(img_filenames_all)} images')

        # ignoring those in Sep/Oct 2021 .. planning to skip those months in final video
        skip = False
        skip_months = ['202108', '202109', '202110']
        for skip_month in skip_months:
            if skip_month in img_filename:
                skip = True

        if skip:
            continue

        with Image.open(os.path.join(IMG_LOAD_DIR, img_filename)) as img:
            img = img.crop(CROP_DIMS)
            img.save(os.path.join(IMG_SAVE_DIR, img_filename))
