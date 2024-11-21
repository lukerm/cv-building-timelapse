import os.path
import shutil

import numpy as np
from PIL import Image, ImageDraw


def right_to_left_finder(pixels, thresh: int = 10, img_height: int = 3024):
    for j in range(pixels.shape[1] - 1, -1, -1):
        if pixels[:, j].max() > thresh:
            i_idx = np.where(pixels[:, j] > thresh)[0]
            if np.mean(i_idx) > img_height / 2:
                corner = (i_idx.max(), j)
            else:
                corner = (i_idx.min(), j)
            corner = (int(corner[0]), int(corner[1]))  # cast to py ints
            return corner[::-1]

def left_to_right_finder(pixels, thresh: int = 10, img_height: int = 3024):
    for j in range(pixels.shape[1]):
        if pixels[:, j].max() > thresh:
            i_idx = np.where(pixels[:, j] > thresh)[0]
            if np.mean(i_idx) > img_height / 2:
                corner = (i_idx.max(), j)
            else:
                corner = (i_idx.min(), j)
            corner = (int(corner[0]), int(corner[1]))  # cast to py ints
            return corner[::-1]

def top_to_bottom_finder(pixels, thresh: int = 10, img_width: int = 4032):
    for i in range(pixels.shape[0]):
        if pixels[i, :].max() > thresh:
            j_idx = np.where(pixels[i, :] > thresh)[0]
            if np.mean(j_idx) > img_width / 2:
                corner = (i, j_idx.max())
            else:
                corner = (i, j_idx.min())
            corner = (int(corner[0]), int(corner[1]))  # cast to py ints
            return corner[::-1]


def bottom_to_top_finder(pixels, thresh: int = 10, img_width: int = 4032):
    for i in range(pixels.shape[0] - 1, -1, -1):
        if pixels[i, :].max() > thresh:
            j_idx = np.where(pixels[i, :] > thresh)[0]
            if np.mean(j_idx) > img_width / 2:
                corner = (i, j_idx.max())
            else:
                corner = (i, j_idx.min())
            corner = (int(corner[0]), int(corner[1]))  # cast to py ints
            return corner[::-1]


def get_rectangular_hull(corners, img_width: int = 4032, img_height: int = 3024):

    left_points = [c for c in corners if c[0] < img_width / 2]
    right_points = [c for c in corners if c[0] >= img_width / 2]
    top_points = [c for c in corners if c[1] < img_height / 2]
    bottom_points = [c for c in corners if c[1] >= img_height / 2]

    top_left = (max([c[0] for c in left_points]), max([c[1] for c in top_points]))
    top_right = (min([c[0] for c in right_points]), max([c[1] for c in top_points]))
    bottom_left = (max([c[0] for c in left_points]), min([c[1] for c in bottom_points]))
    bottom_right = (min([c[0] for c in right_points]), min([c[1] for c in bottom_points]))

    return {
        'top_left': top_left,
        'top_right': top_right,
        'bottom_left': bottom_left,
        'bottom_right': bottom_right,
    }


def process_image_corners(img_fname: str):

    orig_pil = Image.open(img_fname)
    img_grayscale = orig_pil.convert('L')
    pixels = np.array(img_grayscale)

    thresh_set = [10, 5, 2, 1, 0]

    for thresh in thresh_set:
        try:
            rl_corner = right_to_left_finder(pixels, thresh=thresh)
        except ValueError:
            print(f'thresh {thresh} failed')
            continue
        break

    for thresh in thresh_set:
        try:
            lr_corner = left_to_right_finder(pixels, thresh=thresh)
        except ValueError:
            print(f'thresh {thresh} failed')
            continue
        break

    for thresh in thresh_set:
        try:
            tb_corner = top_to_bottom_finder(pixels, thresh=thresh)
        except ValueError:
            print(f'thresh {thresh} failed')
            continue
        break

    for thresh in thresh_set:
        try:
            bt_corner = bottom_to_top_finder(pixels, thresh=thresh)
        except ValueError:
            print(f'thresh {thresh} failed')
            continue
        break

    return rl_corner, lr_corner, tb_corner, bt_corner



if __name__ == "__main__":

    save_dir = os.path.expanduser('~/cv-building-timelapse/data/adjust_translated_rotated/v6_preds_v3_rectangles/')
    os.makedirs(save_dir, exist_ok=True)
    load_dir = os.path.expanduser('~/cv-building-timelapse/data/adjust_translated_rotated/v6_preds_v3/')
    img_fnames = sorted([f for f in os.listdir(load_dir) if f.endswith('.jpg')])

    img_size = (4032, 3024)

    # for storing good and bad corner-location inferences
    # 'bad' is when the corner is too far into the image, which we deem erroneous (but needs checking later)
    corner_store = {
        'good': {'top_left': {}, 'top_right': {}, 'bottom_left': {}, 'bottom_right': {}},
        'bad': {'top_left': {}, 'top_right': {}, 'bottom_left': {}, 'bottom_right': {}},
    }

    # derived from statistics of good images (post-run)
    # the aim is to maximise the area of the global blue box whilst avoiding any blacks areas in any frame (which are
    # caused by translation-rotation transformations)
    left_star = 190  # v3*
    right_star = 3740
    top_star = 146
    bottom_star = 2963
    interior_hull = {
        'top_left': (left_star, top_star),
        'top_right': (right_star, top_star),
        'bottom_left': (left_star, bottom_star),
        'bottom_right': (right_star, bottom_star),
    }

    MAX_IDX = 25_000_000
    for i, img_fname in enumerate(img_fnames[:MAX_IDX]):
        if i > 0 and i % 100 == 0:
            print(f'Processed {i} / {len(img_fnames)}')

        # ignoring those in Sep/Oct 2021 .. planning to skip those months in final video
        if '202109' in img_fname or '202110' in img_fname:
            continue

        inferred_corners = process_image_corners(os.path.join(load_dir, img_fname))
        hull = get_rectangular_hull(inferred_corners)

        # Rules for ignoring / declaring the inference bad
        # TODO: note: should really be doing ignore on a corner-by-corner basis
        ignored = False
        pc_ignore = {'left': 0.0469, 'right': 0.0479, 'top': 0.0483, 'bottom': 0.0198}
        # E.g. the minimal left line is more than 4.7% into the interior of the image (deemed too far)
        if hull['top_left'][0] > img_size[0]*pc_ignore['left'] or hull['top_left'][1] > img_size[1]*pc_ignore['top']:
            ignored = True
        elif hull['top_right'][0] < img_size[0]*(1-pc_ignore['right']) or hull['top_right'][1] > img_size[1]*pc_ignore['top']:
            ignored = True
        elif hull['bottom_left'][0] > img_size[0]*pc_ignore['left'] or hull['bottom_left'][1] < img_size[1]*(1-pc_ignore['bottom']):
            ignored = True
        elif hull['bottom_right'][0] < img_size[0]*(1-pc_ignore['right']) or hull['bottom_right'][1] < img_size[1]*(1-pc_ignore['bottom']):
            ignored = True

        orig_pil = Image.open(os.path.join(load_dir, img_fname))
        draw = ImageDraw.Draw(orig_pil)
        draw.polygon([hull['top_left'], hull['top_right'], hull['bottom_right'], hull['bottom_left']], outline='red', width=5)
        draw.polygon([interior_hull['top_left'], interior_hull['top_right'], interior_hull['bottom_right'], interior_hull['bottom_left']], outline='blue', width=5)
        if ignored:
            draw.text((img_size[0]/2, img_size[1]/2), 'X', fill='red', font_size=500)
        orig_pil.save(os.path.join(save_dir, img_fname))


        keyword = 'good' if not ignored else 'bad'
        corner_store[keyword]['top_left'][img_fname] = hull['top_left']
        corner_store[keyword]['top_right'][img_fname] = hull['top_right']
        corner_store[keyword]['bottom_left'][img_fname] = hull['bottom_left']
        corner_store[keyword]['bottom_right'][img_fname] = hull['bottom_right']


    ## post-hoc analysis
    # infer minimal bounding box (based on all data)
    left = max([c[0] for c in corner_store['good']['top_left'].values()])
    left2 = max([c[0] for c in corner_store['good']['bottom_left'].values()])
    assert left == left2
    right = min([c[0] for c in corner_store['bad']['top_right'].values()])
    # right = min([c[0] for c in corner_store['good']['top_right'].values()])
    right2 = min([c[0] for c in corner_store['good']['bottom_right'].values()])
    assert right == right2
    top = max([c[1] for c in corner_store['good']['top_left'].values()])
    top2 = max([c[1] for c in corner_store['good']['top_right'].values()])
    assert top == top2
    bottom = min([c[1] for c in corner_store['good']['bottom_left'].values()])
    bottom2 = min([c[1] for c in corner_store['good']['bottom_right'].values()])
    assert bottom == bottom2

    # initially: left = 186; right = 3831; top = 150; bottom = 2879
    # 2nd round: left = 122; right = 3839; top = 146; bottom = 2963  # v2
    # *note: moving left back to its original position (ish) as 122 is just catching the RHS of the left house
    # note: there were quite a few badly predicted frames that had the right boundary cut off - moved back by ~100 pix
    # 3rd round: left = 190; right = 3740; top = 146; bottom = 2963  # v3



    #####


    from pprint import pp
    # generally using start date 2021-11-01 (sunny)
    pp({k: v for k, v in corner_store['good']['top_left'].items() if v[0] > left - 70})  # left
    # pp({k: v for k, v in corner_store['good']['top_right'].items() if v[0] < right + 10})  # right
    pp({k: v for k, v in corner_store['bad']['top_right'].items() if v[0] < 3800 and 'PXL_20230317' < k < 'PXL_202304019'})  # right
    # PXL_20230326_125915201.jpg => 3839  if keeping frame
    pp({k: v for k, v in corner_store['good']['top_left'].items() if v[1] > top - 15})  # top
    pp({k: v for k, v in corner_store['good']['bottom_left'].items() if v[1] < bottom + 25})  # bottom


    #top
    d = {k: v for k, v in corner_store['good']['top_left'].items() if v[1] > top - 50}
    for img_fname, c in d.items() :
        shutil.copyfile(os.path.join(save_dir, img_fname), os.path.join(save_dir, 'top', f'{int(c[1])}'.zfill(3)  + '_' + img_fname))

    #bottom
    d = {k: v for k, v in corner_store['good']['bottom_left'].items() if v[1] < bottom + 100}
    for img_fname, c in d.items() :
        shutil.copyfile(os.path.join(save_dir, img_fname), os.path.join(save_dir, 'bottom', f'{int(c[1])}'.zfill(3)  + '_' + img_fname))

    # 2963 ?


    # move bad images into a subfolder
    for img_fname in corner_store['bad']['top_left']:
        shutil.move(os.path.join(save_dir, img_fname), os.path.join(save_dir, 'bad', img_fname))


    # RHS cut-off frames:
    # 20230318: right cutoff
    # 20230327: right
    # 20230329: right (big!)
    # 20230331: right (big!)  bigger than 0329   (CUT? It's evening dark == not necessary)
    # 20230401: right (big!)  red line aligned - useful?
    # 20230402: right (big!)  bigger than 0401
    # 20230403: right
    # 20230404: right (big!)  red line aligned - useful?
    # 20230407: right (big!)  bigger than 0407
    # 20230418: right (big!!!)  <- maxi (& red aligned)

    # through 20230507 <- maxi (& red aligned)

    # 20230611
    # 20231014
