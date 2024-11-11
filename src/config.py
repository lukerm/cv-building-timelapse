import os

IMAGE_ROOT = os.path.expanduser('~/Pictures/London/.../balcony/')
HI_RES_FOLDER = os.path.join(IMAGE_ROOT, 'construction')
LO_RES_FOLDER = os.path.join(IMAGE_ROOT, 'construction_lores')

CROP_SIZE = (256, 256)

MODEL_DIR = os.path.expanduser('~/cv-building-timelapse/models/experiments/256/')
MODEL_MAP = {
    'D2': os.path.join(MODEL_DIR, '025_gpu', 'e88.model.weights'),
    'L2': os.path.join(MODEL_DIR, '015_gpu', 'best.model.weights'),
    'R1': os.path.join(MODEL_DIR, '004_gpu', 'best.model.weights'),
    'R3': os.path.join(MODEL_DIR, '035_gpu', 'best.model.weights'),
}





