import os

IMAGE_ROOT = os.path.expanduser('~/Pictures/London/.../balcony/')
HI_RES_FOLDER = os.path.join(IMAGE_ROOT, 'construction')
HI_RES_SIZE = (4032, 3024)
LO_RES_FOLDER = os.path.join(IMAGE_ROOT, 'construction_lores')
LO_RES_SIZE = (1024, 768)

CROP_SIZE = (512, 512)

MODEL_DIR = os.path.expanduser(f'~/cv-building-timelapse/models/experiments/{CROP_SIZE[0]}/')
MODEL_MAP = {
    'D1': os.path.join(MODEL_DIR, '051_gpu', 'best.model.weights'),
    'D2': os.path.join(MODEL_DIR, '025_gpu', 'e88.model.weights'),
    'L2': os.path.join(MODEL_DIR, '015_gpu', 'best.model.weights'),
    'R1': os.path.join(MODEL_DIR, '004_gpu', 'best.model.weights'),
    'R3': os.path.join(MODEL_DIR, '042_gpu', 'best.model.weights'),
    'R4': os.path.join(MODEL_DIR, '056_gpu', 'best.model.weights'),
    'R_group': os.path.join(MODEL_DIR, '507_gpu', 'best.model.weights'),
    'DL_group': os.path.join(MODEL_DIR, '530_gpu', 'best.model.weights'),
}





