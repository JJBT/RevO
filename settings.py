import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if os.getenv('coco_path') is None:
    COCO_PATH = os.path.join(BASE_DIR, 'coco')
else:
    COCO_PATH = os.getenv('coco_path')

COCO_TRAIN_PATH = os.path.join(COCO_PATH, 'train2017')
COCO_VAL_PATH = os.path.join(COCO_PATH, 'val2017')
COCO_ANN_PATH = os.path.join(COCO_PATH, 'annotations')
COCO_KPS_TRAIN_PATH = os.path.join(COCO_ANN_PATH, 'person_keypoints_train2017.json')
COCO_KPS_VAL_PATH = os.path.join(COCO_ANN_PATH, 'person_keypoints_val2017.json')
