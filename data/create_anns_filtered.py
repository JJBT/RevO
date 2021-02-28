from pycocotools.coco import COCO
import pandas as pd
import os
import json
import settings


def filter_coco_csv(filename, thr_kps=17, thr_y_scale=0.6, path=None):
    # Filter coco instances by num_keypoints and bbox_y_scale
    if path is None:
        path = settings.COCO_KPS_TRAIN_PATH

    df = pd.read_csv(filename, index_col=0)
    df = df[(df['num_keypoints'] == thr_kps) & (df['bbox_y_scale'] > thr_y_scale)]
    print(f'With thr_kps = {thr_kps} and thr_y_scale {thr_y_scale} found {df.shape[0]} samples')

    ids = df['id'].values
    coco_kps = COCO(path)
    ann_kps = coco_kps.loadAnns(ids=ids)

    imgs_id_to_keep = set()
    for ann in ann_kps:
        imgs_id_to_keep.add(ann['image_id'])

    imgs_id_to_keep = list(imgs_id_to_keep)

    imgs_to_keep = coco_kps.loadImgs(ids=imgs_id_to_keep)

    with open(path, 'r') as load_file:
        dataset = json.load(load_file)
        save_info = dataset['info']
        save_licenses = dataset['licenses']
        save_categories = dataset['categories']

    filtered_coco = {
        'info': save_info,
        'licenses': save_licenses,
        'images': imgs_to_keep,
        'annotations': ann_kps,
        'categories': save_categories
    }
    return filtered_coco


def save_anns(anns, filename_to_save):
    with open(filename_to_save, 'w') as file:
        json.dump(anns, file)


if __name__ == '__main__':
    filename = os.path.join(settings.BASE_DIR, 'data/anns_info.csv')
    filtered_ans = filter_coco_csv(filename)

    filename_to_save = os.path.join(settings.BASE_DIR, 'data/anns_filtered.json')
    save_anns(filtered_ans, filename_to_save)
