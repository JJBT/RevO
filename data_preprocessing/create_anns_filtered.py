from pycocotools.coco import COCO
import pandas as pd
import os
import json

from .utils import get_anns_info_df, save_coco_anns
import sys
sys.path.append('..')
import settings


def filter_coco_csv(anns_df, coco, path_to_anns, thr_x_scale=0., thr_y_scale=0.):
    # Drop invalid coco person keypoints instances
    # and filter instances by num_keypoints, bbox_x_scale and bbox_y_scale


    num_anns = len(anns_df)
    print(f'Originally {num_anns} annotation instances found.')
    # Check validity
    anns_df = anns_df[anns_df['is_crowd'] < 1]
    anns_df = anns_df[(anns_df['bbox_x'] >= 0) & (anns_df['bbox_y'] >= 0)]
    anns_df = anns_df[(anns_df['bbox_width'] > 0) & (anns_df['bbox_height'] > 0)]
    print(f'{num_anns - len(anns_df)} invalid annotation instances dropped.')
    num_anns = len(anns_df)

    anns_df = anns_df[(anns_df['bbox_x_scale'] > thr_x_scale) & (anns_df['bbox_y_scale'] > thr_y_scale)]
    print(f'{num_anns - len(anns_df)} annotation instances filtered:')
    print(f'bbox_x_scale > {thr_x_scale}\n\tbbox_y_scale > {thr_y_scale}')
    num_anns = len(anns_df)
    print(f'{num_anns} annotation instances remained.')

    ids = anns_df['id'].values
    anns_to_keep = coco.loadAnns(ids=ids)

    img_ids_to_keep = set()
    for ann in anns_to_keep:
        img_ids_to_keep.add(ann['image_id'])

    img_ids_to_keep = list(img_ids_to_keep)

    imgs_to_keep = coco.loadImgs(ids=img_ids_to_keep)

    with open(path_to_anns, 'r') as load_file:
        dataset = json.load(load_file)
        save_info = dataset['info']
        save_licenses = dataset['licenses']
        save_categories = dataset['categories']

    filtered_coco = {
        'info': save_info,
        'licenses': save_licenses,
        'images': imgs_to_keep,
        'annotations': anns_to_keep,
        'categories': save_categories
    }
    return filtered_coco


if __name__ == '__main__':
    path_to_anns = settings.COCO_KPS_TRAIN_PATH
    coco = COCO(path_to_anns)

    anns_df = get_anns_info_df(coco=coco)

    q_filtered_ans = filter_coco_csv(anns_df=anns_df, coco=coco, path_to_anns=path_to_anns,
                                     thr_y_scale=0.1)

    os.makedirs(os.path.join(settings.BASE_DIR, 'data_preprocessing/annotations'), exist_ok=True)
    q_fname_to_save = os.path.join(settings.BASE_DIR, 'data_preprocessing/annotations/q_person_keypoints_train2017.json')
    save_coco_anns(q_filtered_ans, q_fname_to_save)

    s_filtered_anns = filter_coco_csv(anns_df=anns_df, coco=COCO, path_to_anns=path_to_anns,
                                      thr_y_scale=0.1)

    s_fname_to_save = os.path.join(settings.BASE_DIR, 'data_preprocessing/annotations/s_person_keypoints_train2017.json')
    save_coco_anns(s_filtered_anns, s_fname_to_save)
