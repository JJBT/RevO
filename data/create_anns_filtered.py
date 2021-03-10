from pycocotools.coco import COCO
import pandas as pd
import os
import json
import settings


def filter_coco_csv(df_fname, path_to_anns=None, thr_kps=0, thr_x_scale=0., thr_y_scale=0.):
    # Drop invalid coco person keypoints instances
    # and filter instances by num_keypoints, bbox_x_scale and bbox_y_scale
    if path_to_anns is None:
        path_to_anns = settings.COCO_KPS_TRAIN_PATH

    df = pd.read_csv(df_fname, index_col=0)
    num_anns = len(df)
    print(f'Originally {num_anns} annotation instances found.')
    # Check validity
    df = df[df['is_crowd'] < 1]
    df = df[(df['bbox_x'] >= 0) & (df['bbox_y'] >= 0)]
    df = df[(df['bbox_width'] > 0) & (df['bbox_height'] > 0)]
    df = df[df['num_keypoints'] >= 0]
    print(f'{num_anns - len(df)} invalid annotation instances dropped.')
    num_anns = len(df)

    df = df[(df['num_keypoints'] > thr_kps)]
    df = df[(df['bbox_x_scale'] > thr_x_scale) & (df['bbox_y_scale'] > thr_y_scale)]
    print(f'{num_anns - len(df)} annotation instances filtered:')
    print(f'\tnum_keypoints > {thr_kps}\n\tbbox_x_scale > {thr_x_scale}\n\tbbox_y_scale > {thr_y_scale}')
    num_anns = len(df)
    print(f'{num_anns} annotation instances remained.')

    ids = df['id'].values
    coco = COCO(path_to_anns)
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


def save_anns(anns, filename_to_save):
    with open(filename_to_save, 'w') as file:
        json.dump(anns, file)


if __name__ == '__main__':
    df_fname = os.path.join(settings.BASE_DIR, 'data/anns_info.csv')
    q_filtered_ans = filter_coco_csv(df_fname=df_fname, path_to_anns=settings.COCO_KPS_TRAIN_PATH,
                                     thr_y_scale=0.1)

    os.makedirs(os.path.join(settings.BASE_DIR, 'data/annotations'))
    q_fname_to_save = os.path.join(settings.BASE_DIR, 'data/annotations/q_person_keypoints_train2017.json')
    save_anns(q_filtered_ans, q_fname_to_save)

    s_filtered_anns = filter_coco_csv(df_fname=df_fname, path_to_anns=settings.COCO_KPS_TRAIN_PATH,
                                      thr_y_scale=0.1)

    s_fname_to_save = os.path.join(settings.BASE_DIR, 'data/annotations/s_person_keypoints_train2017.json')
    save_anns(s_filtered_anns, s_fname_to_save)
