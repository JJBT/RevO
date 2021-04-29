import json
from pycocotools.coco import COCO
from utils.data import save_coco_anns
from utils.data import get_category_based_anns


def filter_coco_by_cats(coco):
    category_based_anns = get_category_based_anns(coco)

    ann_ids_to_keep = []
    img_ids_to_keep = set()
    cat_ids_to_keep = set()

    for sample in category_based_anns:
        if len(sample['anns']) < 20:
            img_ids_to_keep.add(sample['image_id'])
            for ann in sample['anns']:
                ann_ids_to_keep.append(ann['id'])
                cat_ids_to_keep.add(ann['category_id'])

    cats_to_keep = [coco.cats[cat_id] for cat_id in cat_ids_to_keep]

    anns_to_keep = coco.loadAnns(ann_ids_to_keep)
    imgs_to_keep = coco.loadImgs(list(img_ids_to_keep))

    filtered_coco = {
        'images': imgs_to_keep,
        'annotations': anns_to_keep,
        'categories': cats_to_keep
    }
    return filtered_coco


if __name__ == '__main__':
    path_to_anns_train = '/data/fsod/annotations/fsod_train.json'
    path_to_save_train = '/data/fsod/annotations/fsod_train_filtered.json'
    path_to_anns_test = '/data/fsod/annotations/fsod_test.json'
    path_to_save_test = '/data/fsod/annotations/fsod_test_filtered.json'
    coco_train = COCO(path_to_anns_train)
    coco_test = COCO(path_to_anns_test)
    coco_train_filtered_json = filter_coco_by_cats(coco_train)
    coco_test_filtered_json = filter_coco_by_cats(coco_test)
    save_coco_anns(coco_train_filtered_json, path_to_save_train)
    save_coco_anns(coco_test_filtered_json, path_to_save_test)
