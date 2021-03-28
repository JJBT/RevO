import json
from pycocotools.coco import COCO
from utils.data import save_coco_anns


def filter_coco_by_cats(coco, cat_ids_to_keep):
    ann_ids_to_keep = []
    img_ids_to_keep = set()

    for ann in coco.loadAnns(coco.getAnnIds()):
        if ann['category_id'] in cat_ids_to_keep:
            ann_ids_to_keep.append(ann['id'])
            img_ids_to_keep.add(ann['image_id'])

    cats_to_keep = {cat_id: coco.cats[cat_id] for cat_id in cat_ids_to_keep}

    anns_to_keep = coco.loadAnns(ann_ids_to_keep)
    imgs_to_keep = coco.loadImgs(list(img_ids_to_keep))

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
        'categories': cats_to_keep
    }
    return filtered_coco


if __name__ == '__main__':
    path_to_anns = '/data/fsod/annotations/train_part.json'
    path_to_save = '/home/objdet/FSKD/data/filtered_fsod_train_part.json'
    coco = COCO(path_to_anns)
    cat_ids_to_keep = []
    coco_filtered_json = filter_coco_by_cats(coco, cat_ids_to_keep)
    save_coco_anns(coco_filtered_json, path_to_save)
