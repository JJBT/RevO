from pycocotools.coco import COCO
import os
import json
import cv2
import matplotlib.pyplot as plt
from data_utils import check_bbox_validity, get_anns_info_df, save_anns

import settings


def show_image(img, anns):
    for ann in anns:
        x, y, w, h = ann['bbox']
        cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (255, 0, 0), 5)

    cv2.imshow(' ', img)
    plt.show()


def print_dataset_info(ann_file):
    coco = COCO(ann_file)
    anns_df = get_anns_info_df(coco, save=False)
    print('/-/-/-/-/-/-DATASET INFO-\\-\\-\\-\\-\\-\\')
    print(f'{anns_df["image_id"].nunique()} images collected with {len(anns_df)} annotations totally.')
    instances_per_image = anns_df.groupby(by='image_id')['id'].count()
    print(f'Number of instances per image: min={instances_per_image.min()}, mean={instances_per_image.mean()}, max={instances_per_image.max()}')
    print(f'Bounding boxes y scale: min={anns_df["bbox_y_scale"].min()}, mean={anns_df["bbox_y_scale"].mean()}, max={anns_df["bbox_y_scale"].max()}')
    print('/-/-/-/-/-/-/-/-/-/-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\')


def main(coco, path_to_anns, path_to_images):
    print('Golden standard vaidation dataset creation.')
    print('To add image to the dataset press "y".')
    print('To pass image press "n".')
    print('To terminate dataset collecting press "e" or "q".')
    img_metas = coco.loadImgs(ids=coco.getImgIds())

    img_ids_to_keep = []
    for i, img_meta in enumerate(img_metas):
        img_id = img_meta['id']
        anns = coco.loadAnns(coco.getAnnIds(img_id))
        for ann in anns:
            if ann['iscrowd'] or not check_bbox_validity(ann['bbox']):
                break
        else:
            img_name = coco.loadImgs(img_id)[0]['file_name']
            img = cv2.imread(os.path.join(path_to_images, img_name))
            show_image(img, anns)
            k = cv2.waitKey(0)
            if k == ord('n') or k == ord('N'):
                print(f'Passing image {img_id}')
            elif k == ord('y') or k == ord('Y'):
                img_ids_to_keep.append(img_id)
                print(f'Saving image {img_id}')
            elif k == ord('e') or ord('E') or ord('q') or ord('Q'):
                cv2.destroyAllWindows()
                print('Golden standard dataset collecting terminated.')
                break

    imgs_to_keep = coco.loadImgs(img_ids_to_keep)
    anns_to_keep = coco.loadAnns(coco.getAnnIds(imgIds=img_ids_to_keep))

    with open(path_to_anns, 'r') as load_file:
        dataset = json.load(load_file)
        save_info = dataset['info']
        save_licenses = dataset['licenses']
        save_categories = dataset['categories']

    gs_coco = {
        'info': save_info,
        'licenses': save_licenses,
        'images': imgs_to_keep,
        'annotations': anns_to_keep,
        'categories': save_categories
    }
    return gs_coco


if __name__ == '__main__':
    PATH_TO_COCO = 'D:/datasets/annotations/person_keypoints_val2017.json'
    PATH_TO_ANNS = PATH_TO_COCO
    PATH_TO_IMAGES = 'D:/datasets/val2017'
    PATH_TO_SAVE = os.path.join(settings.BASE_DIR, 'data', 'gs_coco_object_presence_anns.json')
    coco = COCO(PATH_TO_COCO)
    gs_coco = main(coco, PATH_TO_ANNS, PATH_TO_IMAGES)
    save_anns(gs_coco, PATH_TO_SAVE)
    print_dataset_info(PATH_TO_SAVE)
