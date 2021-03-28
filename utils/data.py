from pycocotools.coco import COCO
import json

def get_coco_img_ids(coco):
    """
    Return a list of image ids according to annotations
    (use in case annotations were changed after coco loading).
    """
    img_ids = set()
    for ann in coco.loadAnns(coco.getAnnIds()):
        img_ids.add(ann['image_id'])

    return list(img_ids)


def load_coco_samples(coco):
    samples = []

    img_ids = get_coco_img_ids(coco)
    for img_meta in coco.loadImgs(img_ids):
        image_id = img_meta['id']
        file_name = img_meta['file_name']
        height = img_meta['height']
        width = img_meta['width']
        anns = coco.loadAnns(coco.getAnnIds(image_id))

        sample = {
            'image_id': image_id,
            'file_name': file_name,
            'height': height,
            'width': width,
            'anns': anns
        }
        samples.append(sample)

    return samples


def save_coco_anns(anns, filename_to_save):
    with open(filename_to_save, 'w') as file:
        json.dump(anns, file)
