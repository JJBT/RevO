import argparse
import os
from data_preprocessing import create_megapixel_mnist
from omegaconf import OmegaConf


def create_local_config(megapixel_mnist_path):
    cfg_dict = {
        'data':
            {
                'megapixel_mnist_train_root': os.path.join(megapixel_mnist_path, 'train'),
                'megapixel_mnist_val_root': os.path.join(megapixel_mnist_path, 'val'),
                'megapixel_mnist_train_val_root': os.path.join(megapixel_mnist_path, 'train_val'),
                'megapixel_mnist_annotations_root': os.path.join(megapixel_mnist_path, 'annotations')
            }
    }
    cfg = OmegaConf.create(cfg_dict)
    cfg_path = 'conf/data_path/local.yaml'
    OmegaConf.save(cfg, f=cfg_path)
    with open(cfg_path, 'r+') as f:
        lines = f.readlines()
        lines.insert(0, '# @package _global_\n')
        f.seek(0)
        f.writelines(lines)
        f.truncate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mnist_path", help="path to save mnist dataset (if already exists will ignore it)")
    parser.add_argument("--megapixel_mnist_path", help="path to save generated megapixel_mnist dataset")
    args = parser.parse_args()

    create_megapixel_mnist.main(args.mnist_path, args.megapixel_mnist_path)
    create_local_config(args.megapixel_mnist_path)




