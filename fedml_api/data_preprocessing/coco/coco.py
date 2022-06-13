# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path

import torch
import torch.utils.data
import torchvision
from pycocotools import mask as coco_mask

from . import transforms as T

class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, dataidxs=None):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        if dataidxs is not None:
            self.ids = dataidxs
        self._transforms = transforms
        

    def __getitem__(self, idx):
        image_id = self.ids[idx]
        img, target = super(CocoDetection, self).__getitem__(idx)
        target = {'image_id': image_id, 'annotations': target}
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        boxes, labels = [], []
        for obj in target['annotations']:
            boxes.append(obj['bbox'])
            labels.append(obj['category_id'])
        target = {}
        target['boxes'] = torch.tensor(boxes)
        target['labels'] = torch.tensor(labels)
        return img, target


def make_coco_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(data_dir, dataidxs=None, train=True):
    image_set = "train" if train is True else "val"
    root = Path(data_dir)
    assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'instances'
    PATHS = {
        "train": (root / "train2017", root / "annotations" / f'{mode}_train2017.json'),
        "val": (root / "val2017", root / "annotations" / f'{mode}_val2017.json'),
    }

    img_folder, ann_file = PATHS[image_set]
    dataset = CocoDetection(img_folder, ann_file, transforms=None, dataidxs=dataidxs)
    return dataset
