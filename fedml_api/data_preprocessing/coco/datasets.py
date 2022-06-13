import logging
from matplotlib import transforms

import numpy as np
import torch.utils.data as data
from PIL import Image
from torchvision.datasets import VisionDataset
from pathlib import Path
from . import transforms as T
import os
from typing import Any, Callable, List, Optional, Tuple

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class Coco_truncated(VisionDataset):

    def __init__(self, root, dataidxs=None, train=True, transform=None, download=False):
        super().__init__(root)
        self.root = root
        self.dataidxs = dataidxs
        self.download = download

        if train:
            self.image_set = "train"
        else:
            self.image_set = "val"
        if transform is None:
            self.transform = make_coco_transforms(self.image_set)
        else:
            self.transform = transform
        from pycocotools.coco import COCO
        self.img_folder, self.ann_file = coco_path(self.image_set, self.root)
        self.coco = COCO(self.ann_file) 
        self.imgs, self.targets = self.__build_truncated_dataset__()
        self.ids = list(sorted(self.imgs.keys()))

    def __build_truncated_dataset__(self):
        
        if self.dataidxs is not None:
            imgs = self.coco.loadImgs(self.dataidxs)
            targets = self.coco.loadAnns(self.dataidxs)
        else:
            imgs = self.coco.imgs
            targets = self.coco.anns
        return imgs, targets
    
    def _load_image(self, id: int) -> Image.Image:
        path = self.imgs[id]["file_name"]
        return Image.open(os.path.join(self.root, path)).convert("RGB")

    def _load_target(self, id: int) -> List[Any]:
        return self.targets(self.coco.getAnnIds(id))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self) -> int:
        return len(self.ids)

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


def coco_path(image_set, root):
    root = Path(root)
    assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'instances'
    PATHS = {
        "train": (root / "train2017", root / "annotations" / f'{mode}_train2017.json'),
        "val": (root / "val2017", root / "annotations" / f'{mode}_val2017.json'),
    }

    img_folder, ann_file = PATHS[image_set]
    return img_folder, ann_file