import os
import random
from contextlib import redirect_stdout
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils import data
from pycocotools.coco import COCO
import math
from torchvision.ops.boxes import _box_xywh_to_xyxy
from torchvision.transforms.functional import adjust_brightness, adjust_contrast, adjust_hue, adjust_saturation
from pathlib import Path

class CocoDataset(data.dataset.Dataset):
    'Dataset looping through a set of images'

    def __init__(self, path, resize, max_size, stride, annotations=None, training=False, rotate_augment=False,
                 augment_brightness=0.0, augment_contrast=0.0,
                 augment_hue=0.0, augment_saturation=0.0, dataidxs=None):
        super().__init__()

        self.path = os.path.expanduser(path)
        self.resize = resize
        self.max_size = max_size
        self.stride = stride
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.training = training
        self.rotate_augment = rotate_augment
        self.augment_brightness = augment_brightness
        self.augment_contrast = augment_contrast
        self.augment_hue = augment_hue
        self.augment_saturation = augment_saturation

        with redirect_stdout(None):
            self.coco = COCO(annotations)
        if dataidxs is not None:
            self.ids = dataidxs
        else: 
            self.ids = list(self.coco.imgs.keys())
        if 'categories' in self.coco.dataset:
            self.categories_inv = {k: i for i, k in enumerate(self.coco.getCatIds())}

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        ' Get sample'

        # Load image
        id = self.ids[index]
        if self.coco:
            image = self.coco.loadImgs(id)[0]['file_name']
        im = Image.open('{}/{}'.format(self.path, image)).convert("RGB")

        # Randomly sample scale for resize during training
        resize = self.resize
        if isinstance(resize, list):
            resize = random.randint(self.resize[0], self.resize[-1])

        ratio = resize / min(im.size)
        if ratio * max(im.size) > self.max_size:
            ratio = self.max_size / max(im.size)
        im = im.resize((int(ratio * d) for d in im.size), Image.BILINEAR)

        # Get annotations
        boxes, categories = self._get_target(id)
        boxes *= ratio
        
        if self.training:
            
            # Random rotation, if self.rotate_augment
            random_angle = random.randint(0, 3) * 90
            if self.rotate_augment and random_angle != 0:
                # rotate by random_angle degrees.
                im = im.rotate(random_angle)
                x, y, w, h = boxes[:, 0].clone(), boxes[:, 1].clone(), boxes[:, 2].clone(), boxes[:, 3].clone()
                if random_angle == 90:
                    boxes[:, 0] = y - im.size[1] / 2 + im.size[0] / 2
                    boxes[:, 1] = im.size[0] / 2 + im.size[1] / 2 - x - w
                    boxes[:, 2] = h
                    boxes[:, 3] = w
                elif random_angle == 180:
                    boxes[:, 0] = im.size[0] - x - w
                    boxes[:, 1] = im.size[1] - y - h
                elif random_angle == 270:
                    boxes[:, 0] = im.size[0] / 2 + im.size[1] / 2 - y - h
                    boxes[:, 1] = x - im.size[0] / 2 + im.size[1] / 2
                    boxes[:, 2] = h
                    boxes[:, 3] = w

            # Random horizontal flip
            if random.randint(0, 1):
                im = im.transpose(Image.FLIP_LEFT_RIGHT)
                boxes[:, 0] = im.size[0] - boxes[:, 0] - boxes[:, 2]

            # Apply image brightness, contrast etc augmentation
            if self.augment_brightness:
                brightness_factor = random.normalvariate(1, self.augment_brightness)
                brightness_factor = max(0, brightness_factor)
                im = adjust_brightness(im, brightness_factor)
            if self.augment_contrast:
                contrast_factor = random.normalvariate(1, self.augment_contrast)
                contrast_factor = max(0, contrast_factor)
                im = adjust_contrast(im, contrast_factor)
            if self.augment_hue:
                hue_factor = random.normalvariate(0, self.augment_hue)
                hue_factor = max(-0.5, hue_factor)
                hue_factor = min(0.5, hue_factor)
                im = adjust_hue(im, hue_factor)
            if self.augment_saturation:
                saturation_factor = random.normalvariate(1, self.augment_saturation)
                saturation_factor = max(0, saturation_factor)
                im = adjust_saturation(im, saturation_factor)
            
        boxes = _box_xywh_to_xyxy(boxes)
        target = torch.cat([boxes, categories], dim=1)

        # Convert to tensor and normalize
        data = torch.ByteTensor(torch.ByteStorage.from_buffer(im.tobytes()))
        data = data.float().div(255).view(*im.size[::-1], len(im.mode))
        data = data.permute(2, 0, 1)

        # for t, mean, std in zip(data, self.mean, self.std):
        #     t.sub_(mean).div_(std)

        # Apply padding
        pw, ph = ((self.stride - d % self.stride) % self.stride for d in im.size)
        data = F.pad(data, (0, pw, 0, ph))

        return data, target

    def _get_target(self, id):
        'Get annotations for sample'

        ann_ids = self.coco.getAnnIds(imgIds=id)
        annotations = self.coco.loadAnns(ann_ids)

        boxes, categories = [], []
        for ann in annotations:
            if ann['bbox'][2] < 1 and ann['bbox'][3] < 1:
                continue
            boxes.append(ann['bbox'])
            cat = ann['category_id']
            if 'categories' in self.coco.dataset:
                cat = self.categories_inv[cat]
            categories.append(cat)

        if boxes:
            target = (torch.FloatTensor(boxes),
                      torch.FloatTensor(categories).unsqueeze(1))
        else:
            target = (torch.ones([1, 4]), torch.ones([1, 1]) * -1)

        return target

    def collate_fn(self, batch):
        'Create batch from multiple samples'

        if self.training:
            data, targets = zip(*batch)
            max_det = max([t.size()[0] for t in targets])
            targets = [torch.cat([t, torch.ones([max_det - t.size()[0], 5]) * -1]) for t in targets]
            targets = torch.stack(targets, 0)
        else:
            data, indices, ratios = zip(*batch)

        # Pad data to match max batch dimensions
        sizes = [d.size()[-2:] for d in data]
        w, h = (max(dim) for dim in zip(*sizes))

        data_stack = []
        for datum in data:
            pw, ph = w - datum.size()[-2], h - datum.size()[-1]
            data_stack.append(
                F.pad(datum, (0, ph, 0, pw)) if max(ph, pw) > 0 else datum)

        data = torch.stack(data_stack)

        if self.training:
            return data, targets

        ratios = torch.FloatTensor(ratios).view(-1, 1, 1)
        return data, torch.IntTensor(indices), ratios
    
    def collate_fn_2(self, batch):
        data, targets = zip(*batch)
        targets = [{"boxes": t[:, :-1], "labels": t[:, -1].type(torch.int64)} for t in targets]
        return data, targets

class DataIterator():
    'Data loader for data parallel'

    def __init__(self, path, resize, max_size, batch_size, stride,annotations, world=1,training=False,
                 rotate_augment=False, augment_brightness=0.0,
                 augment_contrast=0.0, augment_hue=0.0, augment_saturation=0.0, dataidxs=None):
        self.resize = resize
        self.max_size = max_size

        self.dataset = CocoDataset(path, resize=resize, max_size=max_size,
                                   stride=stride, annotations=annotations, training=training,
                                   rotate_augment=rotate_augment,
                                   augment_brightness=augment_brightness,
                                   augment_contrast=augment_contrast, augment_hue=augment_hue,
                                   augment_saturation=augment_saturation, dataidxs=dataidxs)
        self.ids = self.dataset.ids
        self.coco = self.dataset.coco

        self.sampler = data.distributed.DistributedSampler(self.dataset) if world > 1 else None
        self.dataloader = data.DataLoader(self.dataset, batch_size=batch_size // world,
                                          sampler=self.sampler, collate_fn=self.dataset.collate_fn_2, num_workers=2,
                                          pin_memory=True)

    def __repr__(self):
        return '\n'.join([
            '    loader: pytorch',
            '    resize: {}, max: {}'.format(self.resize, self.max_size),
        ])

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        for output in self.dataloader:
            data, target = output


            if torch.cuda.is_available():
                data = [d.cuda(non_blocking=True) for d in data]

           
            if torch.cuda.is_available():
                target = [{k: v.cuda(non_blocking=True) for k, v in t.items()} for t in target]
            yield data, target

def build(args, dataidxs=None, train=True, ):
    image_set = "train" if train is True else "val"
    root = Path(args.data_dir)
    assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'instances'
    PATHS = {
        "train": (root / "train2017", root / "annotations" / f'{mode}_train2017.json'),
        "val": (root / "val2017", root / "annotations" / f'{mode}_val2017.json'),
    }

    img_folder, ann_file = PATHS[image_set]
    dataset = DataIterator(img_folder, args.resize, args.max_size, args.batch_size, 128, ann_file, training=train, dataidxs=dataidxs)
    return dataset

def build_2(args, dataidxs=None, train=True):
    image_set = "train" if train is True else "val"
    root = Path(args.data_dir)
    assert root.exists(), f'provided COCO path {root} does not exist'
    PATHS = {
        "train": (root / "train" / "JPEGImages", root / "train" / "ann.json"),
        "val": (root / "test"/ "JPEGImages", root / "test" / "ann.json"),
    }

    img_folder, ann_file = PATHS[image_set]
    dataset = DataIterator(img_folder, args.resize, args.max_size, args.batch_size, 128, ann_file, training=train, dataidxs=dataidxs)
    return dataset