import os
import argparse
from dataset import build_dataset
import torch
from torch.utils.data import DataLoader
import util.misc as utils

def get_args_parser():
    parser = argparse.ArgumentParser('Set dataset', add_help=False)
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")
    #parser.add_argument('--batch_size')
    return parser

def main(args):
    dataset_train = build_dataset(image_set='train', args=args)
    #dataset_val = build_dataset(image_set='val', args=args)

    # if args.distributed:
    #     sampler_train = DistributedSampler(dataset_train)
    #     sampler_val = DistributedSampler(dataset_val, shuffle=False)
    # else:
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    #     sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    # batch_sampler_train = torch.utils.data.BatchSampler(
    #     sampler_train, args.batch_size, drop_last=True)
    data_loader_train = DataLoader(dataset_train, batch_sampler=sampler_train,
                                    collate_fn=utils.collate_fn)
    print(data_loader_train)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('test', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)