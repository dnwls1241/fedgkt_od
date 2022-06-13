import errno
import logging
from pkgutil import get_data

import numpy as np
from odtk.train import train
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import json
import os
from .data import build, build_2


logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def read_data_distribution(filename='./data_preprocessing/non-iid-distribution/CIFAR10/distribution.txt'):
    distribution = {}
    with open(filename, 'r') as data:
        for x in data.readlines():
            if '{' != x[0] and '}' != x[0]:
                tmp = x.split(':')
                if '{' == tmp[1].strip():
                    first_level_key = int(tmp[0])
                    distribution[first_level_key] = {}
                else:
                    second_level_key = int(tmp[0])
                    distribution[first_level_key][second_level_key] = int(tmp[1].strip().replace(',', ''))
    return distribution

def read_net_dataidx_map(filename='./dataidx_map.json'):
    with open(filename, 'r') as data:
        net_dataidx_map = json.load(data) 
        return net_dataidx_map

def save_net_dataidx_map(map_file, net_dataidx_map):
    map_dir = os.path.dirname(map_file)
    try:
        if not os.path.exists(map_dir):
            os.makedirs(map_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            print("Faild to create directiory!!!!")
            raise
    with open(map_file, 'w', encoding='utf-8') as f:
            json.dump(net_dataidx_map, f)
            print("save netidxmap complete")


def init_distirbuted_data(args):
    map_file = args.partition_file
    if os.path.exists(map_file) and not args.change_map:
        print("aleady map file exist. if you want to change map, use change_map option")
        return read_net_dataidx_map(map_file)
    dataidx_map = partition_data("train", args)
    save_net_dataidx_map(map_file, dataidx_map)
    return dataidx_map

def init_distirbuted_data_test(args):
    map_file = args.partition_file_test
    if os.path.exists(map_file) and not args.change_map:
        print("aleady map file exist. if you want to change map, use change_map option")
        return read_net_dataidx_map(map_file)
    dataidx_map = partition_data("val", args)
    save_net_dataidx_map(map_file, dataidx_map)
    return dataidx_map
   
    

def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
    logging.debug('Data statistics: %s' % str(net_cls_counts))
    return net_cls_counts

def load_coco_dataset(args):

    coco_train_ds = build_2(args, train=True)
    coco_test_ds = build_2(args, train=False)
    print("load_coco")

    return (coco_train_ds, coco_test_ds)

def partition_data(dataset, args):
    logging.info("*********partition data***************")
    train_ds , test_ds= load_coco_dataset(args)
    n_train = train_ds.__len__()
    # n_test = X_test.shape[0]
    if dataset == "train":
        ds = train_ds
    elif dataset == "val":
        ds = test_ds
    if args.partition_method == "homo":
        idxs = np.random.permutation(ds.ids)
        batch_idxs = np.array_split(idxs, args.partition_num)
        net_dataidx_map = {str(i): sorted(batch_idxs[i].tolist()) for i in range(args.partition_num)}
    # elif partition == "hetero":
    #     min_size = 0
    #     K = 10
    #     N = y_train.shape[0]
    #     logging.info("N = " + str(N))
    #     net_dataidx_map = {}

    #     while min_size < 10:
    #         idx_batch = [[] for _ in range(n_nets)]
    #         # for each class in the dataset
    #         for k in range(K):
    #             idx_k = np.where(y_train == k)[0]
    #             np.random.shuffle(idx_k)
    #             proportions = np.random.dirichlet(np.repeat(alpha, n_nets))
    #             ## Balance
    #             proportions = np.array([p * (len(idx_j) < N / n_nets) for p, idx_j in zip(proportions, idx_batch)])
    #             proportions = proportions / proportions.sum()
    #             proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
    #             idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
    #             min_size = min([len(idx_j) for idx_j in idx_batch])

    #     for j in range(n_nets):
    #         np.random.shuffle(idx_batch[j])
    #         net_dataidx_map[j] = idx_batch[j]

    # elif partition == "hetero-fix":
    #     dataidx_map_file_path = './data_preprocessing/non-iid-distribution/CIFAR10/net_dataidx_map.txt'
    #     net_dataidx_map = read_net_dataidx_map(dataidx_map_file_path)

    # if partition == "hetero-fix":
    #     distribution_file_path = './data_preprocessing/non-iid-distribution/CIFAR10/distribution.txt'
    #     traindata_cls_counts = read_data_distribution(distribution_file_path)
    # else:
    #     traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)
    
    return net_dataidx_map
# for centralized train
def get_dataloader_coco_v2(args, dataidxs=None, dataidxs_test=None):
    
    train_dl = build_2(args, dataidxs, train=True)
    test_dl = build_2(args, dataidxs_test, train=False)

    return train_dl, test_dl
