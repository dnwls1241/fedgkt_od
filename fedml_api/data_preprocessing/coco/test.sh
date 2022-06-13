#!/bin/bash

python3 test.py --dataset coco --data_dir /workspace/dataset/coco \
        --dataidx_map /workspace/dataset/captone/test1 \
        --partition_method homo --n_nets 2 --alpha 1.0