#!/bin/bash
python test.py --dataset=coco --train_dir=/workspace/space/dataset/train/train2 --train_ann=/workspace/space/dataset/traincoco.json \
--test_dir=/workspace/space/dataset/test/test --test_ann=/workspace/space/dataset/testcoco.json \
--client_number=100 --train_bs=10 --test_bs=10