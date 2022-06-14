import argparse
import logging
import os
import socket
import sys
import tqdm
import numpy as np
import shutil
import setproctitle
import torch
import wandb
import gc
# add the FedML root directory to the python path
from torchinfo import summary

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))
from fedml_api.distributed.fedgkt.GKTGlobalTrainer import GKTGlobalTrainer
from fedml_api.distributed.fedgkt.GKTLocalTrainer import GKTLocalTrainer
from fedml_api.model.gkt.GKTRetinanet import gktservermodel, gktclientmodel
from fedml_api.data_preprocessing.coco.data_loader import init_distirbuted_data, get_dataloader_coco_v2, init_distirbuted_data_test
from fedml_api.detection.engine import evaluate
from fedml_api.detection.coco_utils import CocoDetection
from fedml_api.utils import utils_ObjectDetection as utils
def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """

    parser.add_argument('--image_mean', default=None)
    parser.add_argument('--image_std', default=None)
    parser.add_argument('--score_thresh', default=0.5)
    parser.add_argument('--nms_thresh', default=0.5)
    parser.add_argument('--detections_per_img', default=300)
    parser.add_argument('--fg_iou_thresh', default=0.5)
    parser.add_argument('--bg_iou_thresh', default=0.4)
    parser.add_argument('--topk_candidates', default=1000)
    parser.add_argument('--is_server', default=False)
    
    parser.add_argument('--pretrained', default=False)
    parser.add_argument('--weight_dir', type=str, default="./weight")
    parser.add_argument('--num_classes', type=int, default=91)
    # Training settings
    parser.add_argument('--model_client', type=str, default='resnet5', metavar='N',
                        help='neural network used in training')

    parser.add_argument('--model_server', type=str, default='resnet32', metavar='N',
                        help='neural network used in training')
    parser.add_argument('--backbone_name', type=str, default='resnet50', metavar='N',
                        help='neural network used in training')
    parser.add_argument('--dataset', type=str, default='coco', metavar='N',
                        help='dataset used for training')

    parser.add_argument('--server_weight', type=str, default=None)

    parser.add_argument('--project_dir', type=str, default='./project', metavar='N',
                        help='train results dir')

    parser.add_argument('--resize', metavar='scale', type=int, help='resize to given size', default=800)
    parser.add_argument('--max-size', metavar='max', type=int, help='maximum resizing size', default=1333)
    # parser.add_argument('--annotations', metavar='path', type=str, help='path to COCO style annotations',
    #                           required=True)
    parser.add_argument('--data_dir', type=str, default="/workspace/dataset/VOCdevkit")
    parser.add_argument('--partition_file', type=str, default="./dataidx_map.json")
    parser.add_argument('--partition_file_test', type=str, default="./dataidx_map_test.json")
    parser.add_argument('--change_map', action='store_true')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--batch_size_test', type=int, default=2)

    parser.add_argument('--partition_num', type=int, default=10, metavar='N')

    parser.add_argument('--partition_method', type=str, default='homo', metavar='N',
                        help='how to partition the dataset on local workers')

    parser.add_argument('--partition_alpha', type=float, default=0.5, metavar='PA',
                        help='partition alpha (default: 0.5)')

    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')

    parser.add_argument('--wd', help='weight decay parameter;', type=float, default=5e-4)

    parser.add_argument('--epochs_client', type=int, default=3, metavar='EP',
                        help='how many epochs will be trained locally')

    parser.add_argument('--local_points', type=int, default=5000, metavar='LP',
                        help='the approximate fixed number of data points we will have on each local worker')

    parser.add_argument('--client_number', type=int, default=4, metavar='NN',
                        help='number of workers in a distributed cluster')

    parser.add_argument('--server_chan', type=int, default=2,
                        help='server_model_channel_multiplyer')

    parser.add_argument('--comm_round', type=int, default=10,
                        help='how many round of communications we shoud use')

    parser.add_argument('--gpu', type=int, default=0,
                        help='gpu')
    parser.add_argument('--resume_round', type=int, default=0)
    parser.add_argument('--loss_scale', type=float, default=1024,
                        help='Loss scaling, positive power of 2 values can improve fp16 convergence.')
    parser.add_argument('--no_bn_wd', action='store_true', help='Remove batch norm from weight decay')
    parser.add_argument('--device', type=int, default=0)
    # knowledge distillation
    parser.add_argument('--temperature', default=3.0, type=float, help='Input the temperature: default(3.0)')
    parser.add_argument('--epochs_server', type=int, default=3, metavar='EP',
                        help='how many epochs will be trained on the server side')
    parser.add_argument('--alpha', default=1.0, type=float, help='Input the relative weight: default(1.0)')
    parser.add_argument('--optimizer', default="Adam", type=str, help='optimizer: SGD, Adam, etc.')
    parser.add_argument('--whether_training_on_client', default=1, type=int)
    parser.add_argument('--whether_training_on_server', default=1, type=int)
    parser.add_argument('--whether_distill_on_the_server', default=0, type=int)
    parser.add_argument('--client_model', default="resnet4", type=str)
    parser.add_argument('--weight_init_model', default="resnet32", type=str)
    parser.add_argument('--running_name', default="default", type=str)
    parser.add_argument('--sweep', default=0, type=int)
    parser.add_argument('--multi_gpu_server', action='store_true')
    parser.add_argument('--test', action='store_true',
                        help='test mode, only run 1-2 epochs to test the bug of the program')
    parser.add_argument('--gpu_num_per_server', type=int, default=8,
                        help='gpu_num_per_server')
    parser.add_argument('--last_server_epoch', type=int, default=0)
    parser.add_argument('--servermodel_eval', type=int, default=1)
    parser.add_argument('--clientmodel_eval', type=int, default=0)
    parser.add_argument('--server_make_logits', type=int, default=1)
    parser.add_argument('--client_make_logits', type=int, default=1)
    args = parser.parse_args()
    return args

def make_prediction(model, img, threshold):
    model.eval()
    preds, _, _ = model(img)
    for id in range(len(preds)) :
        idx_list = []

        for idx, score in enumerate(preds[id]['scores']) :
            if score > threshold : #threshold 넘는 idx 구함
                idx_list.append(idx)

        preds[id]['boxes'] = preds[id]['boxes'][idx_list]
        preds[id]['labels'] = preds[id]['labels'][idx_list]
        preds[id]['scores'] = preds[id]['scores'][idx_list]


    return preds

if __name__ == "__main__":
    # initialize distributed computing (MPI)
    # comm, process_id, worker_number = FedML_init()

    # parse python script input parameters
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    logging.info(args)
    
    seed = 0
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    

    # wandb.init(project="[Fedgkt Object Detection]", entity="ddojackin")

    dataidxs = init_distirbuted_data(args)["0"]
    dataidxs_test = init_distirbuted_data_test(args)["0"]
    round_idx = args.resume_round
    last_epoch = args.last_server_epoch
    server_dir = args.project_dir+"/server"
    torch.autograd.set_detect_anomaly(True)
    weight_path = args.weight_dir+"/client/round4_client0.pth"
    
    # model =  gktservermodel(pretrained=args.pretrained, path=weight_path,
    #                                         num_classes=args.num_classes, backbone_name=args.backbone_name, server_chan=args.server_chan)
    model =  gktclientmodel(pretrained=args.pretrained, path=weight_path,
                                            num_classes=args.num_classes, backbone_name=args.backbone_name)
    model.to(args.device)
    model.eval()

    test_data_loader, _ = get_dataloader_coco_v2(args, dataidxs, dataidxs_test)
    labels = []
    preds_adj_all = []
    annot_all = []

    for im, annot in test_data_loader:
        im = list(img.to(args.device) for img in im)
        annot = [{k: v.to(torch.device('cpu')) for k, v in t.items()} for t in annot]

        for t in annot:
            labels += t['labels']

        with torch.no_grad():
            preds_adj = make_prediction(model, im, 0.4)
            preds_adj = [{k: v.to(torch.device('cpu')) for k, v in t.items()} for t in preds_adj]
            preds_adj_all.append(preds_adj)
            annot_all.append(annot)

    sample_metrics = []
    for batch_i in range(len(preds_adj_all)):
        sample_metrics += utils.get_batch_statistics(preds_adj_all[batch_i], annot_all[batch_i], iou_threshold=0.5) 

    true_positives, pred_scores, pred_labels = [torch.cat(x, 0) for x in list(zip(*sample_metrics))]  # 배치가 전부 합쳐짐
    precision, recall, AP, f1, ap_class = utils.ap_per_class(true_positives, pred_scores, pred_labels, torch.tensor(labels))
    mAP = torch.mean(AP)
    print(f'mAP : {mAP}')
    print(f'AP : {AP}')

    print("every eval completed!")

