from __future__ import annotations
import argparse
import torch
import torchvision
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))
from fedml_api.distributed.fedgkt.GKTLocalTrainer import GKTLocalTrainer
from fedml_api.model.gkt.GKTRetinanet import gktservermodel, gktclientmodel
from fedml_api.data_preprocessing.coco.data_loader import init_distirbuted_data, get_dataloader_coco_v2, init_distirbuted_data_test, partition_data
from fedml_api.data_preprocessing.coco.data import DataIterator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


def get_args_parser():
    parser = argparse.ArgumentParser('Set model', add_help=False)
    # model args
    parser.add_argument('--image_mean', default=None)
    parser.add_argument('--image_std', default=None)
    parser.add_argument('--score_thresh', default=0.5)
    parser.add_argument('--nms_thresh', default=0.5)
    parser.add_argument('--detections_per_img', default=300)
    parser.add_argument('--fg_iou_thresh', default=0.5)
    parser.add_argument('--bg_iou_thresh', default=0.4)
    parser.add_argument('--topk_candidates', default=1000)
    
    parser.add_argument('--pretrained', default=False)
    parser.add_argument('--weight_dir', type=str, default=None)
    parser.add_argument('--num_classes', default=91)
    
    # dataset args
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')

    parser.add_argument('--wd', help='weight decay parameter;', type=float, default=5e-4)

    parser.add_argument('--epochs_client', type=int, default=3, metavar='EP',
                        help='how many epochs will be trained locally')

    parser.add_argument('--local_points', type=int, default=5000, metavar='LP',
                        help='the approximate fixed number of data points we will have on each local worker')

    parser.add_argument('--client_number', type=int, default=10, metavar='NN',
                        help='number of workers in a distributed cluster')

    parser.add_argument('--comm_round', type=int, default=300,
                        help='how many round of communications we shoud use')

    parser.add_argument('--gpu', type=int, default=0,
                        help='gpu')

    parser.add_argument('--loss_scale', type=float, default=1024,
                        help='Loss scaling, positive power of 2 values can improve fp16 convergence.')
    parser.add_argument('--no_bn_wd', action='store_true', help='Remove batch norm from weight decay')

##  
    parser.add_argument('--resize', metavar='scale', type=int, help='resize to given size', default=800)
    parser.add_argument('--max-size', metavar='max', type=int, help='maximum resizing size', default=1333)
    # parser.add_argument('--annotations', metavar='path', type=str, help='path to COCO style annotations',
    #                           required=True)
    parser.add_argument('--dataset', default='coco')
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--partition_file', type=str, default="./dataidx_map.json")
    parser.add_argument('--partition_file_test', type=str, default="./dataidx_map_test.json")
    parser.add_argument('--change_map', action='store_true')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--batch_size_test', type=int, default=2)
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")
    parser.add_argument('--partition_method', default="homo")
    parser.add_argument('--partition_alpha', type=float, default=0.5, metavar='PA',
                        help='partition alpha (default: 0.5)')
    parser.add_argument('--project_dir', type=str, default='./project', metavar='N',
                        help='train results dir')
##
    parser.add_argument('--temperature', default=3.0, type=float, help='Input the temperature: default(3.0)')
    parser.add_argument('--epochs_server', type=int, default=3, metavar='EP',
                        help='how many epochs will be trained on the server side')
    parser.add_argument('--alpha', default=1.0, type=float, help='Input the relative weight: default(1.0)')
    parser.add_argument('--optimizer', default="SGD", type=str, help='optimizer: SGD, Adam, etc.')
    parser.add_argument('--whether_training_on_client', default=1, type=int)
    parser.add_argument('--whether_distill_on_the_server', default=0, type=int)
    parser.add_argument('--client_model', default="resnet4", type=str)
    parser.add_argument('--weight_init_model', default="resnet32", type=str)
    parser.add_argument('--running_name', default="default", type=str)
    parser.add_argument('--sweep', default=0, type=int)
    parser.add_argument('--multi_gpu_server', action='store_true')
    parser.add_argument('--test', action='store_true',
                        help='test mode, only run 1-2 epochs to test the bug of the program')
    # image_mean=None, image_std=None,
    #              # Anchor parameters
    #              anchor_generator=None, head=None,
    #              proposal_matcher=None,
    #              score_thresh=0.05,
    #              nms_thresh=0.5,
    #              detections_per_img=300,
    #              fg_iou_thresh=0.5, bg_iou_thresh=0.4,
    #              topk_candidates=1000,
    #              is_server=False
    return parser


def run(args):

    # anchor_generator = AnchorGenerator(
    #     sizes = ((32, 64, 128, 256, 512),),
    #     aspect_ratios=((0.5, 1.0, 2.0),)
    # )
    dataidx_map = init_distirbuted_data(args)
    dataidxs = dataidx_map["0"]
    dataidx_map_test = init_distirbuted_data_test(args)
    dataidxs_test = dataidx_map_test["0"]
    model = gktclientmodel(backbone_name='resnet50', pretrained=args.pretrained, \
                                num_classes=args.num_classes, image_mean=args.image_mean, \
                                image_std=args.image_std, score_thresh=args.score_thresh, nms_thresh=args.nms_thresh, \
                                detections_per_img=args.detections_per_img, fg_iou_thresh=args.fg_iou_thresh, \
                                bg_iou_thresh=args.bg_iou_thresh, topk_candidates=args.topk_candidates)
    # # model = RetinaNet(backbone,num_classes=2, anchor_generator=anchor_generator)
    # # val_ds = build(args.data_dir, dataidxs=dataidxs, train=False)
    # print(val_ds.__getitem__(0))
    # train_dl, test_dl = get_dataloader_coco_v2(args, args.batch_size, args.batch_size_test, dataidxs=dataidxs)
    # data_dir, annotations = coco_path("train", args.data_dir)
    # train_dl = DataIterator(data_dir, args.resize, args.max_size, args.batch_size, 128, annotations, training=True, dataidxs=dataidx_map)
    train_dl, test_dl = get_dataloader_coco_v2(args, dataidxs, dataidxs_test)
    print(len(test_dl))
    local_trainer = GKTLocalTrainer(0, 0, train_dl, test_dl, 0, model, args)
    # a, b, c, d, e = local_trainer.train()
    # # x = [torch.rand(3, 300, 400),  torch.rand(3, 500, 400)]
    # # detection, logits, features = model(x)
    # # features, losses, logits = model(x)
    result = local_trainer.train()
    print(result)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser('test', parents=[get_args_parser()])
    args = parser.parse_args()
    run(args)
