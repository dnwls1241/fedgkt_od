import argparse
import torch
import torchvision
from .GKTRetinanet import gktservermodel, gktclientmodel
# from ...data_preprocessing.coco.data_loader import load_partition_data_distributed_coco, get_dataloader_coco
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection import RetinaNet
import json


def get_args_parser():
    parser = argparse.ArgumentParser('Set model', add_help=False)
    
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
    parser.add_argument('--weight_path', default=None)
    parser.add_argument('--num_classes', default=91)

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
    backbone = torchvision.models.mobilenet_v2(pretrained=True).features

    # anchor_generator = AnchorGenerator(
    #     sizes = ((32, 64, 128, 256, 512),),
    #     aspect_ratios=((0.5, 1.0, 2.0),)
    # )

    model = gktclientmodel(backbone=backbone, pretrained=args.pretrained, \
                                path=args.weight_path, num_classes=args.num_classes, image_mean=args.image_mean, \
                                image_std=args.image_std, score_thresh=args.score_thresh, nms_thresh=args.nms_thresh, \
                                detections_per_img=args.detections_per_img, fg_iou_thresh=args.fg_iou_thresh, \
                                bg_iou_thresh=args.bg_iou_thresh, topk_candidates=args.topk_candidates, is_server=args.is_server)
    # model = RetinaNet(backbone,num_classes=2, anchor_generator=anchor_generator)
   
    model.train()
    print(model)
    
    x = [torch.rand(3, 300, 400),  torch.rand(3, 500, 400)]
    # detection, logits, features = model(x)
    features, losses, logits = model(x)
    print(losses)
    print(logits)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser('test', parents=[get_args_parser()])
    args = parser.parse_args()
    run(args)
