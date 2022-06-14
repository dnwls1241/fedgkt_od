import torch
from torchvision.models.detection.retinanet import RetinaNet, RetinaNetHead, RetinaNetClassificationHead, RetinaNetRegressionHead
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone, _validate_trainable_layers
from torchvision.models.detection._utils import overwrite_eps
from . import utils
from torchvision.ops.focal_loss import sigmoid_focal_loss
from torchvision.ops import boxes as box_ops
from .utils import softmax_focal_loss, kd_l2_loss

from torch import nn, Tensor
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict
import warnings


class GKTHead(nn.Module):
    def __init__(self, in_channels, num_anchors, num_classes, is_server, 
    server_chan, cls_alpha, reg_alpha):
        super().__init__()
        if is_server:
            origin_channels = in_channels
            in_channels = server_chan*in_channels
        self.classification_head = GKTClassificationHead(in_channels, num_anchors, num_classes, cls_alpha)
        self.regression_head = GKTRegressionHead(in_channels, num_anchors, reg_alpha)
        if is_server:
            self.classification_head.conv = nn.Sequential(nn.Conv2d(origin_channels, in_channels, kernel_size=3, stride=1, padding=1), self.classification_head.conv)
            self.regression_head.conv = nn.Sequential(nn.Conv2d(origin_channels, in_channels, kernel_size=3, stride=1, padding=1), self.regression_head.conv)

    def compute_loss(self, targets, head_outputs, anchors, matched_idxs, input_logits=None):
        if input_logits is not None:            
            cls_logits = input_logits['cls_logits']
            bbox_regression = input_logits['bbox_regression']
        else:
            cls_logits = None
            bbox_regression = None
        return {
            'classification': self.classification_head.compute_loss(targets, head_outputs, matched_idxs, cls_logits),
            'bbox_regresstion': self.regression_head.compute_loss(targets, head_outputs, anchors, matched_idxs, bbox_regression)
        }

    def forward(self, x):
        # type: (List[Tensor]) -> Dict[str, Tensor]
        return {
            'cls_logits': self.classification_head(x),
            'bbox_regression': self.regression_head(x)
        }

    


def _sum(x: List[Tensor]) -> Tensor:
    res = x[0]
    for i in x[1:]:
        res = res + i
    return res

class GKTClassificationHead(RetinaNetClassificationHead):
    def __init__(self, in_channels, num_anchors, num_classes, alpha):
        super().__init__(in_channels, num_anchors, num_classes)
        self.alpha = alpha
        self.KL_Loss = utils.KL_Loss()
    def compute_loss(self, targets, head_outputs, matched_idxs, input_logits=None):
    #    # type: (List[Dict[str, Tensor]], Dict[str, Tensor], List[Tensor]) -> Tensor
        losses = []

        cls_logits = head_outputs['cls_logits']
        
        idx = 0
        for targets_per_image, cls_logits_per_image, matched_idxs_per_image in zip(targets, cls_logits, matched_idxs): 
            # determine only the foreground
            foreground_idxs_per_image = matched_idxs_per_image >= 0
            num_foreground = foreground_idxs_per_image.sum()

            # create the target classification
            gt_classes_target = torch.zeros_like(cls_logits_per_image)
            gt_classes_target[
                foreground_idxs_per_image,
                targets_per_image['labels'][matched_idxs_per_image[foreground_idxs_per_image]]
            ] = 1.0

            # find indices for which anchors should be ignored
            valid_idxs_per_image = matched_idxs_per_image != self.BETWEEN_THRESHOLDS

            # compute the classification loss
            if input_logits is not None:
                dstil_logits_per_image = input_logits[idx]
                losses.append(((sigmoid_focal_loss(
                    cls_logits_per_image[valid_idxs_per_image],
                    gt_classes_target[valid_idxs_per_image],
                    reduction='sum',
                )) + self.alpha * (softmax_focal_loss(
                    cls_logits_per_image[valid_idxs_per_image],
                    dstil_logits_per_image[valid_idxs_per_image],
                    reduction='sum'))) / max(1, num_foreground))
                idx += 1
            else:
                losses.append((sigmoid_focal_loss(cls_logits_per_image[valid_idxs_per_image],
                    gt_classes_target[valid_idxs_per_image],
                    reduction='sum',
                )) / max(1, num_foreground))

        return _sum(losses) / len(targets)

class GKTRegressionHead(RetinaNetRegressionHead):
    def __init__(self, in_channels, num_anchors, alpha):
        super().__init__(in_channels, num_anchors)
        self.alpha = alpha
    
    def compute_loss(self, targets, head_outputs, anchors, matched_idxs, input_logits=None):
    #    # type: (List[Dict[str, Tensor]], Dict[str, Tensor], List[Tensor], List[Tensor]) -> Tensor
        losses = []

        bbox_regression = head_outputs['bbox_regression']

        idx = 0
        for targets_per_image, bbox_regression_per_image, anchors_per_image, matched_idxs_per_image in \
                zip(targets, bbox_regression, anchors, matched_idxs):
            # determine only the foreground indices, ignore the rest
            foreground_idxs_per_image = torch.where(matched_idxs_per_image >= 0)[0]
            num_foreground = foreground_idxs_per_image.numel()

            # select only the foreground boxes
            matched_gt_boxes_per_image = targets_per_image['boxes'][matched_idxs_per_image[foreground_idxs_per_image]]
            bbox_regression_per_image = bbox_regression_per_image[foreground_idxs_per_image, :]
            anchors_per_image = anchors_per_image[foreground_idxs_per_image, :]


            # compute the regression targets
            target_regression = self.box_coder.encode_single(matched_gt_boxes_per_image, anchors_per_image)

            # compute the loss
            if input_logits is not None:
                dstil_logits_per_image = input_logits[idx][foreground_idxs_per_image, :]
                losses.append((torch.nn.functional.l1_loss(
                    bbox_regression_per_image,
                    target_regression,
                    reduction='sum'
                ) + self.alpha*kd_l2_loss(
                    bbox_regression_per_image,
                    dstil_logits_per_image,
                    target_regression,
                    reduction='sum'
                ))/ max(1, num_foreground))
                idx += 1
            else:
                losses.append(torch.nn.functional.l1_loss(
                bbox_regression_per_image,
                target_regression,
                reduction='sum'
            ) / max(1, num_foreground))

        return _sum(losses) / max(1, len(targets))


class GKTRetinaNet(RetinaNet):
    def __init__(self, backbone, num_classes,
                 # transform parameters
                 min_size=800, max_size=1333,
                 image_mean=None, image_std=None,
                 # Anchor parameters
                 anchor_generator=None, head=None,
                 proposal_matcher=None,
                 score_thresh=0.05,
                 nms_thresh=0.5,
                 detections_per_img=300,
                 fg_iou_thresh=0.5, bg_iou_thresh=0.4,
                 topk_candidates=1000,
                 is_server=False,
                 server_chan=2, cls_alpha=0.5, reg_alpha=0.5):
        if anchor_generator is None:
            anchor_generator = AnchorGenerator()

        if head is None:
            head = GKTHead(backbone.out_channels, anchor_generator.num_anchors_per_location()[0],
             num_classes, is_server, server_chan, cls_alpha, reg_alpha)
        super().__init__(backbone=backbone, num_classes=num_classes,
                min_size=min_size, max_size=max_size,
                image_mean=image_mean, image_std=image_std,
                anchor_generator=anchor_generator, head=head,
                proposal_matcher=proposal_matcher,
                score_thresh=score_thresh,
                nms_thresh=nms_thresh,
                detections_per_img=detections_per_img,
                fg_iou_thresh=fg_iou_thresh,bg_iou_thresh=bg_iou_thresh,
                topk_candidates=topk_candidates)
        self.is_server = is_server

    def compute_loss(self, targets, head_outputs, anchors, input_logits=None):
        matched_idxs = []
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            if targets_per_image['boxes'].numel() == 0:
                matched_idxs.append(torch.full((anchors_per_image.size(0),), -1, dtype=torch.int64,
                                               device=anchors_per_image.device))
                continue
            false_boxes = targets_per_image['boxes'] < 0
            targets_per_image['boxes'][false_boxes] = 0
            match_quality_matrix = box_ops.box_iou(targets_per_image['boxes'], anchors_per_image)
            matched_idxs.append(self.proposal_matcher(match_quality_matrix))

        return self.head.compute_loss(targets, head_outputs, anchors, matched_idxs, input_logits)

    def forward(self, input, targets=None, input_logits=None, eval_loss=False):
    #    # type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        """
        Args:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        # targets = [{"boxes": t[:, :-1], "labels": t[:, -1].type(torch.int64)} for t in targets] 
        
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        if self.training:
            assert targets is not None
            for target in targets:
                boxes = target["boxes"]
                if isinstance(boxes, torch.Tensor):
                    if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                        raise ValueError("Expected target boxes to be a tensor"
                                         "of shape [N, 4], got {:}.".format(
                                             boxes.shape))
                else:
                    raise ValueError("Expected target boxes to be of type "
                                     "Tensor, got {:}.".format(type(boxes)))


        # Check for degenerate boxes
        # TODO: Move this to a function
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # print the first degenerate box
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    raise ValueError("All bounding boxes should have positive height and width."
                                     " Found invalid box {} for target at index {}."
                                     .format(degen_bb, target_idx))

    
        images = input
        # get the original image sizes
        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        # transform the input
        images, targets = self.transform(images, targets)
        # get the features from the backbone
        features = self.backbone(images.tensors)
    
        if isinstance(features, torch.Tensor):
            features = OrderedDict([('0', features)])

        # TODO: Do we want a list or a dict?
        features = list(features.values())

        # compute the retinanet heads outputs using the features
        head_outputs = self.head(features)
        output_logits = head_outputs
        # create the set of anchors
        anchors = self.anchor_generator(images, features)

        losses = {}
        detections: List[Dict[str, Tensor]] = []
        if self.training or eval_loss:
            assert targets is not None

            # compute the losses
            losses = self.compute_loss(targets, head_outputs, anchors, input_logits)
            detections = losses
        else:
            # recover level sizes
            num_anchors_per_level = [x.size(2) * x.size(3) for x in features]
            HW = 0
            for v in num_anchors_per_level:
                HW += v
            HWA = head_outputs['cls_logits'].size(1)
            A = HWA // HW
            num_anchors_per_level = [hw * A for hw in num_anchors_per_level]

            # split outputs per level
            split_head_outputs: Dict[str, List[Tensor]] = {}
            for k in head_outputs:
                split_head_outputs[k] = list(head_outputs[k].split(num_anchors_per_level, dim=1))
            split_anchors = [list(a.split(num_anchors_per_level)) for a in anchors]

            # compute the detections
            detections = self.postprocess_detections(split_head_outputs, split_anchors, images.image_sizes)
            detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn("RetinaNet always returns a (Losses, Detections) tuple in scripting")
                self._has_warned = True
            if self.is_server:
                return losses, output_logits, detections
            else:
                return features, losses, output_logits, detections
        return self.eager_outputs(losses, output_logits, detections, features)

    @torch.jit.unused
    def eager_outputs(self, losses, output_logits, detections, features=None):
    #   # type: (Dict[str, Tensor], List[Dict[str, Tensor]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        if self.training:
            if self.is_server:
                return losses, output_logits
            else:
                return losses, output_logits, features
        else:
            if self.is_server:
                return detections, output_logits
            else:
                return detections, output_logits, features


def gktservermodel(backbone=None, pretrained=False, path=None, anchor_generator=None,
                           num_classes=91, pretrained_backbone=True, trainable_backbone_layers=None, backbone_name="resnet18", **kwargs):

    trainable_backbone_layers = _validate_trainable_layers(
        pretrained_backbone, trainable_backbone_layers, 5, 3)

    if pretrained:
        # no need to download the backbone if pretrained is set
        pretrained_backbone = False
    # skip P2 because it generates too many anchors (according to their paper)
    returned_layers=[2, 3, 4]
    anchor_generator = AnchorGenerator(sizes=tuple([(64, 128, 256) for _ in range(len(returned_layers)+1)]), aspect_ratios=tuple([(1.0, 2.0) for _ in range(len(returned_layers)+1)]))
    if backbone is None:
        backbone = resnet_fpn_backbone(backbone_name=backbone_name, pretrained=pretrained_backbone, returned_layers=[2, 3, 4], trainable_layers=trainable_backbone_layers)
    model = GKTRetinaNet(backbone, num_classes, anchor_generator=anchor_generator, is_server=True, **kwargs)
    if path is not None:
        model.load_state_dict(torch.load(path))
        print('load model', path)
        overwrite_eps(model, 0.0)
    return model

def gktclientmodel(backbone=None, pretrained=False, path=None, anchor_generator=None,
                           num_classes=91, pretrained_backbone=True, trainable_backbone_layers=None, backbone_name="resnet18", **kwargs):

    trainable_backbone_layers = _validate_trainable_layers(pretrained_backbone, trainable_backbone_layers, 5, 3)

    if pretrained:
        # no need to download the backbone if pretrained is set
        pretrained_backbone = False
    # skip P2 because it generates too many anchors (according to their paper)
    returned_layers=[2, 3, 4]
    if backbone is None:
        backbone = resnet_fpn_backbone(backbone_name=backbone_name, pretrained=pretrained_backbone, returned_layers=returned_layers, trainable_layers=trainable_backbone_layers)
    anchor_generator = AnchorGenerator(sizes=tuple([(64, 128, 256) for _ in range(len(returned_layers)+1)]), aspect_ratios=tuple([(1.0, 2.0) for _ in range(len(returned_layers)+1)]))
    #head = RetinaNetHead(args.head_channels, anchor_generator.num_anchors_per_location()[0], num_classes)
    model = GKTRetinaNet(backbone, num_classes, anchor_generator=anchor_generator, **kwargs)
    if path is not None:
        model.load_state_dict(torch.load(path))
        overwrite_eps(model, 0.0)
    return model