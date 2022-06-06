import torch
from torchvision.models.detection.retinanet import RetinaNet, RetinaNetHead, RetinaNetClassificationHead, RetinaNetRegressionHead
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone, _validate_trainable_layers
from torchvision.models.detection._utils import overwrite_eps
from . import utils
from torchvision.ops.focal_loss import sigmoid_focal_loss
from .utils import softmax_focal_loss

from torch import nn, Tensor
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict
import warnings


class GKTHead(RetinaNetHead):
    def __init__(self, in_channels, num_anchors, num_classes):
        super().__init__(in_channels, num_anchors, num_classes)
        self.classification_head = GKTClassificationHead(in_channels, num_anchors, num_classes)
        self.regression_head = GKTRegressionHead(in_channels, num_anchors)
        
    def compute_loss(self, targets, head_outputs, anchors, matched_idxs, output_logits=None):
        cls_logits = output_logits['cls_logits']
        bbox_regression = output_logits['bbox_regression']
        return {
            'classification': self.classification_head.compute_loss(targets, head_outputs, matched_idxs, cls_logits),
            'bbox_regresstion': self.regression_head.compute_loss(targets, head_outputs, anchors, matched_idxs, bbox_regression)
        }
    


def _sum(x: List[Tensor]) -> Tensor:
    res = x[0]
    for i in x[1:]:
        res = res + i
    return res

class GKTClassificationHead(RetinaNetClassificationHead):
    def __init__(self, in_channels, num_anchors, num_classes):
        super().__init__(in_channels, num_anchors, num_classes)
        self.KL_Loss = utils.KL_Loss()
    def compute_loss(self, targets, head_outputs, matched_idxs, output_logits=None):
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
            if output_logits is not None:
                dstil_logits_per_image = output_logits[idx]
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
    def __init__(self, in_channels, num_anchors):
        super().__init__(in_channels, num_anchors)
    
    def compute_loss(self, targets, head_outputs, anchors, matched_idxs, output_logits=None):
    #    # type: (List[Dict[str, Tensor]], Dict[str, Tensor], List[Tensor], List[Tensor]) -> Tensor
        losses = []

        bbox_regression = head_outputs['bbox_regression']

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
                 is_server=False):
        if anchor_generator is None:
            anchor_generator = AnchorGenerator()
        print(backbone.out_channels, anchor_generator.num_anchors_per_location()[0], num_classes)
        if head is None:
            head = GKTHead(backbone.out_channels, anchor_generator.num_anchors_per_location()[0], num_classes)
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


    def forward(self, input, targets=None, intput_logits=None):
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
        targets = {"boxes": targets[:, :, :-1], "labels": targets[:, :, -1]} 
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

        if self.is_server and self.training:
            features = input
        else:
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
        if self.training:
            assert targets is not None

            # compute the losses
            losses = self.compute_loss(targets, head_outputs, anchors, intput_logits)
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
                           num_classes=91, pretrained_backbone=True, trainable_backbone_layers=None, **kwargs):

    trainable_backbone_layers = _validate_trainable_layers(
        pretrained or pretrained_backbone, trainable_backbone_layers, 5, 3)

    if pretrained:
        # no need to download the backbone if pretrained is set
        pretrained_backbone = False
    # skip P2 because it generates too many anchors (according to their paper)
    if backbone is None:
        backbone = resnet_fpn_backbone(backbone, pretrained_backbone, returned_layers=[2, 3, 4], trainable_layers=trainable_backbone_layers)
    model = GKTRetinaNet(backbone, num_classes, anchor_generator=anchor_generator, **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(path))
        overwrite_eps(model, 0.0)
    return model

def gktclientmodel(backbone=None, pretrained=False, path=None, anchor_generator=None,
                           num_classes=91, pretrained_backbone=True, trainable_backbone_layers=None, **kwargs):

    trainable_backbone_layers = _validate_trainable_layers(pretrained_backbone, trainable_backbone_layers, 5, 3)

    if pretrained:
        # no need to download the backbone if pretrained is set
        pretrained_backbone = False
    # skip P2 because it generates too many anchors (according to their paper)
    if backbone is None:
        backbone = resnet_fpn_backbone(backbone_name=backbone, pretrained=pretrained_backbone, returned_layers=[2, 3, 4], trainable_layers=trainable_backbone_layers)
    #head = RetinaNetHead(args.head_channels, anchor_generator.num_anchors_per_location()[0], num_classes)
    print(backbone.out_channels)
    model = GKTRetinaNet(backbone, num_classes, anchor_generator=anchor_generator, **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(path))
        overwrite_eps(model, 0.0)
    return model