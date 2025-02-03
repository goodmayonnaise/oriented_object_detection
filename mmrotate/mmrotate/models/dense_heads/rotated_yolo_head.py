# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import math
from typing import List, Sequence, Tuple, Union

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
import torch.nn.functional as F

from ..builder import ROTATED_HEADS, build_loss
from ..blocks import *

from mmengine.model import BaseModule
from mmengine.model import bias_init_with_prob
from mmcv.cnn import ConvModule
from mmcv.cnn import Scale
from mmcv.runner import force_fp32
from mmcv.cnn.bricks import build_norm_layer
from mmdet.core import multi_apply, reduce_mean
from mmdet.core.anchor.point_generator import MlvlPointGenerator
from mmdet.models.dense_heads.base_dense_head import BaseDenseHead
from mmrotate.core import build_bbox_coder, multiclass_nms_rotated

INF = 1e8

def make_divisible(x: float,
                   widen_factor: float = 1.0,
                   divisor: int = 8) -> int:
    """Make sure that x*widen_factor is divisible by divisor."""
    return math.ceil(x * widen_factor / divisor) * divisor

@ROTATED_HEADS.register_module()
class RotatedYOLOv6Head(BaseDenseHead):
    """Anchor-Free Rotated Yolov6 Head predicting four side distances (without angle loss)"""
    def __init__(self,
                 num_classes : int = 15,
                 in_channels : Union[int, Sequence] = [256, 512, 1024],
                 widen_factor: float = 1.0,
                 reg_max=0,
                 featmap_strides: Sequence[int] = (8, 16, 32),
                 regress_ranges=((-1, 64), (64, 128), (128, 256)),
                 bbox_coder=dict(type='DistanceAnglePointCoder', angle_version='le90'),
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox=dict(type='IoULoss', loss_weight=1.0),
                 matching:int=0,
                 debug:bool=False,
                #  loss_angle=dict(type='L1Loss', loss_weight=1.0),
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 init_cfg=dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='cls_preds',
                         std=0.01,
                         bias_prob=0.01)),               
                 train_cfg=None,
                 test_cfg=None,
                 center_sample_radius=1.5):
        super().__init__(init_cfg)

        self.num_classes = num_classes
        self.featmap_strides = featmap_strides
        self.num_levels = len(self.featmap_strides)
        self.reg_max = reg_max
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        if self.use_sigmoid_cls:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        # self.loss_angle = build_loss(loss_angle)
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.prior_generator = MlvlPointGenerator(featmap_strides)
        # In order to keep a more general interface and be consistent with
        # anchor_head. We can think of point like one anchor
        self.num_base_priors = self.prior_generator.num_base_priors[0]
        self.regress_ranges = regress_ranges
        self.center_sample_radius = center_sample_radius
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        
        self.matching = matching
        self.debug = debug

        # if isinstance(in_channels, int):
        #     self.in_channels = [int(in_channels * widen_factor)
        #                         ] * self.num_levels
        # else:
        #     self.in_channels = [int(i * widen_factor) for i in in_channels]
        self.in_channels = in_channels
        in_channels = []
        for channel in self.in_channels:
            channel = make_divisible(channel, widen_factor)
            in_channels.append(channel)
        self.in_channels = in_channels
        self._init_layers()

    def _init_layers(self):
        """initialize conv layers in YOLOv6 head."""
        # Init decouple head
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.ang_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.ang_preds = nn.ModuleList()
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.featmap_strides])
        self.stems = nn.ModuleList()

        if self.reg_max > 1:
            proj = torch.arange(
                self.reg_max + self.num_base_priors, dtype=torch.float)
            self.register_buffer('proj', proj, persistent=False)

        for i in range(self.num_levels):
            self.stems.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.in_channels[i],
                    kernel_size=1,
                    stride=1,
                    padding=1 // 2,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
            
            self.cls_convs.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.in_channels[i],
                    kernel_size=3,
                    stride=1,
                    padding=3 // 2,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
            self.reg_convs.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.in_channels[i],
                    kernel_size=3,
                    stride=1,
                    padding=3 // 2,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
            #rotated yolov6
            self.ang_convs.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.in_channels[i],
                    kernel_size=3,
                    stride=1,
                    padding=3 // 2,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
            
            self.cls_preds.append(
                nn.Conv2d(
                    in_channels=self.in_channels[i],
                    out_channels=self.num_base_priors * self.num_classes,
                    kernel_size=1))
            self.reg_preds.append(
                nn.Conv2d(
                    in_channels=self.in_channels[i],
                    out_channels=(self.num_base_priors + self.reg_max) * 4,
                    kernel_size=1))
            #rotated yolov6
            self.ang_preds.append(
                nn.Conv2d(
                    in_channels=self.in_channels[i],
                    out_channels=(self.num_base_priors + self.reg_max) * 1,
                    kernel_size=1))

    def init_weights(self):
        super().init_weights()
        bias_init = bias_init_with_prob(0.01)
        for conv in self.cls_preds:
            conv.bias.data.fill_(bias_init)
            conv.weight.data.fill_(0.)

        for conv in self.reg_preds:
            conv.bias.data.fill_(1.0)
            conv.weight.data.fill_(0.)

        for conv in self.ang_preds:
            conv.bias.data.fill_(1.0)
            conv.weight.data.fill_(0.)

    def forward(self, x: Tuple[Tensor]) -> Tuple[List]:
        """Forward features from the upstream network.

        Args:
            x (Tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
        Returns:
            Tuple[List]: A tuple of multi-level classification scores, bbox
            predictions.
        """
        assert len(x) == self.num_levels
        return multi_apply(self.forward_single, x, self.stems, self.cls_convs,
                           self.cls_preds, self.reg_convs, self.reg_preds, 
                           self.ang_convs, self.ang_preds, self.scales, self.featmap_strides)

    def forward_single(self, x: Tensor, stem: nn.Module, cls_conv: nn.Module,
                       cls_pred: nn.Module, reg_conv: nn.Module,
                       reg_pred: nn.Module, ang_conv: nn.Module,
                       ang_pred: nn.Module, scale: List, stride) -> Tuple[Tensor, Tensor]:
        """Forward feature of a single scale level."""
        b, _, h, w = x.shape
        y = stem(x)
        cls_x = y
        reg_x = y
        ang_x = y
        cls_feat = cls_conv(cls_x)
        reg_feat = reg_conv(reg_x)
        ang_feat = ang_conv(ang_x)

        cls_score = cls_pred(cls_feat)
        bbox_dist_preds = reg_pred(reg_feat)
        predicted_angle = ang_pred(ang_feat)

        if self.reg_max > 1:
            bbox_dist_preds = bbox_dist_preds.reshape(
                [-1, 4, self.reg_max + self.num_base_priors,
                 h * w]).permute(0, 3, 1, 2)

            # TODO: The get_flops script cannot handle the situation of
            #  matmul, and needs to be fixed later
            # bbox_preds = bbox_dist_preds.softmax(3).matmul(self.proj)
            bbox_preds = bbox_dist_preds.softmax(3).matmul(
                self.proj.view([-1, 1])).squeeze(-1)
            bbox_preds = bbox_preds.transpose(1, 2).reshape(b, -1, h, w)
        else:
            bbox_preds = scale(bbox_dist_preds).float()
            bbox_preds = bbox_preds.clamp(min=0)
            if not self.training:
                bbox_preds *= stride
        if self.training:
            return cls_score, bbox_preds, predicted_angle
        else:
            return cls_score, bbox_preds, predicted_angle

    def special_init(self):
        """Since YOLO series algorithms will inherit from YOLOv5Head, but
        different algorithms have special initialization process.

        The special_init function is designed to deal with this situation.
        """
        if self.train_cfg:
            # Add common attributes to reduce calculation
            self.featmap_sizes_train = None
            self.num_level_priors = None
            self.flatten_priors_train = None
            self.stride_tensor = None

    @force_fp32(
        apply_to=('cls_scores', 'bbox_preds', 'angle_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             angle_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute loss of the head.
        Args:
            cls_scores (list[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_points * num_classes.
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_points * 4.
            angle_preds (list[Tensor]): Box angle for each scale level, \
                each is a 4D-tensor, the channel number is num_points * 1.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert len(cls_scores) == len(bbox_preds) == len(angle_preds)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=bbox_preds[0].dtype,
            device=bbox_preds[0].device)
        
        labels, bbox_targets, angle_targets = self.get_targets(
            all_level_points, gt_bboxes, gt_labels, 
            bbox_preds, angle_preds, cls_scores)
        
        num_imgs = cls_scores[0].size(0)
        # flatten_cls_scores = [
        #     cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
        #     for cls_score in cls_scores
        # ]
        
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, cls_score.shape[1])
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_angle_preds = [
            angle_pred.permute(0, 2, 3, 1).reshape(-1, 1)
            for angle_pred in angle_preds
        ]

        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_angle_preds = torch.cat(flatten_angle_preds)
        flatten_labels = torch.cat(labels)
        flatten_bbox_targets = torch.cat(bbox_targets)
        flatten_angle_targets = torch.cat(angle_targets)
        
        # repeat points to align with bbox_preds
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points])

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((flatten_labels >= 0)
                    & (flatten_labels < bg_class_ind)).nonzero().reshape(-1)
        num_pos = torch.tensor(
            len(pos_inds), dtype=torch.float, device=bbox_preds[0].device)
        num_pos = max(reduce_mean(num_pos), 1.0)
        # loss_cls = self.loss_cls(
        #     flatten_cls_scores, flatten_labels, avg_factor=num_pos)

        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_angle_preds = flatten_angle_preds[pos_inds]

        pos_bbox_targets = flatten_bbox_targets[pos_inds]
        pos_angle_targets = flatten_angle_targets[pos_inds]

        # la 9-2 -> loss varify focal 변경 전 원본 loss code 
        # if len(pos_inds) > 0:
            # pos_points = flatten_points[pos_inds]
            # bbox_coder = self.bbox_coder
            # pos_bbox_preds = torch.cat([pos_bbox_preds, pos_angle_preds],
            #                             dim=-1)
            # pos_bbox_targets = torch.cat(
            #     [pos_bbox_targets, pos_angle_targets], dim=-1)
            # pos_decoded_bbox_preds = bbox_coder.decode(pos_points,
            #                                            pos_bbox_preds)
            # pos_decoded_target_preds = bbox_coder.decode(
            #     pos_points, pos_bbox_targets)
            # loss_bbox = self.loss_bbox(
            #     pos_decoded_bbox_preds,
            #     pos_decoded_target_preds)
        # else:
        #     loss_bbox = pos_bbox_preds.sum()

        # return dict(
        #         loss_cls=loss_cls,
        #         loss_bbox=loss_bbox)
        

        if len(pos_inds) > 0:
            pos_points = flatten_points[pos_inds]
            bbox_coder = self.bbox_coder
            pos_bbox_preds = torch.cat([pos_bbox_preds, pos_angle_preds],
                                        dim=-1)
            pos_bbox_targets = torch.cat(
                [pos_bbox_targets, pos_angle_targets], dim=-1)
            pos_decoded_bbox_preds = bbox_coder.decode(pos_points,
                                                       pos_bbox_preds)
            pos_decoded_target_preds = bbox_coder.decode(
                pos_points, pos_bbox_targets)
            loss_bbox = self.loss_bbox(
                pos_decoded_bbox_preds,
                pos_decoded_target_preds)
            
            if self.loss_cls.__str__() == 'VarifocalLoss()':                
                cls_iou_targets = torch.zeros_like(flatten_cls_scores)
                bbox_overlap = build_loss(dict(type='RotatedIoULoss', loss_weight=1.0, reduction='none', mode='linear')) ##
                iou_targets = 1 - bbox_overlap(
                    pos_decoded_bbox_preds,
                    pos_decoded_target_preds.detach()).clamp(min=1e-6)
                
                cls_iou_targets[pos_inds, flatten_labels[pos_inds]] = iou_targets
                loss_cls = self.loss_cls(
                    flatten_cls_scores,
                    cls_iou_targets,
                    avg_factor=num_pos)
            else:     
                if 'MultiLoss' in self.loss_cls.__str__():
                    losses = self.loss_cls(flatten_cls_scores, flatten_labels, avg_factor=num_pos)
                    
                    if len(losses) == 3:
                        loss_cls, loss_cls1, loss_cls2 = losses
                        return dict(
                            loss_cls=loss_cls,
                            loss_cls1=loss_cls1,
                            loss_cls2=loss_cls2,
                            loss_bbox=loss_bbox
                        )
                    elif len(losses) == 4:
                        loss_cls, loss_cls1, loss_cls2, loss_cls3 = losses
                        return dict(
                            loss_cls=loss_cls,
                            loss_cls1=loss_cls1,
                            loss_cls2=loss_cls2,
                            loss_cls3=loss_cls3,
                            loss_bbox=loss_bbox
                        )
                else:
                    loss_cls = self.loss_cls(flatten_cls_scores, flatten_labels, avg_factor=num_pos)
                    # if len(self.loss_cls.loss_type) == 2:
                    #     loss_cls, loss_cls1, loss_cls2 = self.loss_cls(flatten_cls_scores, flatten_labels, avg_factor=num_pos)
                    #     return dict(
                    #         loss_cls=loss_cls,
                    #         loss_cls1=loss_cls1,
                    #         loss_cls2=loss_cls2,
                    #         loss_bbox=loss_bbox
                    #     )
                    # elif len(self.loss_csl.loss_type) == 3:
                    #     loss_cls, loss_cls1, loss_cls2, loss_cls3 = self.loss_cls(flatten_cls_scores, flatten_labels, avg_factor=num_pos)
                    #     return dict(
                    #         loss_cls=loss_cls,
                    #         loss_cls1=loss_cls1,
                    #         loss_cls2=loss_cls2,
                    #         loss_cls3=loss_cls3,
                    #         loss_bbox=loss_bbox
                    #     )

        else:
            if self.loss_cls.__str__() == 'VarifocalLoss()':
                flatten_labels = torch.zeros_like(flatten_cls_scores)
            loss_cls = self.loss_cls(
                flatten_cls_scores, flatten_labels, avg_factor=num_pos)
            loss_bbox = pos_bbox_preds.sum()

        return dict(
                loss_cls=loss_cls,
                loss_bbox=loss_bbox)
    
    def get_targets(self, points, gt_bboxes_list, gt_labels_list, 
                    bbox_preds, angle_preds, cls_scores):

        assert len(points) == len(self.regress_ranges)
        num_levels = len(points)
        # expand regress ranges to align with points
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i])[None].expand_as(
                points[i]) for i in range(num_levels)
        ]
        # concat all levels points and regress ranges
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)
        
        '''labelassignment'''
        bn = bbox_preds[0].shape[0]
        rbbox_preds = [
            torch.cat([b.view(bn, 4, -1).permute(0,2,1), 
                       a.view(bn, 1, -1).permute(0,2,1)], dim=-1)
            for b, a in zip(bbox_preds, angle_preds)
        ]
        # rbbox_probs = [torch.cat([p.view(bn, 15, -1).permute(0,2,1)], dim=-1) for p in cls_scores]
        rbbox_probs = [torch.cat([p.view(bn, p.shape[1], -1).permute(0,2,1)], dim=-1) for p in cls_scores]
        concat_rbboxes = [preds for preds in torch.cat(rbbox_preds, dim=1)]
        concat_probs = [probs for probs in torch.cat(rbbox_probs, dim=1)]
        '''---------------'''
        
        # the number of points per img, per lvl
        num_points = [center.size(0) for center in points]
        # get labels and bbox_targets of each image
        labels_list, bbox_targets_list, angle_targets_list = multi_apply(
            self._get_target_single,
            gt_bboxes_list,
            gt_labels_list,
            concat_rbboxes, 
            concat_probs, 
            points=concat_points,  
            regress_ranges=concat_regress_ranges, 
            num_points_per_lvl= num_points)
        
        # split to per img, per level
        
        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        bbox_targets_list = [
            bbox_targets.split(num_points, 0)
            for bbox_targets in bbox_targets_list
        ]
        angle_targets_list = [
            angle_targets.split(num_points, 0)
            for angle_targets in angle_targets_list
        ]

        # concat per level image
        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        concat_lvl_angle_targets = []
        for i in range(num_levels):
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list]))
            bbox_targets = torch.cat(
                [bbox_targets[i] for bbox_targets in bbox_targets_list])
            angle_targets = torch.cat(
                [angle_targets[i] for angle_targets in angle_targets_list])
            bbox_targets = bbox_targets / self.featmap_strides[i]
            concat_lvl_bbox_targets.append(bbox_targets)
            concat_lvl_angle_targets.append(angle_targets)
        return (concat_lvl_labels, concat_lvl_bbox_targets,
                concat_lvl_angle_targets)

    def _get_target_single(self, gt_bboxes, gt_labels, bbox_preds, probs, points, regress_ranges,
                           num_points_per_lvl):
        """Compute regression, classification and angle targets for a single
        image."""
        RIoU = build_loss(dict(type='RotatedIoULoss', loss_weight=1.0,
                               reduction='none', mode='linear')) ##
        num_points = points.size(0)
        num_gts = gt_labels.size(0)
        annotation_bbox = gt_bboxes
        if num_gts == 0:
            return gt_labels.new_full((num_points,), self.num_classes), \
                   gt_bboxes.new_zeros((num_points, 4)), \
                   gt_bboxes.new_zeros((num_points, 1))
        
        areas = gt_bboxes[:, 2] * gt_bboxes[:, 3] # wh
        # TODO: figure out why these two are different
        # areas = areas[None].expand(num_points, num_gts)
        areas = areas[None].repeat(num_points, 1)
        regress_ranges = regress_ranges[:, None, :].expand(
            num_points, num_gts, 2)
        points = points[:, None, :].expand(num_points, num_gts, 2)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 5)
        gt_ctr, gt_wh, gt_angle = torch.split(gt_bboxes, [2, 2, 1], dim=2)

        cos_angle, sin_angle = torch.cos(gt_angle), torch.sin(gt_angle)
        rot_matrix = torch.cat([cos_angle, sin_angle, -sin_angle, cos_angle],
                               dim=-1).reshape(num_points, num_gts, 2, 2)
        offset = points - gt_ctr
        offset = torch.matmul(rot_matrix, offset[..., None])
        offset = offset.squeeze(-1)

        w, h = gt_wh[..., 0], gt_wh[..., 1]
        offset_x, offset_y = offset[..., 0], offset[..., 1]
        left = w / 2 + offset_x
        right = w / 2 - offset_x
        top = h / 2 + offset_y
        bottom = h / 2 - offset_y
        bbox_targets = torch.stack((left, top, right, bottom), -1)

        # condition1: inside a gt bbox
        inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0
        # condition1: inside a `center bbox`
        radius = self.center_sample_radius
        stride = offset.new_zeros(offset.shape)
        bboxes = bbox_preds.clone().detach() ##
        
        # project the points on current lvl back to the `original` sizes
        lvl_begin = 0
        
        for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
            lvl_end = lvl_begin + num_points_lvl
            stride[lvl_begin:lvl_end] = self.featmap_strides[lvl_idx] * radius
            
            ''' bbox pred scale'''
            stride_bboxes = bboxes[lvl_begin:lvl_end, :4] * self.featmap_strides[lvl_idx]
            bboxes[lvl_begin:lvl_end, :4] = stride_bboxes
            
            lvl_begin = lvl_end

        inside_center_bbox_mask = (abs(offset) < stride).all(dim=-1)
        inside_gt_bbox_mask = torch.logical_and(inside_center_bbox_mask,
                                                inside_gt_bbox_mask)
        
        # condition2: limit the regression range for each location
        max_regress_distance = bbox_targets.max(-1)[0]
        inside_regress_range = (
            (max_regress_distance >= regress_ranges[..., 0])
            & (max_regress_distance <= regress_ranges[..., 1]))
    
        if self.matching == 0:
            '''
            Default
            '''
            # if there are still more than one objects for a location,
            # we choose the one with minimal area
            areas[inside_gt_bbox_mask == 0] = INF ## (1,2)
            areas[inside_regress_range == 0] = INF ## (3)
            min_area, min_area_inds = areas.min(dim=1)
            labels = gt_labels[min_area_inds] 
            labels[min_area == INF] = self.num_classes  # set as BG
            bbox_targets = bbox_targets[range(num_points), min_area_inds]
            angle_targets = gt_angle[range(num_points), min_area_inds]
    
        elif self.matching == 6:
            alpha = 0.5
            areas[inside_gt_bbox_mask == 0] = 0 ## (1,2)
            # areas[inside_regress_range == 0] = 0 ## (3)
            overlap_idx = torch.where(torch.bincount(torch.where(areas != 0)[0]) >= 2)[0] # 위에서 2개이상의 bbox가 할당된 det 인덱스
            if len(overlap_idx) != 0:
                pts = points[overlap_idx]
                iou_matrix = torch.zeros(len(overlap_idx), len(annotation_bbox), device=areas.device) 
                det_overlapbboxes = self.bbox_coder.decode(pts[:,0], bboxes[overlap_idx])
                det_probs = probs[overlap_idx].softmax(-1)
                mask = (areas[overlap_idx] != 0).float()
                for i, gt in enumerate(annotation_bbox):
                    cls = gt_labels[i]
                    gt = gt.expand(len(overlap_idx), 5)
                    iou =  1-(RIoU(det_overlapbboxes, gt)*(10**3)).round()/ (10**3)
                    prob = det_probs[:, cls]
                    iou_matrix[:,i] = alpha * iou + (1-alpha) * prob
                iou_matrix *= mask
                areas[overlap_idx] = iou_matrix
            max_area, max_area_inds = areas.max(dim=1) # 더 큰 IoU 선택
            labels = gt_labels[max_area_inds]
            labels[max_area == 0] = self.num_classes
            bbox_targets = bbox_targets[range(num_points), max_area_inds]
            angle_targets = gt_angle[range(num_points), max_area_inds]
            
        elif self.matching == 91:
            delta_x = 2*offset_x/w
            delta_y = 2*offset_y/h
            centerness_targets = 1 -torch.sqrt(((delta_x**2 + delta_y**2)+ 1e-8)/2)
            centerness_targets = centerness_targets.clamp_min(0)
        
            areas *= inside_gt_bbox_mask ## 1
            areas *= inside_regress_range ## 2
            
            # ingt_idx = torch.where(areas!=0)[0] # inside of GT bbox indicator
            det_probs = probs.softmax(-1)
            # matrix = torch.zeros(len(points[ingt_idx]), len(annotation_bbox), device=areas.device) 
            det_rbboxes = self.bbox_coder.decode(points[:,0], bboxes) #lrtba -> ctr wh a
            inside_topk_lvl = torch.zeros_like(inside_gt_bbox_mask).float()
            for i, gt in enumerate(annotation_bbox):
                matching_cost = centerness_targets[:,i].pow_(2) *\
                    (1-(RIoU(det_rbboxes, gt.expand(len(points), 5))*(10**3)).round()/ (10**3)).pow_(2) *\
                        det_probs[:, gt_labels[i]].rsqrt_() # matching matrix

                matching_cost *= inside_gt_bbox_mask[:,i]
                matching_cost *= inside_regress_range[:,i]
                
                if matching_cost.shape[0] >= 15:
                    k = 15
                else:
                    k = matching_cost.shape[0]
                    
                val, idx = matching_cost.sort(descending=True)
                idx = idx[:k]
                val = val[:k]
                inside_topk_lvl[idx,i] = val # cost가 0인 값은 False가 되도록
                
            areas *= inside_topk_lvl.bool()
                
            max_area, max_area_inds = areas.max(dim=1) # 겹치는 points를 늘린 후 max값 선택
            
            if 0 in max_area_inds.bincount().tolist(): # 0개의 anchorpoint가 할당된 경우
                noacp = torch.where(max_area_inds[max_area != 0].bincount() == 0)[0] # GT 중 match가 안된 gt ind
                max_area_inds[centerness_targets.argmax(dim=0)[noacp]] = noacp # ctr기반으로 할당
            
            labels = gt_labels[max_area_inds]
            labels[max_area == 0] = self.num_classes
            bbox_targets = bbox_targets[range(num_points), max_area_inds]
            angle_targets = gt_angle[range(num_points), max_area_inds]
            
        return labels, bbox_targets, angle_targets

    @force_fp32(
        apply_to=('cls_scores', 'bbox_preds', 'angle_preds'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   angle_preds,
                   img_metas,
                   cfg=None,
                   rescale=None):
        
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]

        mlvl_points = self.prior_generator.grid_priors(featmap_sizes,
                                                       bbox_preds[0].dtype,
                                                       bbox_preds[0].device)
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            angle_pred_list = [
                angle_preds[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            det_bboxes = self._get_bboxes_single(cls_score_list,
                                                 bbox_pred_list,
                                                 angle_pred_list,
                                                 mlvl_points, img_shape,
                                                 scale_factor, cfg, rescale)
            result_list.append(det_bboxes)
        return result_list

    def _get_bboxes_single(self,
                           cls_scores,
                           bbox_preds,
                           angle_preds,
                           mlvl_points,
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False):
        
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_points)
        mlvl_bboxes = []
        mlvl_scores = []
        for cls_score, bbox_pred, angle_pred, points in zip(
                cls_scores, bbox_preds, angle_preds, mlvl_points):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            scores = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels).sigmoid()

            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            angle_pred = angle_pred.permute(1, 2, 0).reshape(-1, 1)
            bbox_pred = torch.cat([bbox_pred, angle_pred], dim=1)
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = scores.max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
            bboxes = self.bbox_coder.decode(
                points, bbox_pred, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            scale_factor = mlvl_bboxes.new_tensor(scale_factor)
            mlvl_bboxes[..., :4] = mlvl_bboxes[..., :4] / scale_factor
        mlvl_scores = torch.cat(mlvl_scores)
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
        # BG cat_id: num_class
        mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)
        det_bboxes, det_labels = multiclass_nms_rotated(
            mlvl_bboxes,
            mlvl_scores,
            cfg.score_thr,
            cfg.nms,
            cfg.max_per_img)
        return det_bboxes, det_labels

    @force_fp32(
        apply_to=('cls_scores', 'bbox_preds', 'angle_preds'))
    def refine_bboxes(self, cls_scores, bbox_preds, angle_preds):
        """This function will be used in S2ANet, whose num_anchors=1."""
        num_levels = len(cls_scores)
        assert num_levels == len(bbox_preds)
        num_imgs = cls_scores[0].size(0)
        for i in range(num_levels):
            assert num_imgs == cls_scores[i].size(0) == bbox_preds[i].size(0)

        # device = cls_scores[0].device
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_points = self.prior_generator.grid_priors(featmap_sizes,
                                                       bbox_preds[0].dtype,
                                                       bbox_preds[0].device)
        bboxes_list = [[] for _ in range(num_imgs)]

        for lvl in range(num_levels):
            bbox_pred = bbox_preds[lvl]
            angle_pred = angle_preds[lvl]
            bbox_pred = bbox_pred.permute(0, 2, 3, 1)
            bbox_pred = bbox_pred.reshape(num_imgs, -1, 4)
            angle_pred = angle_pred.permute(0, 2, 3, 1)
            angle_pred = angle_pred.reshape(num_imgs, -1, 1)
            bbox_pred = torch.cat([bbox_pred, angle_pred], dim=-1)

            points = mlvl_points[lvl]

            for img_id in range(num_imgs):
                bbox_pred_i = bbox_pred[img_id]
                decode_bbox_i = self.bbox_coder.decode(points, bbox_pred_i)
                bboxes_list[img_id].append(decode_bbox_i.detach())

        return bboxes_list


@ROTATED_HEADS.register_module()
class RotatedYOLOv8Head(RotatedYOLOv6Head):
    """YOLOv8 Head"""
    def __init__(self,
                 num_classes: int,
                 in_channels: Union[int, Sequence],
                 widen_factor: float = 1.0,
                 reg_max = 0,
                 featmap_strides: Sequence[int] = (8, 16, 32),
                 regress_ranges=((-1, 64), (64, 128), (128, 256)), ##
                 bbox_coder=dict(type='DistanceAnglePointCoder', angle_version='le90'),
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox=dict(type='IoULoss', loss_weight=1.0),
                 matching : int = 0, ##
                 debug = False, ##
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 init_cfg=dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='cls_preds',
                         std=0.01,
                         bias_prob=0.01)),               
                 train_cfg=None,
                 test_cfg=None):
        self.num_classes = num_classes
        self.featmap_strides = featmap_strides
        self.num_levels = len(self.featmap_strides)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.debug = debug
        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        if self.use_sigmoid_cls:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1
        super().__init__(num_classes, in_channels, widen_factor, reg_max, featmap_strides,
                         regress_ranges=regress_ranges, matching = matching, debug=debug, ##
                         bbox_coder=bbox_coder, loss_cls=loss_cls, loss_bbox=loss_bbox, norm_cfg=norm_cfg,
                         act_cfg=act_cfg, init_cfg=init_cfg, train_cfg=train_cfg, test_cfg=test_cfg)
        
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.prior_generator = MlvlPointGenerator(featmap_strides)
        self.num_base_priors = self.prior_generator.num_base_priors[0]
        
    
    def init_weights(self, prior_prob=0.01):
        """Initialize the weight and bias of PPYOLOE head."""
        # super().init_weights()
        for reg_pred, cls_pred, ang_pred, stride in zip(self.reg_preds, self.cls_preds, self.ang_preds,
                                              self.featmap_strides):
            reg_pred[-1].bias.data[:] = 1.0  # box
            ang_pred[-1].bias.data[:] = 1.0  # angle
            # cls (.01 objects, 80 classes, 640 img)
            cls_pred[-1].bias.data[:self.num_classes] = math.log(
                5 / self.num_classes / (640 / stride)**2)

    def _init_layers(self):
        
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.ang_preds = nn.ModuleList()
        
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.featmap_strides])
        reg_out_channels = max(
            (16, self.in_channels[0] // 4, self.reg_max * 4))
        cls_out_channels = max(self.in_channels[0], self.num_classes)

        for i in range(self.num_levels):
            self.reg_preds.append(
                nn.Sequential(
                    ConvModule(in_channels=self.in_channels[i],
                               out_channels=reg_out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               norm_cfg=self.norm_cfg,
                               act_cfg=self.act_cfg),
                    ConvModule(in_channels=reg_out_channels,
                               out_channels=reg_out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               norm_cfg=self.norm_cfg,
                               act_cfg=self.act_cfg),
                    nn.Conv2d(in_channels=reg_out_channels,
                              out_channels=(self.num_base_priors + self.reg_max) * 4,
                              kernel_size=1)))
            self.cls_preds.append(
                nn.Sequential(
                    ConvModule(in_channels=self.in_channels[i],
                               out_channels=cls_out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               norm_cfg=self.norm_cfg,
                               act_cfg=self.act_cfg),
                    ConvModule(in_channels=cls_out_channels,
                               out_channels=cls_out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               norm_cfg=self.norm_cfg,
                               act_cfg=self.act_cfg),
                    nn.Conv2d(in_channels=cls_out_channels,
                              out_channels=self.num_classes,
                              kernel_size=1)))
            self.ang_preds.append(
                nn.Sequential(
                    ConvModule(in_channels=self.in_channels[i],
                               out_channels=cls_out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               norm_cfg=self.norm_cfg,
                               act_cfg=self.act_cfg),
                    ConvModule(in_channels=cls_out_channels,
                               out_channels=cls_out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               norm_cfg=self.norm_cfg,
                               act_cfg=self.act_cfg),
                    nn.Conv2d(in_channels=cls_out_channels,
                              out_channels=(self.num_base_priors + self.reg_max) * 1,
                              kernel_size=1)))

        if self.reg_max > 1:
            proj = torch.arange(
                self.reg_max + self.num_base_priors, dtype=torch.float)
            self.register_buffer('proj', proj, persistent=False)

    def forward(self, x: Tuple[Tensor]) -> Tuple[List]:
        """Forward features from the upstream network.

        Args:
            x (Tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
        Returns:
            Tuple[List]: A tuple of multi-level classification scores, bbox
            predictions.
        """
        assert len(x) == self.num_levels
        return multi_apply(self.forward_single, x, self.cls_preds, 
                           self.reg_preds, self.ang_preds, self.scales, self.featmap_strides)

    def forward_single(self, x: Tensor, cls_pred: nn.Module, 
                       reg_pred: nn.Module,  ang_pred: nn.Module, 
                       scale: List, stride) -> Tuple[Tensor, Tensor]:
        """Forward feature of a single scale level."""
        b, _, h, w = x.shape
        cls_logit = cls_pred(x)
        bbox_dist_preds = reg_pred(x)
        predicted_angle = ang_pred(x)

        if self.reg_max > 1:
            bbox_dist_preds = bbox_dist_preds.reshape(
                [-1, 4, self.reg_max + self.num_base_priors,
                 h * w]).permute(0, 3, 1, 2)

            # TODO: The get_flops script cannot handle the situation of
            #  matmul, and needs to be fixed later
            # bbox_preds = bbox_dist_preds.softmax(3).matmul(self.proj)
            bbox_preds = bbox_dist_preds.softmax(3).matmul(
                self.proj.view([-1, 1])).squeeze(-1)
            bbox_preds = bbox_preds.transpose(1, 2).reshape(b, -1, h, w)
        else:
            bbox_preds = scale(bbox_dist_preds).float()
            bbox_preds = bbox_preds.clamp(min=0)
            if not self.training:
                bbox_preds *= stride
        return cls_logit, bbox_preds, predicted_angle

@ROTATED_HEADS.register_module()
class LabelAssignment6(RotatedYOLOv8Head):
    def __init__(self,
                 num_classes: int,
                 in_channels: Union[int, Sequence],
                 widen_factor: float = 1.0,
                 reg_max = 0,
                 featmap_strides: Sequence[int] = (8, 16, 32),
                 regress_ranges=((-1, 64), (64, 128), (128, 256)), ##
                 bbox_coder=dict(type='DistanceAnglePointCoder', angle_version='le90'),
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox=dict(type='IoULoss', loss_weight=1.0),
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 init_cfg=dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='cls_preds',
                         std=0.01,
                         bias_prob=0.01)),               
                 train_cfg=None,
                 test_cfg=None):
        super().__init__(num_classes=num_classes,
                         in_channels=in_channels,
                         widen_factor=widen_factor,
                         reg_max=reg_max,
                         featmap_strides=featmap_strides,
                         regress_ranges=regress_ranges,
                         bbox_coder=bbox_coder,
                         loss_cls=loss_cls,
                         loss_bbox=loss_bbox,
                         norm_cfg=norm_cfg, 
                         act_cfg=act_cfg,
                         init_cfg=init_cfg,
                         train_cfg=train_cfg,
                         test_cfg=test_cfg)

    def _get_target_single(self, gt_bboxes, gt_labels, bbox_preds, probs, points, regress_ranges,
                           num_points_per_lvl):
        """Compute regression, classification and angle targets for a single
        image."""
        RIoU = build_loss(dict(type='RotatedIoULoss', loss_weight=1.0,
                               reduction='none', mode='linear')) ##
        num_points = points.size(0)
        num_gts = gt_labels.size(0)
        annotation_bbox = gt_bboxes
        if num_gts == 0:
            return gt_labels.new_full((num_points,), self.num_classes), \
                   gt_bboxes.new_zeros((num_points, 4)), \
                   gt_bboxes.new_zeros((num_points, 1))
        
        areas = gt_bboxes[:, 2] * gt_bboxes[:, 3] # wh
        # TODO: figure out why these two are different
        # areas = areas[None].expand(num_points, num_gts)
        areas = areas[None].repeat(num_points, 1)
        regress_ranges = regress_ranges[:, None, :].expand(
            num_points, num_gts, 2)
        points = points[:, None, :].expand(num_points, num_gts, 2)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 5)
        gt_ctr, gt_wh, gt_angle = torch.split(gt_bboxes, [2, 2, 1], dim=2)

        cos_angle, sin_angle = torch.cos(gt_angle), torch.sin(gt_angle)
        rot_matrix = torch.cat([cos_angle, sin_angle, -sin_angle, cos_angle],
                               dim=-1).reshape(num_points, num_gts, 2, 2)
        offset = points - gt_ctr
        offset = torch.matmul(rot_matrix, offset[..., None])
        offset = offset.squeeze(-1)

        w, h = gt_wh[..., 0], gt_wh[..., 1]
        offset_x, offset_y = offset[..., 0], offset[..., 1]
        left = w / 2 + offset_x
        right = w / 2 - offset_x
        top = h / 2 + offset_y
        bottom = h / 2 - offset_y
        bbox_targets = torch.stack((left, top, right, bottom), -1)

        # condition1: inside a gt bbox
        inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0
        # condition1: inside a `center bbox`
        radius = self.center_sample_radius
        stride = offset.new_zeros(offset.shape)
        bboxes = bbox_preds.clone().detach() ##
        
        # project the points on current lvl back to the `original` sizes
        lvl_begin = 0
        
        for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
            lvl_end = lvl_begin + num_points_lvl
            stride[lvl_begin:lvl_end] = self.featmap_strides[lvl_idx] * radius
            
            ''' bbox pred scale'''
            stride_bboxes = bboxes[lvl_begin:lvl_end, :4] * self.featmap_strides[lvl_idx]
            bboxes[lvl_begin:lvl_end, :4] = stride_bboxes
            
            lvl_begin = lvl_end

        inside_center_bbox_mask = (abs(offset) < stride).all(dim=-1)
        inside_gt_bbox_mask = torch.logical_and(inside_center_bbox_mask,
                                                inside_gt_bbox_mask)
        alpha = 0.5
        areas[inside_gt_bbox_mask == 0] = 0 
        overlap_idx = torch.where(torch.bincount(torch.where(areas != 0)[0]) >= 2)[0] # 위에서 2개이상의 bbox가 할당된 det 인덱스
        if len(overlap_idx) != 0:
            pts = points[overlap_idx]
            iou_matrix = torch.zeros(len(overlap_idx), len(annotation_bbox), device=areas.device) 
            det_overlapbboxes = self.bbox_coder.decode(pts[:,0], bboxes[overlap_idx])
            det_probs = probs[overlap_idx].softmax(-1)
            mask = (areas[overlap_idx] != 0).float()
            for i, gt in enumerate(annotation_bbox):
                cls = gt_labels[i]
                gt = gt.expand(len(overlap_idx), 5)
                iou =  1-(RIoU(det_overlapbboxes, gt)*(10**3)).round()/ (10**3)
                prob = det_probs[:, cls]
                iou_matrix[:,i] = alpha * iou + (1-alpha) * prob
            iou_matrix *= mask
            areas[overlap_idx] = iou_matrix
        max_area, max_area_inds = areas.max(dim=1) 
        labels = gt_labels[max_area_inds]
        labels[max_area == 0] = self.num_classes
        bbox_targets = bbox_targets[range(num_points), max_area_inds]
        angle_targets = gt_angle[range(num_points), max_area_inds]
           
            
        return labels, bbox_targets, angle_targets

