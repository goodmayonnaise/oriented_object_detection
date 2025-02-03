

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from .rotated_yolo_head import RotatedYOLOv6Head
from ..builder import ROTATED_HEADS, build_loss
from ..blocks import *

import torch 
import torch.nn.functional as F 
from torch import nn

from mmcv.cnn import Scale, ConvModule
from mmcv.runner import force_fp32
from mmdet.core import multi_apply, reduce_mean

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

INF = 1e8

class PixelShuffleDecode(nn.Module):
    def __init__(self, 
                 in_channels,
                 num_classes=15,
                 feat_type='p3'
                 ):
        super(PixelShuffleDecode, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.feat_type = feat_type
        self.init_layers()
        
    def init_layers(self):
        self.decode1 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels*2, 3, 1, 1),
            nn.PixelShuffle(2)
        )
        self.decode2 = nn.Sequential(
            nn.Conv2d(self.in_channels//2, self.in_channels, 3, 1, 1),
            nn.PixelShuffle(2)
        )
        
        if self.feat_type == 'p3':        
            self.decode3 = nn.Sequential(
                nn.Conv2d(self.in_channels//4, self.in_channels//2, 3, 1, 1),
                nn.PixelShuffle(2)
            )
            self.decode4 = nn.Sequential(
                nn.Conv2d(self.in_channels//8, self.in_channels//4, 3, 1, 1),
                nn.PixelShuffle(2)
            )
            self.conv1x1 = nn.Conv2d(self.in_channels//16, self.num_classes, 1, 1)
        
        elif self.feat_type == 'p4':
            self.decode3 = nn.Sequential(
                nn.Conv2d(self.in_channels//4, self.in_channels//2, 3, 1, 1),
                nn.PixelShuffle(2)
            )
            self.decode4 = nn.Sequential(
                nn.Conv2d(self.in_channels//8, self.in_channels//8, 3, 1, 1),
                nn.PixelShuffle(2)
            )
            self.conv1x1 = nn.Conv2d(self.in_channels//32, self.num_classes, 1, 1)
        
        elif self.feat_type == 'p5':
            self.decode3 = nn.Sequential(
                nn.Conv2d(self.in_channels//4, self.in_channels//4, 3, 1, 1),
                nn.PixelShuffle(2)
            )
            self.decode4 = nn.Sequential(
                nn.Conv2d(self.in_channels//16, self.in_channels//16, 3, 1, 1),
                nn.PixelShuffle(2)
            )
            self.conv1x1 = nn.Conv2d(self.in_channels//64, self.num_classes, 1, 1)
            
        
    def forward(self, x):
        global_feats, local_feats = x
        
        up1 = self.decode1(local_feats)
        up2 = self.decode2(up1)
        up3 = self.decode3(up2)
        up4 = self.decode4(up3)
        local_feats = self.conv1x1(up4)
        
        return global_feats, local_feats

class UpsampleDecode(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels,
                 num_classes=15,
                 featmap_stride=32):
        super(UpsampleDecode, self).__init__()  
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_classes = num_classes
        self.featmap_stride = featmap_stride
        
        self.init_layers()
        
    def init_layers(self):
        self.pixelshuffle = nn.PixelShuffle(2)
        self.decode1 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.in_channels, self.in_channels//2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.in_channels//2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        self.decode2 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.in_channels//2 + self.in_channels//4, self.in_channels//4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.in_channels//4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        self.decode3 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.in_channels//4, self.in_channels//8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.in_channels//8),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        self.conv1x1 = nn.Conv2d(self.in_channels//8, self.out_channels, kernel_size=1, stride=1)
        
        
    def forward(self, x):
        global_feats, local_feats = x
        pixel_up = self.pixelshuffle(local_feats)
        dup1 = self.decode1(x)
        dup1 = torch.cat([pixel_up, dup1], dim=1)
        dup2 = self.decode2(dup1)
        dup3 = self.decode3(dup2)
        local_feats = self.conv1x1(dup3)
        
        return global_feats, local_feats

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

class CustomViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        
        self.rearrange = Rearrange('b (ph pw) d -> b d ph pw', ph=image_height//patch_height, pw=image_width//patch_width)

        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        x = self.to_patch_embedding(img) # b (h/p w/p) embed dim
        b, n, _ = x.shape # h/p*w/p = n

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b) 
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)] 
        x = self.dropout(x)
        
        x = self.transformer(x)
        
        cls_tokens, patchs = x[:, 0], x[:, 1:]
        
        # x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        global_feats = self.to_latent(cls_tokens)
        global_feats = self.mlp_head(global_feats) # sigmoid는?
        
        local_feats = self.rearrange(patchs)
        
        return global_feats, local_feats
    

@ROTATED_HEADS.register_module()
class RotatedClsAttentionHead0107(RotatedYOLOv6Head):
    def __init__(self,
                 num_classes, 
                 in_channels,
                 widen_factor,
                 featmap_strides,
                 regress_ranges,
                 reg_max=0,
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
                 without_ps=False,
                 vit_config=dict(),
                 loss_cls_g=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0,
                )
                 ):
        
        self.num_levels = len(featmap_strides)
        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        
        if self.use_sigmoid_cls:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1
        
        self.without_ps = without_ps
        self.vit = vit_config
        
        super().__init__(
            num_classes=num_classes,
            in_channels=in_channels,
            widen_factor=widen_factor,
            reg_max=reg_max,
            featmap_strides=featmap_strides,
            regress_ranges=regress_ranges,
            bbox_coder=bbox_coder,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            matching=matching,
            debug=debug,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            init_cfg=init_cfg,
            train_cfg=train_cfg,
            test_cfg=test_cfg
        )
        self.loss_cls_g = build_loss(loss_cls_g)
        
    def init_weights(self):
        for reg_pred, ang_pred in zip(self.reg_preds, self.ang_preds):
            reg_pred[-1].bias.data[:] = 1.0  # box
            ang_pred[-1].bias.data[:] = 1.0  # angle

    def _init_layers(self):
        
        self.reg_preds = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
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
            
            self.cls_preds.append(
                nn.Sequential(
                    CustomViT(
                        image_size=1024//self.featmap_strides[i],
                        patch_size=self.vit.patch_size,
                        num_classes=self.num_classes,
                        dim=self.in_channels[i]*3,
                        depth=self.vit.depth,
                        heads=self.vit.num_heads,
                        channels=self.in_channels[i],
                        mlp_dim=self.vit.mlp_dim),
                    PixelShuffleDecode(in_channels=self.in_channels[i]*3,
                                       num_classes=self.num_classes,
                                       feat_type=f"p{i+3}")
                )
            )
            
            if self.reg_max > 1:
                proj = torch.arange(
                    self.reg_max + self.num_base_priors, dtype=torch.float)
                self.register_buffer('proj', proj, persistent=False)
        self.cls_scores_g = nn.Linear(self.num_classes*3, self.num_classes)

    def forward(self, x):
        """Forward features from the upstream network.

        Args:
            x (Tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
        Returns:
            Tuple[List]: A tuple of multi-level classification scores, bbox
            predictions.
        """
        assert len(x) == self.num_levels
        preds = multi_apply(self.forward_single, x, self.cls_preds, 
                           self.reg_preds, self.ang_preds, self.scales, self.featmap_strides)
        cls_scores, bbox_preds, angle_preds = preds
        
        cls_scores_g, cls_scores_l = [], []
        for i in cls_scores:
            cls_scores_g.append(i[0])
            cls_scores_l.append(i[1])
        
        cls_scores_g = torch.cat(cls_scores_g, 1)
        cls_scores_g = self.cls_scores_g(cls_scores_g)

        return cls_scores_g, cls_scores_l, bbox_preds, angle_preds

    def forward_single(self, x, cls_pred, reg_pred,  ang_pred, scale, stride):
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
    @force_fp32(
        apply_to=('cls_scores', 'bbox_preds', 'angle_preds'))
    def loss(self,
             cls_scores_g,
             cls_scores_l,
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
        assert len(cls_scores_l) == len(bbox_preds) == len(angle_preds)
        
        # cls_scores_g, cls_scores_l = [], []
        # for i in range(len(cls_scores)):
        #     cls_scores_g.append(cls_scores[i][0])
        #     cls_scores_l.append(cls_scores[i][1])
        
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores_l]
        all_level_points = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=bbox_preds[0].dtype,
            device=bbox_preds[0].device)
        
        labels_l, labels_g, bbox_targets, angle_targets = self.get_targets(
            all_level_points, gt_bboxes, gt_labels, 
            bbox_preds, angle_preds, cls_scores_l)
        
        num_imgs = cls_scores_l[0].size(0)
        
        flatten_cls_scores_l = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, cls_score.shape[1])
            for cls_score in cls_scores_l
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_angle_preds = [
            angle_pred.permute(0, 2, 3, 1).reshape(-1, 1)
            for angle_pred in angle_preds
        ]
        
        flatten_cls_scores_l = torch.cat(flatten_cls_scores_l)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_angle_preds = torch.cat(flatten_angle_preds)
        flatten_labels_l = torch.cat(labels_l)
        flatten_bbox_targets = torch.cat(bbox_targets)
        flatten_angle_targets = torch.cat(angle_targets)
        
        
        # repeat points to align with bbox_preds
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points])

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((flatten_labels_l >= 0)
                    & (flatten_labels_l < bg_class_ind)).nonzero().reshape(-1)
        num_pos = torch.tensor(
            len(pos_inds), dtype=torch.float, device=bbox_preds[0].device)
        num_pos = max(reduce_mean(num_pos), 1.0)

        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_angle_preds = flatten_angle_preds[pos_inds]

        pos_bbox_targets = flatten_bbox_targets[pos_inds]
        pos_angle_targets = flatten_angle_targets[pos_inds]

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
                cls_iou_targets = torch.zeros_like(flatten_cls_scores_l)
                bbox_overlap = build_loss(dict(type='RotatedIoULoss', loss_weight=1.0, reduction='none', mode='linear')) ##
                iou_targets = 1 - bbox_overlap(
                    pos_decoded_bbox_preds,
                    pos_decoded_target_preds.detach()).clamp(min=1e-6)
                
                cls_iou_targets[pos_inds, flatten_labels_l[pos_inds]] = iou_targets
                loss_cls = self.loss_cls(
                    flatten_cls_scores_l,
                    cls_iou_targets,
                    avg_factor=num_pos)
            else:     
                if 'MultiLoss' in self.loss_cls.__str__():
                    losses = self.loss_cls(flatten_cls_scores_l, flatten_labels_l, avg_factor=num_pos)
                    
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
                    loss_cls = self.loss_cls(flatten_cls_scores_l, flatten_labels_l, avg_factor=num_pos)

        else:
            if self.loss_cls.__str__() == 'VarifocalLoss()':
                flatten_labels = torch.zeros_like(flatten_cls_scores_l)
            loss_cls = self.loss_cls(
                flatten_cls_scores_l, flatten_labels_l, avg_factor=num_pos)
            loss_bbox = pos_bbox_preds.sum()

        loss_cls_g = self.loss_cls_g(torch.cat([i for i in cls_scores_g]), torch.cat(labels_g))
        
        return dict(
                loss_cls=loss_cls,
                loss_cls_g = loss_cls_g,
                loss_bbox=loss_bbox)
        

    @force_fp32(
        apply_to=('cls_scores', 'bbox_preds', 'angle_preds'))
    def get_bboxes(self,
                   cls_scores_g,
                   cls_scores_l,
                   bbox_preds,
                   angle_preds,
                   img_metas,
                   cfg=None,
                   rescale=None):
        assert len(cls_scores_l) == len(bbox_preds)
        
        num_levels = len(cls_scores_l)

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores_l]

        mlvl_points = self.prior_generator.grid_priors(featmap_sizes,
                                                       bbox_preds[0].dtype,
                                                       bbox_preds[0].device)
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores_l[i][img_id].detach() for i in range(num_levels)
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
        
        # global labels per batch -> p3 p4 p5 
        labels_g = [F.one_hot(i.unique()[:-1], 15).sum(0) for i in labels_list]
        
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

        # labels_g = [F.one_hot(i[2].unique())[:,:-1].sum(0) for i in labels_list]
        
        # concat per level image
        concat_lvl_labels = []
        concat_lvl_labels_g = []
        concat_lvl_bbox_targets = []
        concat_lvl_angle_targets = []
        for i in range(num_levels):
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list]))
            concat_lvl_labels_g.append(
                torch.cat([F.one_hot(labels[i].unique()[:-1], self.num_classes).sum(0) for labels in labels_list])
            )
            bbox_targets = torch.cat(
                [bbox_targets[i] for bbox_targets in bbox_targets_list])
            angle_targets = torch.cat(
                [angle_targets[i] for angle_targets in angle_targets_list])
            bbox_targets = bbox_targets / self.featmap_strides[i]
            concat_lvl_bbox_targets.append(bbox_targets)
            concat_lvl_angle_targets.append(angle_targets)
        return (concat_lvl_labels, labels_g, concat_lvl_bbox_targets,
                concat_lvl_angle_targets)
