from torch import nn
from torchinfo import summary

import mmcv
from mmrotate.models import build_detector, build_backbone,\
    build_neck, build_head

cfg_path = '/mmrotate/configs/jiyeon/vit-psd-head.py'

cfg = mmcv.Config.fromfile(cfg_path)


detector = build_detector(cfg.model, cfg.get('train_cfg'), cfg.get('test_cfg'))
print(f"\ndetector params lenght : {len([i for i in detector.parameters()])}")
d_params = sum(i.numel() for i in detector.parameters())
d_type_size = detector.parameters().__next__().element_size()
print(f"detector total params : {d_params*d_type_size}\n")

backbone = build_backbone(cfg.model.backbone)
print(f"backbone params lenght : {len([i for i in backbone.parameters()])}")
b_params = sum(i.numel() for i in backbone.parameters())
b_type_size = backbone.parameters().__next__().element_size()
print(f"backbone total params : {b_params*b_type_size}\n")

if cfg.model.neck is not None:
    neck = build_neck(cfg.model.neck)
    print(f"neck params lenght : {len([i for i in neck.parameters()])}")
    n_params = sum(i.numel() for i in neck.parameters())
    n_type_size = neck.parameters().__next__().element_size()
    print(f"neck total params : {n_params*n_type_size}\n")

head = build_head(cfg.model.bbox_head)
print(f"head params lenght : {len([i for i in head.parameters()])}")
h_params = sum(i.numel() for i in head.parameters())
h_type_size = head.parameters().__next__().element_size()
print(f"head total params : {h_params*h_type_size}\n")

if cfg.model.neck is not None:
    summary(nn.Sequential(backbone, neck, head), (1, 3, 1024, 1024))

else:
    summary(nn.Sequential(backbone, head), (1, 3, 1024, 1024))

print()
    
