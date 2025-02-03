import cv2, os
import torch
import numpy as np

from mmrotate.models import build_backbone, build_neck, build_head
from mmcv import Config 

import torch.nn.functional as F

def apply_heatmap_to_image(feature_map, original_image, alpha=0.6):
    feature_map_avg = feature_map.mean(dim=1).squeeze().cpu().detach().numpy()  # (1, H, W) -> (H, W)
        
    heatmap = cv2.normalize(feature_map_avg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    original_image = cv2.normalize(original_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
    combined_img = cv2.addWeighted(heatmap, alpha, original_image, 1 - alpha, 0)

    return combined_img

def save_feat_img(imgs, org_img, save_path, type, compose=False): # compose : split save img
    combined_images = []
    for i, feature_map in enumerate(imgs):
        combined_image = apply_heatmap_to_image(feature_map, org_img)
        combined_images.append(combined_image)
    
    if compose:
        for i in range(len(imgs)):
            save_path2 = os.path.join(save_path, type + f'{i}.png')
            cv2.imwrite(save_path2, combined_images[i])
            print(f"\n{save_path2} done")
    else: 
        h, w, _ = combined_images[0].shape
        _image = np.zeros((h, len(combined_images) * w, 3), dtype=np.uint8)
        for i in range(len(combined_images)): 
            _image[:, w*i:w*(i+1),:] = combined_images[i] 
            save_path2 = os.path.join(save_path,type)+ '.png'
            cv2.imwrite(save_path2, _image)
            print(save_path)
    print(f"\n{type} done")

def data_load(data_path):
    img = cv2.imread(data_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (1024, 1024))
    img = img_resized.astype('float32') / 255.0
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0) 
    return img_tensor, img

def hook_fn(module, input, output):
    feature_maps.append(output) 

def weight_keys(part):
    return {key.replace(part+'.', ''): value for key, value in weights['state_dict'].items() if key.startswith(part)}

if __name__ == '__main__':

    # jh 
    sample_data_path = '/mmrotate/data/dummy_vis/area/GTF4_P2011__1024__0___0.png'
    # jg 
    # sample_data_path = '/mmrotate/data/split_ms_dota2_2/val/images/P1512__1365__0___376.png'
    # large plane
    # sample_data_path = '/mmrotate/data/split_ms_dota2_2/val/images/P1023__1365__2097___0.png'

    # ope ps 15 attn block 1 upsample 
    org_pretrained = '/mmrotate/work_dirs/ope-ps15-attnblock-upsample/best_mAP_epoch_50.pth'
    config_path = '/mmrotate/work_dirs/ope-ps15-attnblock-upsample/ope-attn-upsample.py'
    
    
    weights = torch.load(org_pretrained)

    save_path = os.path.join('/',*org_pretrained.split('/')[:-1], 'save', 'vis_featmap', org_pretrained.split('/')[-1].split('.')[0], sample_data_path.split('/')[-1].split('.')[0])
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    cfg = Config.fromfile(config_path)
        
    img_tensor, img = data_load(sample_data_path)
    
    # backbone setting
    backbone_weights = weight_keys('backbone')
    backbone = build_backbone(cfg.model.backbone)
    backbone.eval()

    # neck setting
    if cfg.model.neck is not None:
        neck_weights = weight_keys('neck')
        neck = build_neck(cfg.model.neck)
        neck.eval()
    
    # head_setting
    head_weights = weight_keys('bbox_head')
    head = build_head(cfg.model.bbox_head)
    head.eval()
    
    ## backbone
    backbone.load_state_dict(backbone_weights) 

    feature_maps = []
    
    backbone.stem.register_forward_hook(hook_fn)
    backbone.stage1.register_forward_hook(hook_fn)
    backbone.stage2.register_forward_hook(hook_fn)
    backbone.stage3.register_forward_hook(hook_fn)
    backbone.stage4.register_forward_hook(hook_fn)
        
    # feat map list 
    out_org = backbone(img_tensor) 
    # out = list(out_org) 

    save_feat_img(feature_maps, img, save_path, type='backbone_layers', compose=True)
