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
            cv2.imwrite(os.path.join(save_path,type+f'_test_{i}.png'), combined_images[i])
    else: 
        h, w, _ = combined_images[0].shape
        _image = np.zeros((h, len(combined_images) * w, 3), dtype=np.uint8)
    for i in range(len(combined_images)): 
        _image[:, w*i:w*(i+1),:] = combined_images[i] 
        cv2.imwrite(os.path.join(save_path,type)+ '.png', _image)
    
    print(f"\n{type} done")
    print(os.path.join(save_path,type)+ '.png')

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

def vis_attention_map(input_img_path, attns, save_path, ms):
    save_path = os.path.join(save_path, f"p{3+ms}_attn.png")
    img = cv2.imread(input_img_path)
    h, w, c = img.shape
    attn_maps = []
    for i, attn in enumerate(attns):
        if ms == 0:
            resize_ratio = 8
        else:
            resize_ratio = 8 // (2*ms) # transformer
        attn_avg = attn.mean(dim=1)[0]
        query_patch_idx = attn_avg.shape[1] // 2
        attention_map = attn_avg[:, query_patch_idx]
        
        attention_2d = attention_map.reshape(resize_ratio, resize_ratio)
        attention_resize = F.interpolate(attention_2d.unsqueeze(0).unsqueeze(0),
                                        size = img.shape[:2], 
                                        mode='bilinear',
                                        align_corners=False).squeeze()
        
        # norm 
        attention_resize = (attention_resize - attention_resize.min()) / (attention_resize.max() - attention_resize.min())
        attention_resize = (attention_resize * 255).detach().cpu().numpy().astype(np.uint8)
        attention_colormap = cv2.applyColorMap(attention_resize, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img, 0.6, attention_colormap, 0.4, 0)
        attn_maps.append(overlay)
        
    combine_img = np.zeros((h, w*(len(attns)), 3), dtype=np.uint8)
    for i, attn in enumerate(attn_maps):
        combine_img[:, w*i:w*(i+1),:] = attn
    cv2.imwrite(save_path, combine_img)
    
    print('\nattention map done')
    print(save_path)
        
    # attn_avg = attn.mean(dim=1)[0]
    # query_patch_idx = attn_avg.shape[1] // 2
    # attention_map = attn_avg[:, query_patch_idx]
    # # h_p, w_p = 1024 // attn.shape[-1], 1024 // attn.shape[-1]
    
    # attention_2d = attention_map.reshape(resize_ratio, resize_ratio)
    # attention_resize = F.interpolate(attention_2d.unsqueeze(0).unsqueeze(0),
    #                                  size = img.shape[:2], 
    #                                  mode='bilinear',
    #                                  align_corners=False).squeeze()
    
    # # norm 
    # attention_resize = (attention_resize - attention_resize.min()) / (attention_resize.max() - attention_resize.min())
    # attention_resize = (attention_resize * 255).detach().cpu().numpy().astype(np.uint8)
    # attention_colormap = cv2.applyColorMap(attention_resize, cv2.COLORMAP_JET)
    
    # overlay = cv2.addWeighted(img, 0.6, attention_colormap, 0.4, 0)
    
    # cv2.imwrite(save_path, overlay)

if __name__ == '__main__':
    # input
    # jh 
    sample_data_path = '/mmrotate/data/dummy_vis/area/GTF4_P2011__1024__0___0.png'
    # jg 
    # sample_data_path = '/mmrotate/data/split_ms_dota2_2/val/images/P1512__1365__0___376.png'
    # large plane
    # sample_data_path = '/mmrotate/data/split_ms_dota2_2/val/images/P1023__1365__2097___0.png'

    # org_pretrained = '/mmrotate/work_dirs/prototype3/prototype3-9/epoch_48.pth'
    # config_path = '/mmrotate/work_dirs/prototype3/prototype3-9/prototype3-9.py'
    
    org_pretrained = '/mmrotate/work_dirs/vit-clshead/best_mAP_epoch_46.pth'
    config_path = '/mmrotate/work_dirs/vit-clshead/attn-upsample0107.py'
    
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
    
    # feat map list 
    backbone_out = backbone(img_tensor) 
    
    # neck
    if cfg.model.neck is not None:
        neck.load_state_dict(neck_weights)
        neck_out = neck(backbone_out)
        
    
    ## head ----------------------------------------------------------------

    head.load_state_dict(head_weights)
    feature_maps = []

    ms = 2

    # attention scores
    # [head.cls_preds[ms][0].transformer.layers[i][0].attend.register_forward_hook(hook_fn) for i in range(6)]
    # # # vit out
    # [head.cls_preds[i][0].register_forward_hook(hook_fn) for i in range(3)]    
    # pixel decoder out 
    # head.cls_preds[ms][1].decode1.register_forward_hook(hook_fn)
    # head.cls_preds[ms][1].decode2.register_forward_hook(hook_fn)
    # head.cls_preds[ms][1].decode3.register_forward_hook(hook_fn)
    # head.cls_preds[ms][1].decode4.register_forward_hook(hook_fn)
    # head.cls_preds[ms][1].conv1x1.register_forward_hook(hook_fn)
    
    # conv in decoder
    head.cls_preds[ms][1].decode1[0].register_forward_hook(hook_fn)
    head.cls_preds[ms][1].decode1[1].register_forward_hook(hook_fn)
    head.cls_preds[ms][1].decode2[0].register_forward_hook(hook_fn)
    head.cls_preds[ms][1].decode2[1].register_forward_hook(hook_fn)
    head.cls_preds[ms][1].decode3[0].register_forward_hook(hook_fn)
    head.cls_preds[ms][1].decode3[1].register_forward_hook(hook_fn)
    head.cls_preds[ms][1].decode4[0].register_forward_hook(hook_fn)
    head.cls_preds[ms][1].decode4[1].register_forward_hook(hook_fn)
    head.cls_preds[ms][1].conv1x1.register_forward_hook(hook_fn)
    
    

    if cfg.model.neck is not None:
        head_out = head(neck_out)
    else:
        head_out = head(backbone_out)

    if len(head_out)==4:
        head_out = head_out[1:]
    
    head_cls = ['head_cls', 'head_bbox', 'head_angle']

    # for attention score    
    # feature_maps = [i[:,:,1:,1:] for i in feature_maps]
    # vis_attention_map(sample_data_path, feature_maps, save_path=save_path, ms=ms)
    
    # for vit 
    # feature_maps = [i[1] for i in feature_maps]   
    # save_feat_img(feature_maps, img, save_path, type=f"vit_out")
    
    # for pixel shuffle decoder 
    save_feat_img(feature_maps, img, save_path, type=f"p{ms+3}_decoder_detail")

