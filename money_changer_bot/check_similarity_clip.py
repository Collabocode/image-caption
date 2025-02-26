import torch
import numpy as np
import clip
from PIL import Image
from torch import nn
from loguru import logger
from utils import util
import os
#--ARGS
ref_dirs = 'data/menu/references/images/'
src_dirs = 'data/menu/sources/images'
savepath = 'results_menu_v2/menu/images/'
if not os.path.exists(savepath):
    os.makedirs(savepath)
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/16", device=device)

ref_imgs = []
for ref_img_path in sorted(os.listdir(ref_dirs)):
    ref_imgs.append(os.path.join(ref_dirs, ref_img_path))

src_imgs = []
for src_img_path in sorted(os.listdir(src_dirs)):
    src_imgs.append(os.path.join(src_dirs, src_img_path))

count=0
with torch.no_grad():
    list_data = []
    for src_path in src_imgs:
        scores = []
        src_img = preprocess(Image.open(src_path)).unsqueeze(0).to(device)
        img_name = src_path.split('/')[4]
        src_feat = model.encode_image(src_img)
        
        for ref_path in ref_imgs:
            logger.info(ref_path)
            ref_img = preprocess(Image.open(ref_path)).unsqueeze(0).to(device)
            #--Check similarity
            ref_feat = model.encode_image(ref_img)
            score = util.compare_cosine_image(src_feat, ref_feat)
            scores.append(score.detach().cpu().numpy())
        max_idx = np.argmax(scores)
        ref_data = ref_imgs[max_idx]
        list_data.append({'src_data':src_path, 'ref_data':ref_data})
        
        #--
        ref_img_save = Image.open(ref_data)
        src_img_save = Image.open(src_path).resize((ref_img_save.width, ref_img_save.height))
        total_width = src_img_save.width + ref_img_save.width
        max_height = src_img_save.height #+ ref_img_save.height
        new_im = Image.new('RGB', (total_width, max_height))
        x_offset = 0
        for im in [src_img_save, ref_img_save]:
            new_im.paste(im, (x_offset,0))
            x_offset += im.size[0]
        new_im.save(f'{os.path.join(savepath, img_name)}')
        # count+=1
logger.success(list_data)