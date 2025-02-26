import os
from torch import nn
from PIL import Image
from loguru import logger
import torch
import numpy as np
import clip

loss = nn.CosineSimilarity(dim=1, eps=1e-6)
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/16", device=device)

def list_to_tensor(src_emb, to_tensor=True):
    emb_np = np.array(src_emb)
    emb_np = emb_np[np.newaxis, :]#-- Adding batch in front
    if to_tensor:
       emb = torch.from_numpy(emb_np)
    else:
        emb = emb_np 
    return emb
def compare_cosine_image(feat1, feat2):
    return loss(feat1, feat2)

def save_similar_image(src_data, ref_data, src_dirs, ref_dirs, savepath):
    src_name = src_data['img_name']
    ref_img_save = Image.open(os.path.join(ref_dirs, ref_data))
    src_img_save = Image.open(os.path.join(src_dirs, src_data['img_name'])).resize((ref_img_save.width, ref_img_save.height))
    total_width = src_img_save.width + ref_img_save.width
    max_height = src_img_save.height #+ ref_img_save.height
    new_im = Image.new('RGB', (total_width, max_height))
    x_offset = 0
    for im in [src_img_save, ref_img_save]:
        new_im.paste(im, (x_offset,0))
        x_offset += im.size[0]
    new_im.save(f'{os.path.join(savepath, src_name)}')

def compare_per_field(src_data, ref_json, field_name):
    scores = []
    src_caption_emb = src_data[field_name]
    src_emb = list_to_tensor(src_caption_emb)
    for ref_data in ref_json:
        ref_caption_emb = ref_data[field_name]
        ref_emb = list_to_tensor(ref_caption_emb)
        score = compare_cosine_image(src_emb, ref_emb)
        scores.append(score.detach().cpu().numpy())
        logger.info(score)
    max_idx = np.argmax(scores)
    ref_data = ref_json[max_idx]['img_name']
    return scores, ref_data

def compare_per_field_img(src_data, ref_json, field_name, src_dirs=None, ref_dirs=None):
    scores = []
    src_path = os.path.join(src_dirs, src_data[field_name])
    src_img = preprocess(Image.open(src_path)).unsqueeze(0).to(device)
    src_feat = model.encode_image(src_img)
    for ref_data in ref_json:
        ref_path = os.path.join(ref_dirs, ref_data[field_name])
        ref_img = preprocess(Image.open(ref_path)).unsqueeze(0).to(device)
        ref_feat = model.encode_image(ref_img)
        score = compare_cosine_image(src_feat, ref_feat)
        scores.append(score.detach().cpu().numpy())
        logger.info(score)
    max_idx = np.argmax(scores)
    ref_data = ref_json[max_idx]['img_name']
    return scores, ref_data