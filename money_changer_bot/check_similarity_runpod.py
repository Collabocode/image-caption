#--Check with pre-computed features
import os
import json
import numpy as np
import torch
from loguru import logger
from PIL import Image
from utils import util
# src_dirs = "data/sources/won_1000/"
src_dirs = "data/currencies/sources/images/"
src_json_path = 'data/currencies/sources/srccurrencies_11b_v5.json'
ref_json_path = 'data/currencies/references/refcurrencies_11b_v5.json'
savepath = 'results/currencies_v4/captions_avg/'
ref_dirs = 'data/currencies/references/images/' 
if not os.path.exists(savepath):
    os.makedirs(savepath)

def list_to_tensor(src_emb, to_tensor=True):
    emb_np = np.array(src_emb)
    emb_np = emb_np[np.newaxis, :]#-- Adding batch in front
    if to_tensor:
       emb = torch.from_numpy(emb_np)
    else:
        emb = emb_np 
    return emb

# src_json_path = 'data/sources/currency_sources_1000_won.json'

f = open(src_json_path,)
src_json = json.load(f)
f.close()

f = open(ref_json_path,)
ref_json = json.load(f)
f.close()
list_data = []
# src_img_path = os.path.join(src_dirs, src_data['img_name'])
#     #--Compare between analyze result
#     savepath_analyze = os.path.join(savepath, "text_emb_analyze")
#     if not os.path.exists(savepath_analyze):os.makedirs(savepath_analyze)
#     scores_analyze, ref_data = util.compare_per_field(src_data, ref_json, 'text_emb_analyze')
#     util.save_similar_image(src_data, ref_data, src_dirs, ref_dirs, savepath_analyze)
for src_data in src_json:
    src_img_path = os.path.join(src_dirs, src_data['img_name'])
    scores_text_emb, ref_data = util.compare_per_field(src_data, ref_json, 'text_emb')
    scores_img_emb, ref_data = util.compare_per_field_img(src_data, ref_json, 'img_name', src_dirs, ref_dirs)
    # list_data.append({'src_data':src_img_path, 'ref_data':ref_data})

    #--Save results
    src_name = src_data['img_name']
    scores_text_emb = np.array(scores_text_emb)
    scores_img_emb = np.array(scores_img_emb)
    # avg_scores = ((scores_text_emb + scores_img_emb)/2)
    avg_scores = (((1. *scores_text_emb) + (0.0 * scores_img_emb)))
    avg_scores = avg_scores.tolist()
    max_idx = np.argmax(avg_scores)
    ref_data = ref_json[max_idx]['img_name']

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
