#--Check with pre-computed features
import os
import json
import numpy as np
import torch
from loguru import logger
from PIL import Image
from utils import util
src_dirs = "data/currencies/sources/images/"
src_json_path = 'data/currencies/sources/srccurrencies_11b.json'
ref_json_path = 'data/currencies/references/refcurrencies_11b.json'
ref_dirs = 'data/currencies/references/images/' 
savepath = 'results/currencies/captions/'
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
# list_data = []
for src_data in src_json:
    src_img_path = os.path.join(src_dirs, src_data['img_name'])
    #--Compare between analyze result
    savepath_analyze = os.path.join(savepath, "text_emb_analyze")
    if not os.path.exists(savepath_analyze):os.makedirs(savepath_analyze)
    scores_analyze, ref_data = util.compare_per_field(src_data, ref_json, 'text_emb_analyze')
    util.save_similar_image(src_data, ref_data, src_dirs, ref_dirs, savepath_analyze)

    savepath_extract = os.path.join(savepath, "text_emb_extract")
    if not os.path.exists(savepath_extract):os.makedirs(savepath_extract)
    scores_extract, ref_data = util.compare_per_field(src_data, ref_json, 'text_emb_extract')
    util.save_similar_image(src_data, ref_data, src_dirs, ref_dirs, savepath_extract)

    #--Get average 
    scores_analyze = np.array(scores_analyze)
    scores_extract = np.array(scores_extract)
    avg_scores = ((scores_analyze + scores_extract)/2)
    avg_scores = avg_scores.tolist()
    max_idx = np.argmax(avg_scores)
    ref_data = ref_json[max_idx]['img_name']
    savepath_avg = os.path.join(savepath, "text_emb_avg")
    if not os.path.exists(savepath_avg):os.makedirs(savepath_avg)
    util.save_similar_image(src_data, ref_data, src_dirs, ref_dirs, savepath_avg)


