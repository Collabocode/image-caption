import os
import json
import textwrap
import numpy as np
from PIL import Image
import gradio as gr
from loguru import logger
import base64
import urllib

from constants import menu
from models.clip import run_clip, compare_image_embedding, get_img_embedding
from models import vllm
from models.embedding import compute_caption_similarity, get_embedding
from utils.img_util import encode_image
#--Get all available bliss cake data
ref_json_path = "/mnt/hdd1/jano/VisLang/bliss_cake/bliss_cake_ref.json"
f = open(ref_json_path,)
ref_json = json.load(f)
f.close()
list_data = []
def run_model(img_path, system_prompt, user_prompt, vllm_info_model):
   
    src_base64 = encode_image(img_path)
    
    msg = vllm.generate_base_prompt(src_base64, system_prompt, textwrap.dedent(user_prompt))
    pred_captions, pred_tokens = vllm.vllm_predict(msg, vllm_info_model)
    src_img_emb = get_img_embedding(img_path)
    clip_scores = []
    captions_scores = []
    for data in ref_json:
        ref_img_emb = data['img_emb']
        ref_caption_emb = data['caption_emb']
        score =compare_image_embedding(src_img_emb, ref_img_emb)
        clip_scores.append(score)
    
    clip_data = np.argmax(clip_scores)
    clip_meta_info = menu.bliss_cake_menu[clip_data]
    
    images_with_descriptions= []
    
    check_max_value = max(clip_scores)
    if check_max_value < 0.85:
        logger.success(f"No image in server similar to the input image")
        return gr.update(visible=False), images_with_descriptions, pred_captions
    else:
        images_with_descriptions.append(
            (
                    clip_meta_info['img_path'],
                    "CLIP",
                )
            )
        
        logger.success(f"{clip_meta_info['captions']}")
        return  gr.update(visible=True), images_with_descriptions, pred_captions