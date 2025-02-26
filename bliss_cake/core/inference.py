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
    # src_img = Image.open(img_path)
    src_base64 = encode_image(img_path)
    # img_url="https://cc-playground-testing-jess-9c632.storage.googleapis.com/65f77ab8-f862-4922-b756-efe8a51cede3/testbot-902c77ed-08f9-4193-afe4-2133f7a5a0b5.jpg"
    # resp = urllib.request.urlopen(img_url)
    # src_base64 = base64.b64encode(bytearray(resp.read())).decode('utf-8')
    #--Get base prompt
    msg = vllm.generate_base_prompt(src_base64, system_prompt, textwrap.dedent(user_prompt))
    pred_captions, pred_tokens = vllm.vllm_predict(msg, vllm_info_model)
    src_img_emb = get_img_embedding(img_path)
    # src_caption_emb = get_embedding(pred_captions)
    clip_scores = []
    captions_scores = []
    for data in ref_json:
        ref_img_emb = data['img_emb']
        ref_caption_emb = data['caption_emb']
        # ref_img_path = data['img']
        # ref_img = Image.open(ref_img_path)
        # score =  run_clip(src_img, ref_img)
        score =compare_image_embedding(src_img_emb, ref_img_emb)
        clip_scores.append(score)
        # ref_base64 = encode_image(ref_img_path)
        # msg = vllm.generate_base_prompt(ref_base64, system_prompt, textwrap.dedent(user_prompt))
        # ref_captions, pred_tokens = vllm.vllm_predict(msg, vllm_info_model)
        # caption_score = compute_caption_similarity(src_caption_emb, ref_caption_emb)
        # captions_scores.append(caption_score)
    #--Find max index
    clip_data = np.argmax(clip_scores)
    clip_meta_info = menu.bliss_cake_menu[clip_data]
    # caption_data = np.argmax(captions_scores)
    # caption_meta_info = menu.bliss_cake_menu[caption_data]

    images_with_descriptions= []
    images_with_descriptions.append(
        (
                clip_meta_info['img_path'],
                "CLIP",
            )
        )
    # images_with_descriptions.append(
    #     (
    #             caption_meta_info['img_path'],
    #             "CAPTION",
    #         )
    #     )
    logger.success(f"{clip_meta_info['captions']}")
    return  gr.update(visible=True), images_with_descriptions, pred_captions
    # return  gr.update(visible=True), images_with_descriptions, pred_captions