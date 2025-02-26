import os
import json
import textwrap
import numpy as np
from PIL import Image
import gradio as gr
from loguru import logger

import vllm
import llm
from img_util import encode_image
#--Get all available bliss cake data
def run_model(
        img_path, sys_prompt_caption, user_prompt_caption, 
        sys_prompt_summary, user_prompt_summary,
        sys_prompt_classify, user_prompt_classify,
        vllm_info_model, llm_info_model):
    
    src_base64 = encode_image(img_path)
    msg = vllm.generate_caption_prompt(src_base64, sys_prompt_caption, user_prompt_caption)
    pred_captions, vllm_pred_tokens = vllm.vllm_predict(msg, vllm_info_model)

    #--Run for summary
    msg = llm.generate_summary_prompt(sys_prompt_summary, user_prompt_summary, pred_captions)
    pred_summary, llm_pred_tokens = llm.llm_predict(msg, llm_info_model,is_json=True)
    
    #--Run text description
    msg = llm.generate_summary_prompt(sys_prompt_classify, user_prompt_classify, pred_captions)
    pred_classified, llm_pred_tokens = llm.llm_predict(msg, llm_info_model, is_json=False)
    logger.success(pred_classified)
    return pred_captions, pred_summary, pred_classified, vllm_pred_tokens, llm_pred_tokens

