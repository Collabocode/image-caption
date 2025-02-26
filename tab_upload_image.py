import gradio as gr
import json
import shutil
from loguru import logger
import os
from img_util import encode_image
from langfuse_deps import classify_image_lfprompt
import vllm

base_folder = 'datasets_data'
vllm_info_model = "accounts/fireworks/models/llama-v3p2-11b-vision-instruct"
def get_initial_data_list(path='datasets_data'):
    list_data = []
    for data in sorted(os.listdir(path)):
        list_data.append(data)
    return list_data
def create_folder_from_bot(out_vllm):
    try:
        new_class = json.loads(out_vllm)
        new_folder = new_class['class'] 
    except:
        new_folder = out_vllm
    #--Create save path 
    save_path = os.path.join(base_folder, new_folder)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        logger.success(f"Success create new folder: {save_path}")
    else:
        logger.info(f"New folder already exist: {save_path}")
    return save_path
def copy_data_to_folder(src_path, dst_path):
    img_name =src_path.split('/')[-1]
    save_path = os.path.join(dst_path, img_name)
    if os.path.exists(save_path):
        logger.warning(f"{img_name} in {dst_path} already exists")
    else:
        shutil.copy(src_path, save_path)
        logger.success(f"Success save {img_name} in {dst_path}")
     
def run_upload(img_path, sys_prompt_caption, user_prompt_caption):
    src_base64 = encode_image(img_path)
    msg = vllm.generate_caption_prompt(src_base64, sys_prompt_caption, user_prompt_caption)
    pred_captions, vllm_pred_tokens = vllm.vllm_predict(msg, vllm_info_model=vllm_info_model, is_json=True)
   
    logger.success(pred_captions)
    save_path = create_folder_from_bot(pred_captions)
    copy_data_to_folder(img_path, save_path)
    return pred_captions, gr.update(choices=get_initial_data_list(),)
def generate_upload_tab():
    with gr.Row():
        with gr.Column(scale=1):
            img_path = gr.Image(label="Upload your image", type="filepath", height=400, width=400)
            submit_btn = gr.Button(value="Upload", scale=1, variant="primary")
        with gr.Column(scale=3):
            with gr.Row():
                with gr.Column():
                    sys_prompt_caption = gr.Textbox(label="System prompt caption", value=f"{classify_image_lfprompt[0]['content']}", lines=1, interactive=False)
                with gr.Column():
                    user_prompt_caption = gr.Textbox(label="User prompt caption", value=f"{classify_image_lfprompt[1]['content']}", lines=1, interactive=False)
            with gr.Row():
                data_list = gr.Radio(label="List data", scale=5, choices=get_initial_data_list(), interactive=False)
            with gr.Row():
                text = gr.Textbox(label="Output log", value="", lines=5, interactive=False)
        gr.on(
        triggers=[submit_btn.click],
        fn=run_upload, 
        inputs=[img_path, sys_prompt_caption, user_prompt_caption],
        outputs=[text, data_list])
        