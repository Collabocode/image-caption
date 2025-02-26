import gradio as gr
import json
import shutil
from loguru import logger
import os
from utils.img_util import encode_image
from utils.langfuse_deps import classify_image_lfprompt
from models import vllm

base_folder = '/mnt/hdd1/jano/VisLang/datasets_data'
results_folder = '/mnt/hdd1/jano/VisLang/results/cap_1_sum_1'
def get_initial_data_list(path='/mnt/hdd1/jano/VisLang/datasets_data'):
    list_data = []
    for data in sorted(os.listdir(path)):
        list_data.append(data)
    return list_data
def run_show_data(folder_name):
    #--Get all captioned list data
    check_data = sorted(os.listdir(results_folder))
    selected_data = None
    #--Check whether the folder has been captioned before
    for data in check_data:
        if folder_name in data:
            selected_data = data
            break
    new_list = []
    new_list.append(["IMAGE PATH", "CAPTION", "SUMMARY"])
    if selected_data is not None:
        with open(os.path.join(results_folder, selected_data)) as f:
            d = json.load(f)
        for data in d:
            new_list.append([data['img_path'], data['caption'], data['summary'] ])
        return gr.Dataset(samples=new_list)
    else:
        new_list.append(["NOT FOUND", "NOT FOUND", "NOT FOUND"])
        return gr.Dataset(samples=new_list)
def generate_show_data_tab():
    with gr.Row():
        with gr.Column():
            data_list = gr.Radio(label="List data", scale=5, choices=get_initial_data_list(), interactive=True)
        with gr.Column():
            submit_btn = gr.Button(value="Show data", scale=1, variant="primary")
    with gr.Row():
        examples = gr.Examples(
            examples=[
                ["icon.png", "caption", "summary"],
            ],
            inputs=[gr.Image(visible=False), gr.Textbox(visible=False), gr.Textbox(visible=False)],
        )
    gr.on(
        triggers=[submit_btn.click],
        fn=run_show_data,
        inputs=[data_list],
        outputs=[examples.dataset])
    
