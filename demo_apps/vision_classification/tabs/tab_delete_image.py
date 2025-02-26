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
def run_check_data(folder_name):
    #--Get all images from server
    imgs_dir = os.path.join(base_folder, folder_name)
    check_data = sorted(os.listdir(imgs_dir))
    new_list_images = []
    for data in check_data:
        new_list_images.append(os.path.join(imgs_dir, data))
    return new_list_images
def remove_images(data_list, selected_images, selected_data):
    #--Delete data in gallery 
    path = selected_images[selected_data.value][0]
    img_name =path.split('/')[-1]
    #--Delete from gradio as well as from server 
    imgs_path = os.path.join(base_folder, data_list)
    save_path = os.path.join(imgs_path, img_name)
    shutil.move(save_path, f'./{img_name}')
    # shutil.move(path, f'./tmp_{img_name}')
    #--New list
    new_image_list = []
    for img in sorted(os.listdir(imgs_path)):
        new_image_list.append(os.path.join(imgs_path, img))
    return new_image_list
def update_gallery(selected_images:gr.SelectData):
    return gr.State(selected_images.index)
def generate_delete_data_tab():
    with gr.Row():
        with gr.Column():
            data_list = gr.Radio(label="List data", scale=5, choices=get_initial_data_list(), interactive=True)
        with gr.Column():
            submit_btn = gr.Button(value="Check data", scale=1, variant="primary")
        with gr.Column():
            delete_btn = gr.Button(value="Delete data", scale=1, variant="primary")
    # with gr.Row():
    #     list_data = gr.CheckboxGroup([], label="List deletable data", info="List current data", visible=False)
    with gr.Row():
        gallery = gr.Gallery(
        label="Generated images", show_label=False, elem_id="gallery"
        , columns=[3], rows=[1], selected_index=0, object_fit="contain", height="auto", interactive=True)
        temp_state = gr.State()
    gr.on(
        triggers=[submit_btn.click],
        fn=run_check_data,
        inputs=[data_list],
        outputs=[gallery])
    # Click event: Remove selected images
    gallery.select(fn=update_gallery, inputs=None, outputs=[temp_state])
    delete_btn.click(fn=remove_images, inputs=[data_list, gallery, temp_state], outputs=[gallery])
    # delete_btn.click(fn=gallery.select(fn=delete_data, inputs=None), inputs=gallery, outputs=[new_gallery])
    # gallery.select(fn=delete_data, inputs = None)
    # delete_btn.click(fn=delete_data, inputs=[gallery], outputs=[new_gallery])
    
    
