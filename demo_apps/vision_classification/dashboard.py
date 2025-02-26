import gradio as gr
import sys
sys.path.append('/mnt/hdd1/jano/VisLang/demo_apps/')
from tabs.tab_upload_image import generate_upload_tab
from tabs.tab_show_image import generate_show_data_tab
from tabs.tab_delete_image import generate_delete_data_tab
with gr.Blocks(theme='John6666/YntecDark',) as demo:
    with gr.Tab("Upload image to server"):
        generate_upload_tab()
    with gr.Tab("Show current data in server"):
        generate_show_data_tab()
    with gr.Tab("Delete unnecessary image in server"):
        generate_delete_data_tab()
demo.launch(server_name='100.123.191.144', server_port=1331)
