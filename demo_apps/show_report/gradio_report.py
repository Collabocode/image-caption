import gradio as gr
import json
import glob
import os
my_theme = gr.Theme.from_hub("lone17/kotaemon") #lone17/kotaemon
def read_json_output(path):
    # path = "data/currencies/references/currency_references_v1.json"
    with open(path) as f:
        d = json.load(f)
    #--Only show caption and img_path
    new_list = []
    new_list.append(["/mnt/hdd1/jano/VisLang/icon.png", "CAPTION", "SUMMARY"])
    for data in d:
        # try:
        new_list.append([os.path.join('/mnt/hdd1/jano/VisLang/',data['img_path']), data['caption'], data['summary'] ])
        # except:
        #     new_list.append([data['img_path'], data['caption_analyze']])
    return gr.Dataset(samples=new_list)
def main():
    with gr.Blocks(theme=my_theme, title='Examples') as demo:
        submit_btn = gr.Button(value="Submit", scale=1, variant="primary")
        curr_path = "/mnt/hdd1/jano/VisLang/results/"
        json_files = glob.glob(os.path.join(curr_path, '**/*.json'), recursive=True)
        list_json_files = gr.Dropdown(choices=json_files, value=json_files[0])
        examples = gr.Examples(
        examples=[
            ["/mnt/hdd1/jano/VisLang/icon.png", "caption", "summary"],
        ],
        inputs=[gr.Image(visible=False), gr.Textbox(visible=False), gr.Textbox(visible=False)],
    )
        submit_btn.click(read_json_output, inputs=[list_json_files], outputs=[examples.dataset])
    demo.launch(server_name='100.123.191.144',server_port=41251)
if __name__ == '__main__':
    main()