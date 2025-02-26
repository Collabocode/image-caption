import gradio as gr
import json
import glob
import os
import csv
my_theme = gr.Theme.from_hub("lone17/kotaemon") #lone17/kotaemon
def read_csv_output(path):
    new_list = []
    new_list.append(["PK ID", "IMAGES_IDS", "FULL TEXT"])   
    with open(path, mode ='r')as file:
        csvFile = csv.reader(file)
        count=0
        for lines in csvFile:
            if count==0:
                #--Skip header
                count+=1
                continue
            if lines[17]=='{}':
                continue
            if '[Image caption]' in lines[10]:
                new_list.append([lines[0], lines[17], lines[10] ])
            else: continue
    return gr.Dataset(samples=new_list)
def main():
    with gr.Blocks(theme=my_theme, title='Examples') as demo:
        submit_btn = gr.Button(value="Submit", scale=1, variant="primary")
        curr_path = "staging_datasets/"
        json_files = glob.glob(os.path.join(curr_path, '**/*.csv'), recursive=True)
        list_json_files = gr.Dropdown(choices=json_files)
        examples = gr.Examples(
        examples=[
            
            ["image path", "caption", "summary"],
        ],
        examples_per_page=10,
        inputs=[gr.Textbox(visible=False), gr.Textbox(visible=False), gr.Textbox(visible=False)],
    )
        submit_btn.click(read_csv_output, inputs=[list_json_files], outputs=[examples.dataset])
    demo.launch()
if __name__ == '__main__':
    main()