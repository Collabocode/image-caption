import gradio as gr

from constants.prompts import predefined_system_prompt, predefined_user_prompt
from core.inference import run_model
my_theme = gr.Theme.from_hub("lone17/kotaemon") #lone17/kotaemon
vllm_model_id="accounts/fireworks/models/llama-v3p2-11b-vision-instruct"

def main():
    with gr.Blocks(theme=my_theme, title='Multimodal Llama') as demo:
        with gr.Row():
            with gr.Column():
                img_path = gr.Image(label="Upload your image", type="filepath", height=400)
                with gr.Row():
                    vllm_info_model = gr.Textbox(label="VLLM Model", value=f'{vllm_model_id}',interactive=False)
                    vllm_token_used = gr.Textbox(label="Token used",interactive=False)
            with gr.Column(scale=3):
                sys_prompt = gr.Textbox(label="System prompt", value=f"{predefined_system_prompt}", lines=5, interactive=False)
                with gr.Row():
                    user_prompt = gr.Textbox(label="Input prompt", value=f"{predefined_user_prompt}", scale=5, lines=5, interactive=False)
                    submit_btn = gr.Button(value="Submit", scale=1, variant="primary")
                with gr.Row():
                    output_vllm = gr.Textbox(label="Output captions",scale=5, lines=5)
                with gr.Row(visible=False) as gallery_images:
                    clip_retrieved_images = gr.Gallery(
                    label="Retrieved Images", columns=2, rows=2
                )
        
        gr.on(
        triggers=[user_prompt.submit, submit_btn.click],
        fn=run_model,
        inputs=[img_path, sys_prompt, user_prompt, vllm_info_model],
        outputs=[gallery_images, clip_retrieved_images, output_vllm]
        )
    demo.launch(server_name='100.123.191.144', server_port=60000) #
if __name__ == '__main__':
    main()