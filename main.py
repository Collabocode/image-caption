import gradio as gr
from langfuse_deps import caption_lfprompt, summary_lfprompt, classify_lfprompt
from inference import run_model
my_theme = gr.Theme.from_hub("lone17/kotaemon") #lone17/kotaemon
vllm_model_id="accounts/fireworks/models/llama-v3p2-11b-vision-instruct"
llm_model_id="accounts/fireworks/models/llama-v3p1-8b-instruct"
def main():
    with gr.Blocks(theme=my_theme, title='Multimodal Llama') as demo:
        with gr.Row():
            with gr.Column():
                img_path = gr.Image(label="Upload your image", type="filepath", height=400)
                with gr.Row():
                    vllm_info_model = gr.Textbox(label="VLLM Model", value=f'{vllm_model_id}',interactive=False)
                    llm_info_model = gr.Textbox(label="LLM Model", value=f'{llm_model_id}',interactive=False)
                with gr.Row():
                    vllm_token_used = gr.Textbox(label="VLLM Token used",interactive=False)
                    llm_token_used = gr.Textbox(label="LLM Token used",interactive=False)
            with gr.Column(scale=3):
                sys_prompt_caption = gr.Textbox(label="System prompt caption", value=f"{caption_lfprompt[0]['content']}", lines=5, interactive=False)
                user_prompt_caption = gr.Textbox(label="User prompt caption", value=f"{caption_lfprompt[1]['content']}", lines=1, interactive=False)
                
                sys_prompt_summary = gr.Textbox(label="System prompt summary", value=f"{summary_lfprompt[0]['content']}", lines=5, interactive=False)
                user_prompt_summary = gr.Textbox(label="User prompt summary", value=f"{summary_lfprompt[1]['content']}", lines=1, interactive=False)
                
                sys_prompt_classify = gr.Textbox(label="System prompt classification", value=f"{classify_lfprompt[0]['content']}", lines=5, interactive=False)
                user_prompt_classify = gr.Textbox(label="User prompt classification", value=f"{classify_lfprompt[1]['content']}", lines=1, interactive=False)
                
                submit_btn = gr.Button(value="Submit", scale=1, variant="primary")
        with gr.Row():
            output_vllm = gr.Textbox(label="Output captions",scale=5, lines=5)
            output_llm = gr.Textbox(label="Output summary",scale=5, lines=5)
            output_classified = gr.Textbox(label="Output classification",scale=5, lines=5)
        gr.on(
        triggers=[sys_prompt_summary.submit, submit_btn.click],
        fn=run_model,
        inputs=[img_path, sys_prompt_caption, user_prompt_caption, 
                sys_prompt_summary, user_prompt_summary,
                sys_prompt_classify, user_prompt_classify,
                vllm_info_model, llm_info_model],
        outputs=[output_vllm, output_llm, output_classified, vllm_token_used, llm_token_used]
        )
    demo.launch(server_name='100.123.191.144', server_port=1212) #
if __name__ == '__main__':
    main()