import gradio as gr
# from process import run_model
from simple_process import run_model, build_history

my_theme = gr.Theme.from_hub("lone17/kotaemon") #lone17/kotaemon
# vllm_model_id="accounts/fireworks/models/llama-v3p2-11b-vision-instruct"
vllm_model_id="accounts/fireworks/models/llama-v3p2-90b-vision-instruct"
llm_model_id = "accounts/fireworks/models/llama-v3p1-8b-instruct"

def main():
    with gr.Blocks(theme=my_theme, title='Internal Collabocode Bot') as demo:
        with gr.Row():
            with gr.Column():
                img_path = gr.Image(label="Upload gambar anda", type="filepath", height=400)
                with gr.Row():
                    vllm_info_model = gr.Textbox(label="VLLM Model", value=f'{vllm_model_id}',interactive=False)
                    llm_info_model = gr.Textbox(label="LLM Model", value=f'{llm_model_id}',interactive=False)
                # debug_mode = gr.Checkbox(label="Masuk debug mode")    
            with gr.Column(scale=3):
                chatbot = gr.Chatbot()
                with gr.Row():
                    input_text = gr.Textbox(label="Apa yang bisa dibantu?", scale=5)
                    submit_btn = gr.Button(value="Kirim", scale=1, variant="primary")
        with gr.Row():
                vllm_token_used = gr.Textbox(label="VLLM token yang terpakai",interactive=False)
                llm_token_used = gr.Textbox(label="LLM token yang terpakai",interactive=False)
                total_token_used = gr.Textbox(label="Total token yang terpakai",interactive=False)
                price = gr.Textbox(label="Harga yang dibayar (dalam rupiah)",interactive=False)
                time = gr.Textbox(label="Waktu yang diperlukan",interactive=False)
        with gr.Row():
            with gr.Column():
                input_text_english = gr.Textbox(label="Input text english",interactive=False)
            with gr.Column(scale=2):
                out_vllm_english = gr.Textbox(label="VLLM output in english",interactive=False)
            with gr.Column(scale=2):
                out_llm_english = gr.Textbox(label="LLM output in english",interactive=False)
        gr.on(
        triggers=[input_text.submit, submit_btn.click],
        fn=run_model,
        inputs=[chatbot, img_path, input_text, vllm_info_model, llm_info_model],
        outputs=[chatbot, input_text_english, out_vllm_english, out_llm_english, vllm_token_used, llm_token_used, total_token_used, price, time],
        # queue=False
        ) #.then(build_history, chatbot, chatbot)
    demo.launch(server_name='100.123.191.144', server_port=7862)
if __name__ == '__main__':
    main()