import gradio as gr
# from process import run_model
# from simple_process import run_model
from generate_kb_process import run_model
# personalized_system_prompt = "You are a Money Changer Advisor. Your task is to assist users with any issues related to currency-related topics.\n\nGuidelines:\n1. Analyze the Given Input: Determine the input format (e.g., text, image, document, or data file) and understand the context of the information it contains. Adapt the approach based on the type of task (e.g., vehicle detection, text extraction, sentiment analysis, summarization).\n2. Extract Any Relevant Text: Apply Optical Character Recognition (OCR) to extract readable text. Ensure accuracy in text extraction, especially in cases requiring precision (e.g., license plate recognition, quantitative data).\nNOTES Do not extract the design, color scheme, and nonalphabet text."
# personalized_prompt = "You are a catering assistant bot designed to help users plan, customize, and manage catering services for events."
personalized_system_prompt = "You are a catering assistant bot designed to help users plan, customize, and manage catering services for events.\n\nGuidelines:\n1. Analyze the Given Input: Determine the input format (e.g., text, image, document, or data file) and understand the context of the information it contains. Adapt the approach based on the type of task (e.g., vehicle detection, text extraction, sentiment analysis, summarization).\n2. Extract Any Relevant Text: Apply Optical Character Recognition (OCR) to extract readable text. Ensure accuracy in text extraction, especially in cases requiring precision (e.g., license plate recognition, quantitative data).\nNOTES Do not extract the design, color scheme, and nonalphabet text."
user_prompt= "Please analyze"
my_theme = gr.Theme.from_hub("lone17/kotaemon") #lone17/kotaemon
vllm_model_id="accounts/fireworks/models/llama-v3p2-90b-vision-instruct"
# vllm_model_id="accounts/fireworks/models/llama-v3p2-90b-vision-instruct"
llm_model_id = "X"

def main():
    with gr.Blocks(theme=my_theme, title='Generate Product Knowledge (PK) Bot') as demo:
        with gr.Row():
            with gr.Column():
                img_path = gr.Image(label="Upload gambar anda", type="filepath", height=400)
                with gr.Row():
                    vllm_info_model = gr.Textbox(label="VLLM Model", value=f'{vllm_model_id}',interactive=False)
                    llm_info_model = gr.Textbox(label="LLM Model", value=f'{llm_model_id}',interactive=False)
                # debug_mode = gr.Checkbox(label="Masuk debug mode")    
            with gr.Column(scale=3):
                # sys_prompt = gr.Textbox(label="SYSTEM PROMPT", lines=10, value="You are a very helpful assistant.", interactive=True)
                sys_prompt = gr.Textbox(label="SYSTEM PROMPT", lines=10, value=f"{personalized_system_prompt}", interactive=True)
                # chatbot = gr.Textbox(label="Apa yang bisa dibantu?", scale=5)
                with gr.Row():
                    input_text = gr.Textbox(label="Apa yang harus saya lakukan?", value=f"{user_prompt}", interactive=True, scale=5)
                    submit_btn = gr.Button(value="Kirim", scale=1, variant="primary")
        with gr.Row():
                vllm_token_used = gr.Textbox(label="VLLM token yang terpakai",interactive=False)
                total_token_used = gr.Textbox(label="Total token yang terpakai",interactive=False)
                price = gr.Textbox(label="Harga yang dibayar (dalam rupiah)",interactive=False)
                time = gr.Textbox(label="Waktu yang diperlukan",interactive=False)
        with gr.Row():
            with gr.Column():
                input_text_english = gr.Textbox(label="Input text english",interactive=False)
            with gr.Column(scale=2):
                out_vllm_english = gr.Textbox(label="VLLM output in english",interactive=False)
        gr.on(
        triggers=[input_text.submit, submit_btn.click],
        fn=run_model,
        inputs=[img_path, input_text, sys_prompt, vllm_info_model, llm_info_model],
        outputs=[ input_text_english, out_vllm_english, vllm_token_used, total_token_used, price, time],
        )
    demo.launch(server_name='100.123.191.144', server_port=7888)
if __name__ == '__main__':
    main()