import gradio as gr
# from process import run_model
from vllm_only_process import run_model

my_theme = gr.Theme.from_hub("lone17/kotaemon") #lone17/kotaemon
# vllm_model_id="accounts/fireworks/models/llama-v3p2-11b-vision-instruct"
vllm_model_id="accounts/fireworks/models/llama-v3p2-90b-vision-instruct"
llm_model_id = "accounts/fireworks/models/llama-v3p1-8b-instruct"

def main():
    with gr.Blocks(theme=my_theme, title='Defect Detection Bot') as demo:
        with gr.Row():
            with gr.Column():
                img_path = gr.Image(label="Upload gambar anda", type="filepath", height=400)
                with gr.Row():
                    vllm_info_model = gr.Textbox(label="VLLM Model", value=f'{vllm_model_id}',interactive=False)
                # debug_mode = gr.Checkbox(label="Masuk debug mode")    
            with gr.Column(scale=3):
                chatbot = gr.Chatbot()
                with gr.Row():
                    input_text = gr.Textbox(label="Apa yang bisa dibantu?", scale=5)
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
        inputs=[img_path, input_text, vllm_info_model],
        outputs=[chatbot, out_vllm_english, vllm_token_used, total_token_used, price, time],
        )
    demo.launch()
if __name__ == '__main__':
    main()
#--iNI WORK
''' You are provided with an image of a food menu. Focus on extracting details specifically for the highlighted items on the menu. Here are the key pieces of information you need to extract:

Dish Name: What is the name of the highlighted dish?
Price: What is the price listed for this dish?
Stock or Availability: Is there any indication of stock or availability?
Ingredients or Description: Is there any additional information provided about the dish’s ingredients or preparation?
Promotions: Is there any special mention of promotions or deals related to this item?
Visual Description: Describe the visual representation of the dish (e.g., color, layout).
Other Features: Are there any "Details" or "Order" buttons present that allow further action related to the dish?
'''