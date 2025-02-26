import gradio as gr
from process import run_model, replace, lookup_button_handle
my_theme = gr.Theme.from_hub("lone17/kotaemon") #lone17/kotaemon
vllm_model_id="accounts/fireworks/models/llama-v3p2-90b-vision-instruct"

def main():
    with gr.Blocks(theme=my_theme, title='Multimodal Llama') as demo:
    # with gr.Blocks(title='Multimodal Llama') as demo:
        with gr.Row():
            with gr.Column():
                img_path = gr.Image(label="Upload your image", type="filepath", height=400)
                with gr.Row():
                    vllm_info_model = gr.Textbox(label="VLLM Model", value=f'{vllm_model_id}',interactive=False)
                    vllm_token_used = gr.Textbox(label="Token used",interactive=False)
                # debug_mode = gr.Checkbox(label="Masuk debug mode")
            with gr.Column(scale=3):
                out_vllm_english = gr.Textbox(label="Description", lines=5, interactive=False)
                with gr.Row():
                    input_text = gr.Textbox(label="Input prompt", visible=False, scale=5)
                    submit_btn = gr.Button(value="Submit", scale=1, variant="primary")
                with gr.Row(visible=False) as alternatives_row:
                    object_list = gr.Radio(label="Detected Objects", choices=[])
                    replace_button = gr.Button("Suggest Alternatives")
                with gr.Row(visible=False) as suggestions_row:
                    with gr.Column():
                        suggestion_list = gr.Radio(
                            label="Replacement Suggestions",
                            choices=[],
                        )
                        lookup_button = gr.Button("Look up options")
                with gr.Row(visible=False) as gallery_images:
                    retrieved_images = gr.Gallery(
                    label="Retrieved Images", columns=2, rows=2
                )
        # with gr.Row():
        #         
        #         price = gr.Textbox(label="Price in Rupiah",interactive=False)
        #         time = gr.Textbox(label="Time processing",interactive=False)
        gr.on(
        triggers=[input_text.submit, submit_btn.click],
        fn=run_model,
        inputs=[img_path, vllm_info_model],
        outputs=[out_vllm_english, object_list, object_list, alternatives_row, vllm_token_used],
        )
        replace_button.click(
            fn=replace,
            inputs=[object_list],
            outputs=[suggestion_list, suggestion_list, suggestions_row, ]
        )
        lookup_button.click(fn=lookup_button_handle,
                            inputs=[suggestion_list],
        outputs=[retrieved_images, gallery_images],)
        # with gr.Row():
        #     out_vllm_english = gr.Textbox(label="VLLM output in english",interactive=False)
        # gr.on(
        # triggers=[input_text.submit, submit_btn.click],
        # fn=run_model,
        # inputs=[img_path, input_text, sys_prompt, vllm_info_model, llm_info_model],
        # outputs=[ input_text_english, out_vllm_english, vllm_token_used, total_token_used, price, time],
        # )
    demo.launch(server_name='100.123.191.144', server_port=7892)
if __name__ == '__main__':
    main()