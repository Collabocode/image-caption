import gradio as gr
from PIL import Image
import timeit
from local_vision_llama import model_id, model, processor

def run_model(img_path, text, new_token):
    start = timeit.default_timer()
    image = Image.open(img_path)
    prompt = f"<|image|><|begin_of_text|>{text}\n Answer: "
    inputs = processor(image, prompt, return_tensors="pt").to(model.device)
    
    output = model.generate(**inputs, max_new_tokens=new_token)
    raw_output = processor.decode(output[0])
    final_output = raw_output.replace("<|image|>", "").replace("<|begin_of_text|>", "").replace(f"{text}", "")
    stop = timeit.default_timer()
    parse_output = f"Raw: {raw_output}\n\n\n" + f"Parsed: {final_output}\n\n\n" + f"Time: {stop - start}"
    return parse_output
def generate_header_gradio():
    gr.HTML("<h1><center>Playground for Vision-Large Language Model (VLLM)</center></h1>")
    gr.HTML("<h3 style=""color:white;""> Welcome to Collabocode playground! Here is some tips: </h3>")
    gr.HTML("<h5 style=""color:white;""> 1. Upload image with type (jpeg, jpg, png, svg, gif, bmp, webp) </h5>")
    gr.HTML("<h5 style=""color:white;""> 2. Optional: You can use question or instruction in User Prompt</h5>")
    gr.HTML("<h5 style=""color:white;""> 3. Optional: Set the new token. Longer is better but takes more time </h5>")
    gr.HTML("<h5 style=""color:white;""> 4. Cheat prompt: You are an advanced image analysis bot. Your primary role is to analyze visual input, extract meaningful information, and provide accurate, context-aware interpretations strictly based on the image. Do not include information outside of the image or speculate. If the message contains an inquiry, answer only using details visible within the image. Interpret the scene, objects, actions, or elements within an image to provide a descriptive summary. </h5>")
def main():
    with gr.Blocks(theme='John6666/YntecDark', title='Playground with Local Vision-Llama 11B') as demo:
        generate_header_gradio()
        with gr.Row():
            with gr.Column():
                img_path = gr.Image(label="Upload your image", type="filepath", height=400)
                with gr.Row():
                    vllm_info_model = gr.Textbox(label="VLLM Model", value=f'{model_id}',interactive=False)
                    with gr.Column():
                        new_token = gr.Slider(label="New token", minimum=30, maximum=1024)
            with gr.Column():
                prompt = gr.Textbox(label="User prompt", value="", lines=5, interactive=True)
                with gr.Row():
                    submit_btn = gr.Button(value="Submit", scale=1, variant="primary")
                out_text = gr.Textbox(label="Output text", value="", lines=5, interactive=False)
        gr.on(
        triggers=[prompt.submit, submit_btn.click],
        fn=run_model,
        inputs=[img_path, prompt, new_token],
        outputs=[out_text]
        )
    demo.launch(server_name='100.123.191.144', server_port=1234)
if __name__ == '__main__':
    main()