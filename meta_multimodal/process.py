import gradio as gr
from langfuse.openai import OpenAI
import json
import base64
import textwrap
from knowledge_base.kb import data
import os
client = OpenAI(
    # base_url="https://api.fireworks.ai/inference/v1",
    base_url="https://api.fireworks.ai/inference/v1/",
    api_key="iOADKSeF8E4ImIuj5Yg2256DCDmaAJ5j5Uj5AUVZGn2Xr7qU"
    ) 
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
def update_item_list(options):
    return gr.update(choices=options, value=options[0] if options else None, interactive=True)
def run_model(img_path, vllm_info_model):
    image_base64 = encode_image(img_path)
    text = textwrap.dedent(
            """
            Analyze the image to provide a 4-sentence description of the poster and the featured food items.
            The description should include any prominent food styling or decorations.
            Examples of food items to list include (but are not limited to): main dishes, sides, drinks, desserts, and any featured ingredients.

            Return results in the following format:
            {
                "description": "4-sentence description of the poster's design and food styling",
                "items": list of food items present in the image
            }

            Only list the food items you see in the image. List item names directly without additional text or explanations, 
            for example, "Chicken Teriyaki" instead of "plate of Chicken Teriyaki." or "Fries" instead of "McDonald's French Fries." 

            Return JSON as suggested, with no other text or explanations.
            """
        )
    prompt = [
        {
            "type": "text",
            "text": f'{text}',
        },
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{image_base64}"
            }
        }]
    msg = [{"role": "user", "content": prompt,}]
    vllm_response = client.chat.completions.create(
        model=vllm_info_model,
        messages=msg,
        temperature=0,
        max_tokens=16384,
        response_format={"type": "json_object"},
        top_p=1,)
    json_data = json.loads(vllm_response.choices[0].message.content)
    print(json_data)
    
    return json_data['description'], json_data['items'], update_item_list(json_data['items']), gr.update(visible=True),vllm_response.usage.total_tokens
def replace(object_list):
    key = object_list.lower()
    kb_data = data[key]
    return kb_data, update_item_list(kb_data), gr.update(visible=True), 
def lookup_button_handle(suggestion):
    # res = asyncio.run(API.retrieve_images(suggestion))
    images_with_descriptions = []
    images_with_descriptions.append(
        (
                os.path.join("/mnt/hdd1/jano/VisLang/meta_multimodal/data", "4.jpg"),
                suggestion['text'],
            )
        )
    return images_with_descriptions, gr.update(visible=True)
    
