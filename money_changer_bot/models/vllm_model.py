from langfuse.openai import OpenAI
import json
client = OpenAI(
    # base_url="https://api.fireworks.ai/inference/v1",
    base_url="https://api.fireworks.ai/inference/v1/",
    api_key="iOADKSeF8E4ImIuj5Yg2256DCDmaAJ5j5Uj5AUVZGn2Xr7qU"
    )
def generate_prompt(base64, system_prompt, user_prompt):
    msg = []
    msg.append({"role": "system", "content":f"{system_prompt}"})
    
    user_prompt = [
        {
            "type": "text",
            "text": f'{user_prompt}',
        },
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64}"
            }
        }]
    msg.append({"role": "user", "content": user_prompt})
    return msg
def run_vllm(base64, system_prompt, user_prompt, vllm_info_model):
    msg = generate_prompt(base64, system_prompt, user_prompt)
    vllm_response = client.chat.completions.create(
        model=vllm_info_model,
        messages=msg,
        temperature=0,
        max_tokens=16384,
        top_p=1,
        presence_penalty=0,
        frequency_penalty=0,
        extra_body={'top_k':1, }
    )
    return vllm_response.choices[0].message.content
