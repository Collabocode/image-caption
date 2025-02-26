from langfuse.openai import OpenAI
import json
client = OpenAI(
    base_url="https://api.fireworks.ai/inference/v1/",
    api_key="iOADKSeF8E4ImIuj5Yg2256DCDmaAJ5j5Uj5AUVZGn2Xr7qU"
    )
def generate_summary_prompt(system_prompt, user_prompt, caption):
    messages = []
    messages.append({"role": "system", "content":f"{system_prompt}"})    
    messages.append({"role": "user", "content": f"{user_prompt} {caption}"})
    
    return messages

def llm_predict(msg, llm_info_model, temperature=0, 
                max_tokens=1024, top_p=0.1, is_json=False):
    if is_json:
        llm_response = client.chat.completions.create(
        model=llm_info_model,
        messages=msg,
        response_format={ "type": "json_object" },
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=1,
        )
        #--Parse the text:
        try:
            response = json.loads(llm_response.choices[0].message.content)
            return response['text'], llm_response.usage.total_tokens
        except:
            return llm_response.choices[0].message.content, llm_response.usage.total_tokens
    else:   
        llm_response = client.chat.completions.create(
            model=llm_info_model,
            messages=msg,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=1,
        )
    
        return llm_response.choices[0].message.content, llm_response.usage.total_tokens
