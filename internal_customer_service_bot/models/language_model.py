from langfuse.openai import OpenAI
import json
client = OpenAI(
    base_url="https://api.fireworks.ai/inference/v1",
    api_key="iOADKSeF8E4ImIuj5Yg2256DCDmaAJ5j5Uj5AUVZGn2Xr7qU"
    )
def llm_predict(msg, llm_info_model, temperature=0, max_tokens=1024, top_p=0.1, stream=False):
    llm_response = client.chat.completions.create(
        model=llm_info_model,
        messages=msg,
        temperature=temperature,
        max_tokens=16384,
        stream=stream
        # top_p=0.1,
    )
    # try:
    if stream:
        return llm_response, None
    return llm_response.choices[0].message.content, llm_response.usage.total_tokens
def llm_predict_in_json(msg, llm_info_model, temperature=0, max_tokens=1024, top_p=0.1):
    llm_response = client.chat.completions.create(
        model=llm_info_model,
        messages=msg,
        temperature=temperature,
        max_tokens=16384,
        response_format={"type": "json_object"},
        # top_p=0.1,
    )
    # try:
    return json.loads(llm_response.choices[0].message.content), llm_response.usage.total_tokens
    # except:
    #     return llm_response, None
def llm_followup_recommend(msg, llm_info_model, temperature=0, max_tokens=1024, top_p=0.1):
    llm_response = client.chat.completions.create(
        model=llm_info_model,
        messages=msg,
        temperature=temperature,
        max_tokens=16384,
        # top_p=0.1,
    )
    # try:
    return llm_response.choices[0].message.content, llm_response.usage.total_tokens