import gradio as gr
import base64
import requests
import torch
from loguru import logger
from PIL import Image
import timeit
from langfuse.openai import OpenAI
from deep_translator import GoogleTranslator
from commons.gpt_vllm_instruction import VLLM_instruct_system_payment_verification, examples, VLLM_instruct_system, \
    VLLM_output_template, GPT_VLLM_System_instruction, GPT_JSON_FROMAT, SYSTEM_PROMPT_FROM_CC_TOOLS

import json
# Use any translator you like, in this example GoogleTranslator
translated_id_en = GoogleTranslator(source='id', target='en')
translated_en_id = GoogleTranslator(source='en', target='id')
client = OpenAI(
    base_url="https://api.fireworks.ai/inference/v1",
    api_key="iOADKSeF8E4ImIuj5Yg2256DCDmaAJ5j5Uj5AUVZGn2Xr7qU"
    )
def rewrite_response(llm_response):
    keysList = list(llm_response.keys())
    
    for key in keysList:
        if key=='transaction_id':
            trans_id = llm_response[key]
        elif key=='status':
            status = llm_response[key]
        elif key=='recipient':
            try:
                name = llm_response[key]['name']
            except:
                name = 'confidential'
        elif key=='amount':
            try:
                value = str(llm_response[key]['value'])
            except:
                value = 'undetected'
            try:
                currency = str(llm_response[key]['currency'])
            except:
                currency = 'rupiah'
        elif key=='payment_date':
            date= str(llm_response[key])+ ' '
        elif key=='additional_details':
            name = str(llm_response[key])
        elif key=='admin_fee' and llm_response[key] !=0:
            admin_fee =llm_response[key]

    text = f'Your transaction with ID: {trans_id} to {name} is {status}.\n'+\
    f'The payment date is {date}, total: {value} in {currency} currency'
    try:
        text += f' Plus admin fee {admin_fee}'
    except:
        pass
    return text
def run_vllm_model(image_base64, text, vllm_info_model):
   
    text  = f'Q: Extract and analyze the text and context of the provided image. Please respond in a structured format.' + 'A: Let''s think step by step'
    # # msg = [
    #     {"role": "system", "content": f'{VLLM_instruct_system_payment_verification}' },]
    msg = [
        {"role": "system", "content": f'{VLLM_instruct_system}' },]
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
    msg.append({"role": "user", "content": prompt})
    vllm_response = client.chat.completions.create(
        model=vllm_info_model,
        messages=msg,
        temperature=0,
        # max_tokens=1024,
        # top_p=0.1,
    )
    #--Get token used
    vllm_token_used = vllm_response.usage.total_tokens
    #--Calculate cost for a single feed forward
    if vllm_info_model =="accounts/fireworks/models/llama-v3p2-90b-vision-instruct":
        cost = ((0.9 /1e6) * vllm_token_used )* 15125.90
    else:
        cost = ((0.2 /1e6) * vllm_token_used )* 15125.90
    # str_cost = f'Rp. {cost}'
    return vllm_response.choices[0].message.content, vllm_token_used, cost
def run_llm_model_CoT(vllm_response, trans_input_text, llm_info_model):
    msg = [
        {"role": "system", "content": f'{SYSTEM_PROMPT_FROM_CC_TOOLS}' }]
    llm_input = f'Extract all information from here {vllm_response} and fill the data following the template.'
    prompt_CoT_1 = llm_input + '\nRephrase and expand the question, and respond. ' + 'Ensure adherence to JSON format in your response; any other formats will no be accepted.'
    msg.append({"role": "user", "content": prompt_CoT_1})
    CoT_1_llm_response = client.chat.completions.create(
        model=llm_info_model,
        response_format={"type": "json_object"},
        messages=msg,
        temperature=0,
        max_tokens=1024,
        top_p=0.1,
        # top_k=50,
    )
    llm_token_used=CoT_1_llm_response.usage.total_tokens
    cost = ((0.2 /1e6) * llm_token_used )* 15125.90
    print(CoT_1_llm_response.choices[0].message.content)
    #--Get dict 
    dict_llm_response = json.loads(CoT_1_llm_response.choices[0].message.content) 
    final_response = rewrite_response(dict_llm_response)
    
    return CoT_1_llm_response.choices[0].message.content, final_response, llm_token_used, cost
def run_model(img_path, input_text, vllm_info_model, llm_info_model):
    start = timeit.default_timer()
    image_base64 = encode_image(img_path)
    #--From indonesia to english
    trans_input_text = translated_id_en.translate(input_text)
    vllm_response, vllm_token_used, vllm_cost = run_vllm_model(image_base64, None, vllm_info_model)
    
    #--Create input to LLM
    llm_response, final_response, llm_token_used, llm_cost = run_llm_model_CoT(vllm_response, trans_input_text, llm_info_model)
    # llm_input = trans_input_text + ' from this: ' + vllm_response
    # llm_response, final_response, llm_token_used, llm_cost = run_llm_model_CoT(vllm_response, trans_input_text, llm_info_model)
    
    #--Output mapping 
    img_str = f'<img src="data:image/jpeg;base64,{image_base64}" alt="user upload image" /> <br/>' 
    img_str+= input_text
    # llm_response = 0
    # llm_token_used = 0
    #--Total cost
    total_cost = vllm_cost + llm_cost
    str_cost = 'Rp. %.6f' % total_cost
    #--translate content to indonesia
    trans_response_text = translated_en_id.translate(final_response)
    stop = timeit.default_timer()
    time_process = "{:0,.2f}".format(stop - start) 
    # logger.info(f'Time process: {stop - start}')
    return [(img_str, trans_response_text)], trans_input_text, vllm_response, llm_response, vllm_token_used, \
    llm_token_used, (vllm_token_used + llm_token_used), str_cost, time_process
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    
#--For now, we used VLLM for describe the image only 
#--Specific to currency:

# text = f'What is this?\nRephrase and expand the question, and respond. If there is a word indicates by ''BERHASIL'', the status will be BERHASIL. If there is a word indicates by ''GAGAL'', the status will be GAGAL. ' + \
#     'Identify The bank name in the watermark. If there is ''BIAYA ADMIN'', please put in admin_fee. Otherwise, set to 0'

# text = 'You are tasked with identifying and processing transaction details based on the provided payment receipt.\ **Step 1:** If the word "BERHASIL" is present, set the **status** to "BERHASIL". If no, set the **status** to "GAGAL". **Step 2:** Identify and extract the **bank name** from the watermark of the image. **Step 3:** If the phrase "BIAYA ADMIN" is present, extract the **admin_fee**. If not, set the **admin_fee** to 0.'
# text = f'Please understand all the text within the image and the context?\nRephrase and expand the question, and respond in json structured format. '
# text  = 'Let''s think step by step. Extract and understand the text and context of the provided image'