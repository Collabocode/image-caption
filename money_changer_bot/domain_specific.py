import gradio as gr
import base64
import requests
import torch
from loguru import logger
from PIL import Image
import timeit
from langfuse.openai import OpenAI
from deep_translator import GoogleTranslator
from commons.vllm_instructions import VLLM_instruct_currency_converter
from commons.llm_instructions import LLM_step_by_step_template, LLM_instruct_currency_converter, Instruct_CoT_1_LLM, \
    generate_CoT_2_LLM, DS_money_changer_system_prompt, instruct_LLM_money_changer
from commons.gpt_instructions import GPT_instruction_currency, GPT_intro_currency, GPT_examples
from commons.knowledge_bases import USD_EXCHANGE_RATES, KD_Prompt
from commons.utils import rewrite_final_currency_output
import json
# Use any translator you like, in this example GoogleTranslator
translated_id_en = GoogleTranslator(source='id', target='en')
translated_en_id = GoogleTranslator(source='en', target='id')
client = OpenAI(
    base_url="https://api.fireworks.ai/inference/v1",
    api_key="iOADKSeF8E4ImIuj5Yg2256DCDmaAJ5j5Uj5AUVZGn2Xr7qU"
    )
def compute_conversion(exchange_rate, current_value, target_exchange_rate):
    return float(exchange_rate) * float(current_value) * float(1/target_exchange_rate)

def run_llm(vllm_response, text_input, llm_info_model):
    # msg = [
    #         {"role": "system", "content": f'You are a helpful assistant.' }]
    # msg.append({"role": "assistant", "content": vllm_response})
    # prompt = text + '\nRephrase and expand the question, and respond.' + ' ' + KD_Prompt + ' ' + LLM_step_by_step_template + ' ' \
    #      + LLM_instruct_currency_converter
    llm_input = text_input + ' from this: ' + vllm_response
    prompt = llm_input + '\nRephrase and expand the question, and respond.' + ' ' + GPT_intro_currency + ' ' + GPT_instruction_currency + ' ' +\
        LLM_instruct_currency_converter
    # msg.append()
    llm_response = client.chat.completions.create(
        model=llm_info_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    llm_token_used = llm_response.usage.total_tokens

    #--References from Julius
    #--If the respond is not JSON format do one more
    try:
        metadata = json.loads(llm_response.choices[0].message.content)
        #--If in the expected format, do calculation
        final_price =compute_conversion(metadata['exchange_rate_to_usd'], metadata['current_value'], metadata['target_currency'])
    except:
        ## Julius
        prev_response = llm_response.choices[0].message.content
        msg = [
            {"role": "system", "content": f'You are a helpful assistant.' }]
        msg.append({"role": "assistant", "content": prev_response})
        # msg.append({"role": "user", "content": "There is formatting error. Answer with JSON only."})
        msg.append({"role": "user", "content": "Respond with JSON only."})
        logger.info(msg)

        llm_response = client.chat.completions.create(
            model=llm_info_model,
            response_format={"type": "json_object"},
            messages=msg,
            temperature=0,
            max_tokens=1024,
            top_p=0.1,
            top_k=50,
            )
        
        llm_token_used+=llm_response.usage.total_tokens
        logger.warning(f'Reformat to JSON used token: {llm_response.usage.total_tokens}')
        #--Compute conversion 
        #--If LLM still incapable
        try:
            metadata = json.loads(llm_response.choices[0].message.content)
            # exchange_rate, current_value, target_currency
            final_price =compute_conversion(metadata['exchange_rate_to_usd'], metadata['current_value'], metadata['target_exchange_rate'])
        except:
            final_response = f'I''m sorry i cannot convert the given currency. Can you provide me more detail about the target currency? '
    cost = ((0.2 /1e6) * llm_token_used )* 15125.90
    combine_llm_response = prev_response + '\n\n\n' + 'Reformat to json: ' + llm_response.choices[0].message.content
    #--Rewrite for final output
    try:
        final_response = rewrite_final_currency_output(metadata, final_price)
    except:
        pass
    return combine_llm_response, final_response, llm_token_used, cost

def run_vllm_model(image_base64, text, vllm_info_model):
    #--For now, we used VLLM for describe the image only 
    #--Specific to currency:
    text = f'What is this?\nRephrase and expand the question, and respond. {VLLM_instruct_currency_converter}' 
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
    msg = [
        {"role": "system", "content": f'You are a helpful assistant.' },
        {"role": "user", "content": prompt},
    ]
    vllm_response = client.chat.completions.create(
        model=vllm_info_model,
        messages=msg,
        temperature=0,
    )
    #--Get token used
    vllm_token_used = vllm_response.usage.total_tokens
    #--Calculate cost for a single feed forward
    if vllm_info_model =="accounts/fireworks/models/llama-v3p2-90b-vision-instruct":
        cost = ((0.9 /1e6) * vllm_token_used )* 15125.90
    else:
        cost = int(((0.2 /1e6) * vllm_token_used )* 15125.90)
    # str_cost = f'Rp. {cost}'
    return vllm_response.choices[0].message.content, vllm_token_used, cost
def run_llm_model_CoT(vllm_response, text_input, llm_info_model):
    llm_input = text_input + ' from this: ' + vllm_response
    

    #--Run LLM with system prompt
    msg = [
            {"role": "system", "content": f'{DS_money_changer_system_prompt}' }]
    #--Do rephrase and 
    prompt_CoT_1 = llm_input + '\nRephrase and expand the question, and respond. ' + instruct_LLM_money_changer
    msg.append({"role": "user", "content": prompt_CoT_1})
    CoT_1_llm_response = client.chat.completions.create(
        model=llm_info_model,
        response_format={"type": "json_object"},
        messages=msg,
        temperature=0,
    )
    
    base_response = json.loads(CoT_1_llm_response.choices[0].message.content)
    # prompt_CoT_2 = generate_CoT_2_LLM(base_response['source_currency'], base_response['target_currency'])
    # CoT_2_llm_response = client.chat.completions.create(
    #     model=llm_info_model,
    #     response_format={"type": "json_object"},
    #     messages=[{"role": "user", "content": prompt_CoT_2}],
    #     temperature=0,
    # )
    # updated_dict = json.loads(CoT_2_llm_response.choices[0].message.content)
    # base_response.update(updated_dict)
    final_price =compute_conversion(base_response['source_exchange_rate'], base_response['source_value'], base_response['target_exchange_rate'])
    llm_token_used = CoT_1_llm_response.usage.total_tokens
    cost = ((0.2 /1e6) * llm_token_used )* 15125.90
    combine_llm_response = CoT_1_llm_response.choices[0].message.content
    final_response = rewrite_final_currency_output(base_response, final_price)
    return combine_llm_response, final_response, llm_token_used, cost

    # print(CoT_2_llm_response.choices[0].message.content)
    # source_ER_response = json.loads(CoT_2_llm_response.choices[0].message.content)
    

def run_model(img_path, input_text, vllm_info_model, llm_info_model):
    start = timeit.default_timer()
    image_base64 = encode_image(img_path)
    #--From indonesia to english
    trans_input_text = translated_id_en.translate(input_text)
    vllm_response, vllm_token_used, vllm_cost = run_vllm_model(image_base64, None, vllm_info_model)
    
    llm_response, final_response, llm_token_used, llm_cost = run_llm_model_CoT(vllm_response, trans_input_text, llm_info_model)
    
    #--Output mapping 
    img_str = f'<img src="data:image/jpeg;base64,{image_base64}" alt="user upload image" /> <br/>' 
    img_str+= input_text

    #--Total cost
    total_cost = vllm_cost + llm_cost
    str_cost = 'Rp. %.2f' % total_cost
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