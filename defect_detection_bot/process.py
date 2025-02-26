import gradio as gr
import base64
import time
import requests
import torch
from loguru import logger
from PIL import Image
import timeit
from langfuse.openai import OpenAI
from deep_translator import GoogleTranslator
from commons.gpt_vllm_instruction import VLLM_instruct_system_payment_verification, examples, VLLM_instruct_system, SYSTEM_PROMPT_VLLM_CATHERING, \
    VLLM_output_template, GPT_VLLM_System_instruction, GPT_JSON_FROMAT, SYSTEM_PROMPT_FROM_CC_TOOLS, SYSTEM_PROMPT_FROM_CC_TOOLS_FOOD_CATERING, \
    HOKA_HOKA_BENTO_SYSTEM_PROMPT, HHB_FOLLOWUP_PROMPT, HBB_CLASSIFICATION_PROMPT, HHB_BASE_SYSTEM_PROMPT, HOKA_HOKA_BENTO_PROMO_KB
from commons.apparel import regular_menu_list
import json
from models.visual_language_model import vllm_predict, vllm_predict_in_json, generate_base_prompt
from models.language_model import llm_predict, llm_followup_recommend, llm_predict_in_json
# from commons.hoka_hoka_bento import menu
from util import compute_cost
# Use any translator you like, in this example GoogleTranslator
translated_id_en = GoogleTranslator(source='id', target='en')
translated_en_id = GoogleTranslator(source='en', target='id')
client = OpenAI(
    base_url="https://api.fireworks.ai/inference/v1",
    api_key="iOADKSeF8E4ImIuj5Yg2256DCDmaAJ5j5Uj5AUVZGn2Xr7qU"
    )
def generate_g1_base_response(image_base64, text):
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
    
    messages = [
        {"role": "system", "content": """You are an expert AI assistant that explains your reasoning step by step. For each step, provide a title that describes what you're doing in that step, along with the content. Decide if you need another step or if you're ready to give the final answer. Respond in JSON format with 'title', 'content', and 'next_action' (either 'continue' or 'final_answer') keys. USE AS MANY REASONING STEPS AS POSSIBLE. AT LEAST 3. BE AWARE OF YOUR LIMITATIONS AS AN LLM AND WHAT YOU CAN AND CANNOT DO. IN YOUR REASONING, INCLUDE EXPLORATION OF ALTERNATIVE ANSWERS. CONSIDER YOU MAY BE WRONG, AND IF YOU ARE WRONG IN YOUR REASONING, WHERE IT WOULD BE. FULLY TEST ALL OTHER POSSIBILITIES. YOU CAN BE WRONG. WHEN YOU SAY YOU ARE RE-EXAMINING, ACTUALLY RE-EXAMINE, AND USE ANOTHER APPROACH TO DO SO. DO NOT JUST SAY YOU ARE RE-EXAMINING. USE AT LEAST 3 METHODS TO DERIVE THE ANSWER. USE BEST PRACTICES.

Example of a valid JSON response:
```json
{
    "title": "Identifying Key Information",
    "content": "To begin solving this problem, we need to carefully examine the given information and identify the crucial elements that will guide our solution process. This involves...",
    "next_action": "continue"
}```
"""},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": "Thank you! I will now think step by step following my instructions, starting at the beginning after decomposing the problem."}
    ]
    return messages
def run_g1_vllm_concept(image_base64, text, vllm_info_model):
    response = generate_g1_base_response(image_base64, text)
    steps = []
    step_count = 1
    total_thinking_time = 0
    total_token_used = 0
    #
    history_to_llm = []
    while True:
        start_time = time.time()
        step_data, token_used = vllm_predict_in_json(response, vllm_info_model)
        logger.info('\n' + f'{step_data}')
        end_time = time.time()
        thinking_time = end_time - start_time
        total_thinking_time += thinking_time
        steps.append((f"Step {step_count}: {step_data['title']}", step_data['content'], thinking_time))
        response.append({"role": "assistant", "content": json.dumps(step_data)})
        history_to_llm.append({"role": "assistant", "content": json.dumps(step_data)})
        if step_data['next_action'] == 'final_answer' or step_count > 25: # Maximum of 25 steps to prevent infinite thinking time. Can be adjusted.
            break
        step_count += 1
        total_token_used+=token_used
        # Yield after each step for Streamlit to update
        # yield steps, None, None  # We're not yielding the total time until the end
    # Generate final answer
    # response.append({"role": "user", "content": "Please provide the final answer based solely on your reasoning above."})
    # start_time = time.time()
    # final_data, token_used = vllm_predict(response, vllm_info_model)
    # logger.success(f'Final answer VLLM: {final_data}')
    # total_token_used+=token_used
    end_time = time.time()
    thinking_time = end_time - start_time
    total_thinking_time += thinking_time
    return response, history_to_llm, total_token_used
def run_vllm_model_single(image_base64, text, vllm_info_model):
    response = generate_base_prompt(image_base64, text)
    vllm_output, token_used = vllm_predict(response, vllm_info_model)
    logger.info(f'{vllm_output}')
    return vllm_output, [{"role": "assistant", "content": vllm_output}], token_used

def run_llm_model_chain(vllm_response, trans_input_text, llm_info_model):
    #--Step 1: LLM for classification whether promotion or not
    classify_msg = [
        {"role": "system", "content": f"{HBB_CLASSIFICATION_PROMPT}" }, 
        ]
    classify_msg+=vllm_response
    classify_msg.append({"role": "user", "content": "Please clasify. Reminder: Provide the response only in JSON format." })
    classify_response, classify_token_used = llm_predict_in_json(classify_msg, llm_info_model)
    if classify_response['menu_type'] =='promotion':
        for info in HOKA_HOKA_BENTO_PROMO_KB:
            if info['promo_name'].lower() == classify_response['promo_name'].lower():
                meta_info = info 
    else:
        meta_info = regular_menu_list
    #--Step 2: LLM for checking availability 
    check_stock_msg = [
        {"role": "system", "content": f"{HHB_BASE_SYSTEM_PROMPT}" },
        # {"role": "assistant", "content": f"Here is the reference information: {meta_info}"},
    ]
    check_stock_msg+=vllm_response
    # msg = [
    #     {"role": "system", "content": f'{HOKA_HOKA_BENTO_SYSTEM_PROMPT}' }]
    # msg+=vllm_response
    llm_input = f"Q:{trans_input_text}." + f"Here is the reference information: {meta_info}"
    #--Step 1: Check whether stock is available or not
    check_stock_prompt = llm_input + 'A: Let''s think step by step.'
    check_stock_msg.append({"role": "user", "content": check_stock_prompt})
    check_stock_response, check_stock_token_used = llm_predict(check_stock_msg, llm_info_model)
    logger.info(f'Check stock:\n{check_stock_response}')
    #--Step 2: Recommendation if out of stock
    fu_msg =[{"role": "system", "content": f'{HHB_FOLLOWUP_PROMPT}' },{"role": "assistant", "content": check_stock_response}]
    fu_msg.append({"role": "user", "content": 'Please recommend alternative menu if the content is out of stock. \
                   If the content is available, please encourage to order. Please respond in one of them.'})
    recommend_response, recommend_token_used = llm_followup_recommend(fu_msg, llm_info_model)
    llm_token_used = check_stock_token_used +  recommend_token_used
    cost = ((0.2 /1e6) * llm_token_used )* 15125.90
    #--Concate response
    concate_response = check_stock_response + '\n' + recommend_response
    return concate_response, concate_response, llm_token_used, cost

def run_model(img_path, input_text, vllm_info_model, llm_info_model):
    start = timeit.default_timer()
    image_base64 = encode_image(img_path)
    #--From indonesia to english
    trans_input_text = translated_id_en.translate(input_text)
    # trans_input_text = input_text
    # text_4_vllm = 'Is it a promotion or a menu? If it''s a promotion, identify the promotion name (the largest and most prominent text at the top of the image). If there is any highlighted item, summarize the highlighted set name and briefly describe the highlighted menu. Note: Highlighting is typically done with basic colors like red, green, or blue, and drawn by a pencil, brush, or similar tool.'
    text_4_vllm = "Please analyze this image of a menu or promotional material. Focus on any highlighted items, including those marked with a brush or other indicators. Summarize key information, such as item names, prices, and descriptions."
    if vllm_info_model == 'accounts/fireworks/models/llama-v3p2-90b-vision-instruct':

        # text_4_vllm+='\nRephrase and expand the question, and respond.'
        vllm_response, vllm_history_to_llm, vllm_token_used = run_vllm_model_single(image_base64, text_4_vllm, vllm_info_model)
    # text_4_vllm = 'What is in the image? Is it a promotion or a menu? If it''s a promotion, identify the promo name (usually in the largest font). If there is any highlighted item, summarize the highlighted set name and briefly describe the highlighted menu. Note: Highlighting is typically done with basic colors like red, green, or blue, and drawn with a pencil, brush, or similar tool.'
    else:
        # vllm_response, vllm_token_used, vllm_cost = run_vllm_model(image_base64, trans_input_text, vllm_info_model)
        vllm_response, vllm_history_to_llm, vllm_token_used =run_g1_vllm_concept(image_base64, text_4_vllm, vllm_info_model)
    
    #--Create input to LLM
    llm_response, final_response, llm_token_used, llm_cost = run_llm_model_chain(vllm_history_to_llm, trans_input_text, llm_info_model)
    
    #--Output mapping 
    img_str = f'<img src="data:image/jpeg;base64,{image_base64}" alt="user upload image" />' 
    img_str+= input_text

    #--Compute cost 
    vllm_cost = compute_cost(vllm_token_used, (0.9 /1e6), 15125.90)
    total_cost = vllm_cost + llm_cost
    str_cost = 'Rp. %.6f' % total_cost
    #--translate content to indonesia
    trans_response_text = translated_en_id.translate(final_response)
    stop = timeit.default_timer()
    time_process = "{:0,.2f}".format(stop - start) 
    # logger.info(f'Time process: {stop - start}')
    return [(img_str, trans_response_text)], trans_input_text, vllm_response, vllm_token_used, str_cost, time_process
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    
#--For now, we used VLLM for describe the image only 

# def run_llm_model_CoT(vllm_response, trans_input_text, llm_info_model):
#     # msg = [
#     #     {"role": "system", "content": f'{SYSTEM_PROMPT_FROM_CC_TOOLS_FOOD_CATERING}' }]
#     msg = [
#         {"role": "system", "content": f'{HOKA_HOKA_BENTO_SYSTEM_PROMPT}' }]
#     msg+=vllm_response
#     llm_input = f'Q:{trans_input_text}.'
#     # prompt_CoT_1 = llm_input + '\nRephrase and expand the question, and respond.'
#     prompt_CoT_1 = llm_input + 'A: Let''s think step by step.'
#     msg.append({"role": "user", "content": prompt_CoT_1})
#     #--LLM checking stock
#     llm_response, prev_llm_token_used = llm_predict(msg, llm_info_model)
#     logger.info(f'Check menu:{llm_response}')
#     #--LLM for followup recommendation if stock 
#     #--Set 

#     fu_msg =[{"role": "system", "content": f'{HHB_FOLLOWUP_PROMPT}' },{"role": "assistant", "content": llm_response}]
#     fu_msg.append({"role": "user", "content": 'Please recommend alternative menu if the content is out of stock. If the content is available, please encourage to order. Please respond in one of them.'})
#     llm_response, llm_token_used = llm_followup_recommend(fu_msg, llm_info_model)

#     llm_token_used = prev_llm_token_used 
#     # cost = ((0.2 /1e6) * llm_token_used )* 15125.90
    
#     #--Concate response
#     return llm_response, llm_token_used
# def run_vllm_model(image_base64, text, vllm_info_model):
# #     messages = [
# #         {"role": "system", "content": """You are an expert AI assistant that explains your reasoning step by step. For each step, provide a title that describes what you're doing in that step, along with the content. Decide if you need another step or if you're ready to give the final answer. Respond in JSON format with 'title', 'content', and 'next_action' (either 'continue' or 'final_answer') keys. USE AS MANY REASONING STEPS AS POSSIBLE. AT LEAST 3. BE AWARE OF YOUR LIMITATIONS AS AN LLM AND WHAT YOU CAN AND CANNOT DO. IN YOUR REASONING, INCLUDE EXPLORATION OF ALTERNATIVE ANSWERS. CONSIDER YOU MAY BE WRONG, AND IF YOU ARE WRONG IN YOUR REASONING, WHERE IT WOULD BE. FULLY TEST ALL OTHER POSSIBILITIES. YOU CAN BE WRONG. WHEN YOU SAY YOU ARE RE-EXAMINING, ACTUALLY RE-EXAMINE, AND USE ANOTHER APPROACH TO DO SO. DO NOT JUST SAY YOU ARE RE-EXAMINING. USE AT LEAST 3 METHODS TO DERIVE THE ANSWER. USE BEST PRACTICES.

# # Example of a valid JSON response:
# # ```json
# # {
# #     "title": "Identifying Key Information",
# #     "content": "To begin solving this problem, we need to carefully examine the given information and identify the crucial elements that will guide our solution process. This involves...",
# #     "next_action": "continue"
# # }```
# # """},
# #         {"role": "user", "content": prompt},
# #         {"role": "assistant", "content": "Thank you! I will now think step by step following my instructions, starting at the beginning after decomposing the problem."}
# #     ]
#     msg = [
#         {"role": "system", "content": f'You are a catering bot'},]
#     text = 'Q: What is in the image? Is it a promotion or a menu? If it''s a promotion, identify the promo name (mostly have the largest font).A: Let''s think step by step'
#     prompt = [
#         {
#             "type": "text",
#             "text": f'{text}',
#         },
#         {
#             "type": "image_url",
#             "image_url": {
#                 "url": f"data:image/jpeg;base64,{image_base64}"
#             }
#         }]
#     msg.append({"role": "user", "content": prompt})
#     prev_response, prev_token_used = vllm_predict(msg, vllm_info_model)
#     logger.info(f'Response 1: {prev_response}')
#     msg.append({"role": "assistant", "content": prev_response})
#     text = 'Q: Is there any item highlighted in the image? If yes, please summarize the set name and describe it. Note: Highlighting is typically done using basic colors like red, green, or blue, and is drawn with a pencil, brush, or similar tool.'
#     prompt = [
#         {
#             "type": "text",
#             "text": f'{text}',
#         },
#         ]
#     msg.append({"role": "user", "content": prompt})
#     curr_response, curr_token_used = vllm_predict(msg, vllm_info_model)
#     logger.info(f'Response 2: {curr_response}')
#     #-Computing cost
#     #--Get token used
#     vllm_token_used = prev_token_used + curr_token_used
#     #--Calculate cost for a single feed forward
#     if vllm_info_model =="accounts/fireworks/models/llama-v3p2-90b-vision-instruct":
#         cost = ((0.9 /1e6) * vllm_token_used )* 15125.90
#     else:
#         cost = ((0.2 /1e6) * vllm_token_used )* 15125.90
#     # str_cost = f'Rp. {cost}'
#     #--The output from VLLM temporary assign to contents in LLM
#     history_to_llm = msg = [
#         {"role": "assistant", "content": prev_response},
#         {"role": "assistant", "content": curr_response},
#         ]
#     return history_to_llm, vllm_token_used, cost