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
from models.visual_language_model import vllm_predict, vllm_predict_in_json, generate_base_prompt, debug_vllm
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

def run_model(img_path, input_text, vllm_info_model):
    start = timeit.default_timer()
    image_base64 = encode_image(img_path)
    text_4_vllm = input_text
    #--From indonesia to english
    # text_4_vllm = "You are presented with an image of a food menu showcasing various dishes from a restaurant. Each dish includes the following details: Dish Name: The title of the dish (e.g., Chicken Curry Yaki). Price: The cost of the dish (e.g., Rp. 30,500). Visual Representation: A small image depicting the dish. Ordering Options: Buttons labeled ""Details"" and ""+ Order"" for each dish, suggesting that customers can view more information or place an order online. Your task is to analyze the content of the menu, focusing on identifying the highlighted dish and understanding its significance. Additionally, consider the overall structure of the menu, the pricing strategy, and any other notable elements that might indicate the restaurant's offerings or marketing approach. Questions to consider: What dish is highlighted or emphasized, and why might it be significant? How does the pricing compare among the various dishes? What does the presence of ordering buttons suggest about the restaurant's service model? Are there any promotional or marketing strategies evident from the layout or presentation of the menu?"
    # text_4_vllm = "You are presented with an image of a food menu showcasing various dishes from a restaurant. Each dish includes the following details: Dish Name: The title of the dish (e.g., Chicken Curry Yaki). Price: The cost of the dish (e.g., Rp. 30,500). Visual Representation: A small image depicting the dish. Your task is to analyze the content of the menu, focusing on identifying the highlighted dish and understanding its significance. Questions to consider: What dish is highlighted or emphasized?"
    # text_4_vllm ="Look at the image of the menu, and provide a detailed analysis of the food items shown. Focus on the dishes that are circled, mentioning their names, prices, and a brief description of each. Also, briefly describe other visible dishes with their prices. Mention if there's any special feature or certification visible in the image (like 'Halal'). Keep the response neutral, clear, and structured."
    if vllm_info_model == 'accounts/fireworks/models/llama-v3p2-90b-vision-instruct':
        highlight_prompt = "Are there any items highlighted by borders in this image?" #\nRephrase and expand the question, and respond. Please analyze the main object
        # highlighted_vllm_response, highlighted_vllm_token_used = debug_vllm(image_base64, highlight_prompt, vllm_info_model)
        highlighted_vllm_response, highlighted_vllm_token_used = debug_vllm(image_base64, input_text, vllm_info_model)
        
    else:
        # vllm_response, vllm_token_used, vllm_cost = run_vllm_model(image_base64, trans_input_text, vllm_info_model)
        vllm_response, vllm_token_used =run_g1_vllm_concept(image_base64, text_4_vllm, vllm_info_model)
    vllm_token_used =  highlighted_vllm_token_used #raw_vllm_token_used +
    #--Step 3: Stitch for final output
    final_output = f"Highlighted item from VLLM:\n{highlighted_vllm_response}" #f"Raw output VLLM: \n{raw_vllm_response}" + \
    
    #--Output mapping 
    img_str = f'<img src="data:image/jpeg;base64,{image_base64}" alt="user upload image" />' 

    #--Compute cost 
    vllm_cost = compute_cost(vllm_token_used, (0.9 /1e6), 15125.90)
    total_cost = vllm_cost
    str_cost = 'Rp. %.6f' % total_cost
    #--translate content to indonesia
    stop = timeit.default_timer()
    time_process = "{:0,.2f}".format(stop - start) 
    # logger.info(f'Time process: {stop - start}')
    return [(img_str, final_output)], final_output, vllm_token_used, vllm_token_used, str_cost, time_process
def encode_image(image_path): 
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# text_4_vllm+='\nRephrase and expand the question, and respond.'
        # vllm_response, vllm_history_to_llm, vllm_token_used = run_vllm_model_single(image_base64, text_4_vllm, vllm_info_model)
        #--Step 1: Extract data from image
        # raw_vllm_response, raw_vllm_token_used = debug_vllm(image_base64, text_4_vllm, vllm_info_model)
        #--Step 2: Check any highlighted by user
        # highlight_prompt = "Are there any highlighted items in this image?NOTES: The highlighted items is typically represented by borders\nRephrase and expand the question, and respond."
#--Classify as regular or promotional 
        # clasify_prompt ="Analyze the menu shown in the image. Provide a detailed explanation of the menu items, focusing on whether it is a promotional or regular menu. Look for clues like event names, special pricing, free items, or tags like 'new' or 'limited-time' that might indicate a promotion. Mention the dish names, prices, and any unique features such as certifications (e.g., Halal). Keep the tone informative, neutral, and structured"
        # clasify_vllm_response, clasify_vllm_token_used = debug_vllm(image_base64, clasify_prompt, vllm_info_model)
        # logger.info(clasify_vllm_response)  
    