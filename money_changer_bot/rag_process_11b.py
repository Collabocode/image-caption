import os 
import base64
# from langfuse.openai import OpenAI
from base_prompt import system_prompt, user_prompt, system_prompt_v2, system_prompt_menu_v2, \
    system_prompt_menu_analyze, system_prompt_menu_extract, system_prompt_currency_analyze, system_prompt_currency_extract,\
    user_prompt_extraction, user_prompt_analyze
from loguru import logger
from models.vllm_model import run_vllm
from models.embeddings import get_embedding
import json
# client = OpenAI(
#     base_url="https://api.fireworks.ai/inference/v1",
#     api_key="iOADKSeF8E4ImIuj5Yg2256DCDmaAJ5j5Uj5AUVZGn2Xr7qU"
#     )
# emb_client = OpenAI(
#             base_url="https://api.runpod.ai/v2/fe76dm7srdlmgw/openai/v1",
#             api_key="JEKVENJ8EGRPZ7S34DCXJ1AP6UB8G780TYYQ3CR5",)
# vllm_model_id="accounts/fireworks/models/llama-v3p2-90b-vision-instruct"
vllm_model_id="accounts/fireworks/models/llama-v3p2-11b-vision-instruct"
emb_model_id = "intfloat/multilingual-e5-large"
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
#--Get images directory

# imgs_dir="data/menu/references/images/"
# save_dir ='data/menu/references/'
# save_name = 'hokben'
imgs_dir="data/currencies/references/images/"
save_dir ='data/currencies/references/'
save_name = 'refcurrencies_11b_v2'
list_data = []
#--Check if previous data exists
try:
    f= open(os.path.join(save_dir, save_name+'.json'))
    list_data = json.load(f)
    f.close()
except:
    list_data = []

for img_name in sorted(os.listdir(imgs_dir)):
    is_done = False
    for check_data in list_data:
        if check_data['img_name'] == img_name:
            is_done= True
            logger.warning(f"{img_name} has been extracted")
            break
    if is_done:
        continue
    
    img_path = os.path.join(imgs_dir, img_name)
    image_base64 = encode_image(img_path)
    #--Output caption analyze
    out_text_analyze = run_vllm(image_base64, system_prompt_currency_analyze, user_prompt_analyze, vllm_model_id)
    logger.info(out_text_analyze)
    #--Step 2: Extract embedding
    out_emb_analyze = get_embedding(out_text_analyze, emb_model_id)

    #--Output caption extract
    out_text_extract = run_vllm(image_base64, system_prompt_currency_extract, user_prompt_extraction, vllm_model_id)
    logger.info(out_text_extract)
    #--Step 2: Extract embedding
    out_emb_extract = get_embedding(out_text_extract, emb_model_id)

    #--Step 3: Save in json format
    # list_data.append({'img_name':img_name, 'caption': out_text, 'text_emb': out_emb})
    list_data.append({'img_name':img_name, 'caption_analyze': out_text_analyze, 'text_emb_analyze': out_emb_analyze,
                      'caption_extact': out_text_extract, 'text_emb_extract': out_emb_extract,})
    
    logger.success(img_name)

#--Last save
with open(os.path.join(save_dir, save_name + '.json'), 'w') as f:
    json.dump(list_data, f, indent=4)