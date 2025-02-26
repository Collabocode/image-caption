import json
import os
from loguru import logger
import base64
from PIL import Image
# from models.vllm import run_vllm
from models import vllm
import textwrap

from models.clip import get_img_embedding
from models.embedding import get_embedding
from constants.prompts import predefined_system_prompt
emb_model_id = "intfloat/multilingual-e5-large"
vllm_model_id="accounts/fireworks/models/llama-v3p2-11b-vision-instruct"
save_name = 'bliss_cake_ref'
list_data = []
ref_imgs_path = "/mnt/hdd1/jano/VisLang/bliss_cake/data/reference_images/"
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
def get_input_prompt(image_base64, system_prompt, user_prompt):
    msg = vllm.generate_base_prompt(image_base64, system_prompt, textwrap.dedent(user_prompt))
    return msg


for img_name in sorted(os.listdir(ref_imgs_path)):
   
    img_path = os.path.join(ref_imgs_path, img_name)
    image_base64 = encode_image(img_path)
    msg = get_input_prompt(image_base64, predefined_system_prompt, "Please analyze.")
    pred_captions, pred_tokens = vllm.vllm_predict(msg, vllm_model_id)
    logger.info(pred_captions)
    #--Step 2: Extract caption embedding
    caption_emb = get_embedding(pred_captions, emb_model_id)
    #--Step 3: Extract image embedding
    img_emb = get_img_embedding(img_path)
    list_data.append({'img_name':img_name, 'captions': pred_captions, 'caption_emb': caption_emb, 'img_emb': img_emb})
    logger.success(f"{img_name}")
#--Last step: Save 
with open(save_name + '.json', 'w') as f:
    json.dump(list_data, f, indent=4)
    
