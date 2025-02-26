import json
import os
from datetime import date

import llm, vllm
from loguru import logger
from img_util import encode_image
from langfuse_deps import caption_lfprompt, summary_lfprompt, LANGFUSE_CAPTION_PROMPT_VERSION, LANGFUSE_SUMMARY_PROMPT_VERSION
vllm_info_model="accounts/fireworks/models/llama-v3p2-11b-vision-instruct"
llm_info_model="accounts/fireworks/models/llama-v3p1-8b-instruct"
#--Always read on dataset
all_dataset_path = '/mnt/hdd1/jano/VisLang/datasets'
save_dir = "results/" + f"cap_{LANGFUSE_CAPTION_PROMPT_VERSION}_sum_{LANGFUSE_SUMMARY_PROMPT_VERSION}"

today = date.today()
# print("Today date is: ", today)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
def run_predict(dataset_path):
    data = []
    for img_name in sorted(os.listdir(dataset_path)):
        img_path = os.path.join(dataset_path, img_name)
        src_base64 = encode_image(img_path)
        msg = vllm.generate_caption_prompt(src_base64, caption_lfprompt[0]['content'], caption_lfprompt[1]['content'])
        pred_captions, vllm_pred_tokens = vllm.vllm_predict(msg, vllm_info_model)
        #--Run for summary
        msg = llm.generate_summary_prompt(summary_lfprompt[0]['content'], summary_lfprompt[1]['content'], pred_captions)
        pred_summary, llm_pred_tokens = vllm.vllm_predict(msg, llm_info_model)
        data.append({'img_path': img_path, 'caption' : pred_captions, 'summary':pred_summary})
        logger.success(pred_summary)
    return data
for dataset_name in os.listdir(all_dataset_path):
    dataset_path = os.path.join(all_dataset_path, dataset_name)
    pred_data = run_predict(dataset_path)
    save_path = os.path.join(save_dir, f"{today}_{dataset_name}.json")
    with open(save_path, 'w') as f:
        json.dump(pred_data, f)