import json
import os
import sys
sys.path.append('/mnt/hdd1/jano/VisLang/demo_apps/')
import torch
from loguru import logger
from PIL import Image
from deepseek_vl.utils.conversation import SeparatorStyle
from deepseek_vl.serve.inference import (
    convert_conversation_to_prompts,
    deepseek_generate,
    load_model,
)

from utils.langfuse_deps import caption_lfprompt, summary_lfprompt, LANGFUSE_CAPTION_PROMPT_VERSION, LANGFUSE_SUMMARY_PROMPT_VERSION
from deepseekVL_local.deepseek_vl_config import all_dataset_path, save_dir, today, template_text, model_path, max_context_length_tokens, history
from deepseekvl_model import load_deepseek_vl_model
from deepseek_util import generate_prompt_with_history, strip_stop_words
from models import llm
llm_info_model="accounts/fireworks/models/llama-v3p1-8b-instruct"
tokenizer, vl_gpt, vl_chat_processor = load_deepseek_vl_model(model_path)
def prepare_prompt(img_path):
    image = Image.open(img_path).convert("RGB")
    conversation = generate_prompt_with_history(
        caption_lfprompt[0]['content'],
        template_text,
        image,
        history,
        vl_chat_processor,
        tokenizer,
        max_length=max_context_length_tokens,
    )
    prompts = convert_conversation_to_prompts(conversation)
    stop_words = conversation.stop_str
    return conversation, prompts, stop_words
def run_predict(dataset_path):
    data = []
    for img_name in sorted(os.listdir(dataset_path)):
        img_path = os.path.join(dataset_path, img_name)
        
        full_response = ""
        conversation, prompts, stop_words = prepare_prompt(img_path)
        with torch.no_grad():
            for x in deepseek_generate(
                prompts=prompts,
                vl_gpt=vl_gpt,
                vl_chat_processor=vl_chat_processor,
                tokenizer=tokenizer,
                stop_words=stop_words,
                max_length=max_context_length_tokens,
                temperature=0,
                repetition_penalty=1.,
                top_p=1,
            ):
                full_response += x
                response = strip_stop_words(full_response, stop_words)
                conversation.update_last_message(response)
        msg = llm.generate_summary_prompt(summary_lfprompt[0]['content'], summary_lfprompt[1]['content'], response)
        pred_summary, llm_pred_tokens = llm.llm_predict(msg, llm_info_model)
        data.append({'img_path': img_path, 'caption' : response, 'summary':pred_summary})
        logger.success(response)
        logger.info("flushed result to gradio")
        torch.cuda.empty_cache()
    return data


for dataset_name in os.listdir(all_dataset_path):
    dataset_path = os.path.join(all_dataset_path, dataset_name)
    pred_data = run_predict(dataset_path)
    save_path = os.path.join(save_dir, f"deep_seek{today}_{dataset_name}.json")
    with open(save_path, 'w') as f:
        json.dump(pred_data, f)


