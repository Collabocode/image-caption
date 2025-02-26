import json
import os
import torch
from datetime import date
from PIL import Image
from deepseek_vl.utils.conversation import SeparatorStyle
from deepseek_vl.serve.inference import (
    convert_conversation_to_prompts,
    deepseek_generate,
    load_model,
)
from loguru import logger
from deepseekvl_model import load_deepseek_vl_model
import llm
from langfuse_deps import caption_lfprompt, summary_lfprompt, LANGFUSE_CAPTION_PROMPT_VERSION, LANGFUSE_SUMMARY_PROMPT_VERSION
#--Always read on dataset
all_dataset_path = 'datasets'
save_dir = "results/" + f"cap_deepseek_{1}_sum_{1}"

today = date.today()
# print("Today date is: ", today)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
llm_info_model="accounts/fireworks/models/llama-v3p1-8b-instruct"
tokenizer, vl_gpt, vl_chat_processor = load_deepseek_vl_model('/mnt/hdd1/jano/VisLang/deepseek-vl-7b-chat/')
text = "Interpret the scene, objects, actions, or elements within an image to provide a descriptive summary"

max_context_length_tokens=1024
history = []
def strip_stop_words(x, stop_words):
    for w in stop_words:
        if w in x:
            return x[: x.index(w)].strip()
    return x.strip()
def get_prompt(conv) -> str:
    """Get the prompt for generation."""
    system_prompt = conv.system_template.format(system_message=conv.system_message)
    if conv.sep_style == SeparatorStyle.DeepSeek:
        seps = [conv.sep, conv.sep2]
        if system_prompt == "" or system_prompt is None:
            ret = ""
        else:
            ret = system_prompt + seps[0]
        for i, (role, message) in enumerate(conv.messages):
            if message:
                if type(message) is tuple:  # multimodal message
                    message, _ = message
                ret += role + ": " + message + seps[i % 2]
            else:
                ret += role + ":"
        return ret
    else:
        return conv.get_prompt
def generate_prompt_with_history(
    text, image, history, vl_chat_processor, tokenizer, max_length=2048
):
    """
    Generate a prompt with history for the deepseek application.

    Args:
        text (str): The text prompt.
        image (str): The image prompt.
        history (list): List of previous conversation messages.
        tokenizer: The tokenizer used for encoding the prompt.
        max_length (int): The maximum length of the prompt.

    Returns:
        tuple: A tuple containing the generated prompt, image list, conversation, and conversation copy. If the prompt could not be generated within the max_length limit, returns None.
    """

    sft_format = "deepseek"
    user_role_ind = 0
    bot_role_ind = 1

    # Initialize conversation
    conversation = vl_chat_processor.new_chat_template()
    conversation.system_message = caption_lfprompt[0]['content']
    if history:
        conversation.messages = history

    if image is not None:
        if "<image_placeholder>" not in text:
            text = (
                "<image_placeholder>" + "\n" + text
            )  # append the <image_placeholder> in a new line after the text prompt
        text = (text, image)

    conversation.append_message(conversation.roles[user_role_ind], text)
    conversation.append_message(conversation.roles[bot_role_ind], "")

    # Create a copy of the conversation to avoid history truncation in the UI
    conversation_copy = conversation.copy()
    logger.info("=" * 80)
    logger.info(get_prompt(conversation))

    rounds = len(conversation.messages) // 2

    for _ in range(rounds):
        current_prompt = get_prompt(conversation)
        current_prompt = (
            current_prompt.replace("</s>", "")
            if sft_format == "deepseek"
            else current_prompt
        )

        if torch.tensor(tokenizer.encode(current_prompt)).size(-1) <= max_length:
            return conversation_copy
    return None
def run_predict(dataset_path):
    data = []
    for img_name in sorted(os.listdir(dataset_path)):
        img_path = os.path.join(dataset_path, img_name)
        image = Image.open(img_path).convert("RGB")
        # try:
        
        conversation = generate_prompt_with_history(
            text,
            image,
            history,
            vl_chat_processor,
            tokenizer,
            max_length=max_context_length_tokens,
        )
        prompts = convert_conversation_to_prompts(conversation)
        stop_words = conversation.stop_str
        full_response = ""
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