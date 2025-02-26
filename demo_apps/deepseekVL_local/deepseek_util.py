import torch
from deepseek_vl.utils.conversation import SeparatorStyle
from loguru import logger
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
    system_message, 
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
    conversation.system_message = system_message
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