from langfuse import Langfuse
from langfuse.decorators import langfuse_context

LANGFUSE_SECRET_KEY="sk-lf-d58a0b73-a57a-4d8e-8d1b-64c2f731b7e2"
LANGFUSE_PUBLIC_KEY="pk-lf-97eb1b86-02c8-4600-9417-222b69963448"
LANGFUSE_HOST="https://llm-tracer.internal.collabo.dev"

langfuse_context.configure(
    secret_key=LANGFUSE_SECRET_KEY,
    public_key=LANGFUSE_PUBLIC_KEY,
    host=LANGFUSE_HOST,
)
lf = Langfuse(
    secret_key=LANGFUSE_SECRET_KEY,
    public_key=LANGFUSE_PUBLIC_KEY,
    host=LANGFUSE_HOST,
)
lf.auth_check()
LANGFUSE_CAPTION_PROMPT_KEY="caption_prompt_0.0.0"
LANGFUSE_CAPTION_PROMPT_VERSION="4"
LANGFUSE_CAPTION_PROMPT_FALLBACK="{}"
#--Get caption prompt from langfuse1
caption_lfprompt = lf.get_prompt(
        LANGFUSE_CAPTION_PROMPT_KEY,
        version=LANGFUSE_CAPTION_PROMPT_VERSION,
        fallback=LANGFUSE_CAPTION_PROMPT_FALLBACK,
        type="chat",
    ).prompt

LANGFUSE_SUMMARY_PROMPT_KEY="summary_caption_prompt_0.0.0"  
LANGFUSE_SUMMARY_PROMPT_VERSION="2"
LANGFUSE_SUMMARY_PROMPT_FALLBACK="{}"

summary_lfprompt = lf.get_prompt(
        LANGFUSE_SUMMARY_PROMPT_KEY,
        version=LANGFUSE_SUMMARY_PROMPT_VERSION,
        fallback=LANGFUSE_SUMMARY_PROMPT_FALLBACK,
        type="chat",
    ).prompt 

LANGFUSE_CLASSIFY_PROMPT_KEY="classification"  
LANGFUSE_CLASSIFY_PROMPT_VERSION="4"
LANGFUSE_CLASSIFY_PROMPT_FALLBACK="{}"

classify_lfprompt = lf.get_prompt(
        LANGFUSE_CLASSIFY_PROMPT_KEY,
        version=LANGFUSE_CLASSIFY_PROMPT_VERSION,
        fallback=LANGFUSE_CLASSIFY_PROMPT_FALLBACK,
        type="chat",
    ).prompt 

LANGFUSE_CLASSIFY_IMAGE_PROMPT_KEY="classify_image"  
LANGFUSE_CLASSIFY_IMAGE_PROMPT_VERSION="2"
LANGFUSE_CLASSIFY_IMAGE_PROMPT_FALLBACK="{}"

classify_image_lfprompt = lf.get_prompt(
        LANGFUSE_CLASSIFY_IMAGE_PROMPT_KEY,
        version=LANGFUSE_CLASSIFY_IMAGE_PROMPT_VERSION,
        fallback=LANGFUSE_CLASSIFY_IMAGE_PROMPT_FALLBACK,
        type="chat",
    ).prompt 