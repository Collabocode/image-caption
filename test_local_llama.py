import requests
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor

model_id = "/mnt/hdd1/jano/VisLang/Llama-3.2-11B-Vision"

model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_id)

url = "/mnt/hdd1/jano/VisLang/datasets_data/Cake/Baileys Chocolate.jpeg"
image = Image.open(url)

prompt = "<|image|><|begin_of_text|>What is this?"
inputs = processor(image, prompt, return_tensors="pt").to(model.device)

output = model.generate(**inputs, max_new_tokens=100)
print(processor.decode(output[0]))