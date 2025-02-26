from datetime import date
import os
all_dataset_path = '/mnt/hdd1/jano/VisLang/datasets'
save_dir = "/mnt/hdd1/jano/VisLang/results/" + f"cap_deepseek_{1}_sum_{1}"

today = date.today()
# print("Today date is: ", today)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
model_path = '/mnt/hdd1/jano/VisLang/deepseek-vl-7b-chat/'
max_context_length_tokens=1024
history = []
template_text = "Interpret the scene, objects, actions, or elements within an image to provide a descriptive summary"
