import torch
import clip
from PIL import Image
from torch import nn
from loguru import logger
loss = nn.CosineSimilarity(dim=1, eps=1e-6)

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/16", device=device)

# image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)
# text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)
def compare_cosine_image(feat1, feat2):
    return loss(feat1, feat2)

ref_imgs = [{'img_path': 'references_images/menu_main_hokben.png', 'details': 'regular menu'}, 
            {'img_path': 'references_images/premium_set.png', 'details': 'premium set menu'},
            {'img_path': 'references_images/teriyaki_day.png', 'details': 'teriyaki'},
            {'img_path': 'references_images/bento_ramadan.png', 'details': 'Bento ramadhan'},
            ]
scores = []
for ref_img in ref_imgs:
    img_name = ref_img['img_path']
    # im1 = preprocess(Image.open("references_images/teriyaki_day_from_phone.jpg")).unsqueeze(0).to(device)
    # im1 = preprocess(Image.open("references_images/menu_main_hokben_pilih_curry_yaki_takoyaki_original.png")).unsqueeze(0).to(device)
    im1 = preprocess(Image.open("references_images/bento_ramadan_pilih_1.png")).unsqueeze(0).to(device)
    im2 = preprocess(Image.open(ref_img['img_path'])).unsqueeze(0).to(device)
    with torch.no_grad():
        feat1 = model.encode_image(im1)
        feat2 = model.encode_image(im2)
    score = compare_cosine_image(feat1, feat2)
    scores.append(score)
    logger.success(f'{img_name}, Score similarity: {score}')
    pass
# with torch.no_grad():
#     image_features = model.encode_image(image)
    