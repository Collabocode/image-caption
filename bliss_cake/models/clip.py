import torch
import torch.nn as nn
import numpy as np
import clip
from PIL import Image
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/16", device=device)
loss = nn.CosineSimilarity(dim=1, eps=1e-6)
def list_to_tensor(src_emb, to_tensor=True):
    emb_np = np.array(src_emb)
    emb_np = emb_np[np.newaxis, :]#-- Adding batch in front
    if to_tensor:
       emb = torch.from_numpy(emb_np)
    else:
        emb = emb_np 
    return emb
@torch.no_grad()
def compare_cosine_image(feat1, feat2):
    return loss(feat1, feat2)
@torch.no_grad()
def run_clip(src_img, ref_img):
    src_img = preprocess(src_img).unsqueeze(0).to(device)
    ref_img = preprocess(ref_img).unsqueeze(0).to(device)
    
    src_feat = model.encode_image(src_img)
    print(src_feat.data.detach().cpu().numpy().tolist()[0][:256])
    
    ref_feat = model.encode_image(ref_img)
    score = compare_cosine_image(src_feat, ref_feat)

    return score.data.detach().cpu().numpy()
@torch.no_grad()
def compare_image_embedding(src_emb, ref_emb):
    src_emb = list_to_tensor(src_emb)
    ref_emb = list_to_tensor(ref_emb)
    score = compare_cosine_image(src_emb, ref_emb)
    return score.data.detach().cpu().numpy()
@torch.no_grad()
def get_img_embedding(img):
    img = Image.open(img)
    img = preprocess(img).unsqueeze(0).to(device)
    feat = model.encode_image(img)
    #--return as list
    return feat.data.detach().cpu().numpy().tolist()[0]