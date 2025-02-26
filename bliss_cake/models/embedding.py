from langfuse.openai import OpenAI
import torch
import torch.nn as nn
import numpy as np
julius ="https://api.runpod.ai/v2/fe76dm7srdlmgw/openai/v1"
julius_api_key ="JEKVENJ8EGRPZ7S34DCXJ1AP6UB8G780TYYQ3CR5"
ben = ""
# http://13.229.129.103:30001/embeddings
# client = OpenAI(
#     base_url="http://18.142.231.41:30001",
#     api_key="KYMpVRtMjd3MkehfeMJTn2BHcpcWTH",)
loss = nn.CosineSimilarity(dim=1, eps=1e-6)
def list_to_tensor(src_emb, to_tensor=True):
    emb_np = np.array(src_emb)
    emb_np = emb_np[np.newaxis, :]#-- Adding batch in front
    if to_tensor:
       emb = torch.from_numpy(emb_np)
    else:
        emb = emb_np 
    return emb
def compare_cosine(feat1, feat2):
    return loss(feat1, feat2)
def get_embedding(text, model_name="intfloat/multilingual-e5-large"):
    # text = text.replace("\n", " ")
    emb = client.embeddings.create(input = [text], model=model_name).data[0].embedding #--List
    return emb
# def compute_caption_similarity(src_caption, ref_caption):
#     src_emb =get_embedding(src_caption)
#     src_emb = list_to_tensor(src_emb)
#     ref_emb =get_embedding(ref_caption)
#     ref_emb = list_to_tensor(ref_emb)
#     score = compare_cosine(src_emb, ref_emb)
#     return score.detach().cpu().numpy()
def compute_caption_similarity(src_emb, ref_emb):
    # src_emb =get_embedding(src_caption)
    src_emb = list_to_tensor(src_emb)
    # ref_emb =get_embedding(ref_caption)
    ref_emb = list_to_tensor(ref_emb)
    score = compare_cosine(src_emb, ref_emb)
    return score.detach().cpu().numpy()