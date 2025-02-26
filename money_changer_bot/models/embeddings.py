from langfuse.openai import OpenAI
julius ="https://api.runpod.ai/v2/fe76dm7srdlmgw/openai/v1"
julius_api_key ="JEKVENJ8EGRPZ7S34DCXJ1AP6UB8G780TYYQ3CR5"
ben = ""
client = OpenAI(
    base_url="http://13.229.129.103:30001",
    api_key="KYMpVRtMjd3MkehfeMJTn2BHcpcWTH",)
def get_embedding(text, model_name):
    # text = text.replace("\n", " ")
    emb = client.embeddings.create(input = [text], model=model_name).data[0].embedding #--List
    return emb