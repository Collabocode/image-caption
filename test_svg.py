# from img_util import encode_image
# import cairosvg
# import base64
# import requests
# import urllib.request

# from loguru import logger
# from langfuse_deps import caption_lfprompt
# import vllm
# vllm_info_model="accounts/fireworks/models/llama-v3p2-11b-vision-instruct"
# llm_model_id="accounts/fireworks/models/llama-v3p1-8b-instruct"

# path = "svg_files/adjustment-svgrepo-com.svg"
# # start = timeit.timeit()
# png_byte = cairosvg.svg2png(url=path)
# png_base64 = base64.b64encode(png_byte).decode('utf-8')
# # end = timeit.timeit()   
# # logger.success(end - start)
# msg = vllm.generate_caption_prompt(png_base64, caption_lfprompt[0]['content'], caption_lfprompt[1]['content'])
# pred_captions, vllm_pred_tokens = vllm.vllm_predict(msg, vllm_info_model)
# logger.success(pred_captions)


# # import pyvips

# # # image = pyvips.Image.new_from_file(path, dpi=300)
# # # buffer = image.write_to_buffer(".png")
# # # png_base64 = base64.b64encode(buffer).decode('utf-8')
# # # msg = vllm.generate_caption_prompt(png_base64, caption_lfprompt[0]['content'], caption_lfprompt[1]['content'])
# # img_url= "https://cc-playground-testing-jess-9c632.storage.googleapis.com/null/Image%20from%20iOS%20%281%29-d92b9bb5-4b52-4f82-9c56-19ff093fd482.jpg?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=GOOG1EMQNI4RQO62ECYFSOE4PAXG7E7IW2S7YCUUFUKLK6IBUIF67F5GPJ3QV%2F20241220%2Fasia-southeast-1%2Fs3%2Faws4_request&X-Amz-Date=20241220T111440Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&x-id=GetObject&X-Amz-Signature=ae78f9eb82f2d11ad466b41042d6a91b8b8e28f9d5cbc60cd31bf6d8e15450d3"
# # buffer = urllib.request.urlopen(img_url)
# # # req = requests.get(img_url)
# # resp: pyvips.Image = pyvips.Image.new_from_buffer(buffer.read(), "") # type: ignore
# # resp_buffer = resp.write_to_buffer(".png")

# # png_base64 = base64.b64encode(resp_buffer).decode("utf-8")
# # msg = vllm.generate_caption_prompt(png_base64, caption_lfprompt[0]['content'], caption_lfprompt[1]['content'])
# # pred_captions, vllm_pred_tokens = vllm.vllm_predict(msg, vllm_info_model)
# # logger.success(pred_captions)
# # pass


# import boto3
# import os

# s3 = boto3.client(
#     's3',
#     aws_access_key_id="GOOG1EMQNI4RQO62ECYFSOE4PAXG7E7IW2S7YCUUFUKLK6IBUIF67F5GPJ3QV",
#     aws_secret_access_key="9K3cpAIL0NAl7SEEuYhyN2YuDj5eR7kWWMpssoXR",
#     region_name='asia-southeast-1',
#     endpoint_url='https://storage.googleapis.com',
# )
# #--TODO: After mikeben finish setup, try again.
# def get_convert_to_byte(img_url: str):
#     buffer = urllib.request.urlopen(img_url)
#     resp: pyvips.Image = pyvips.Image.new_from_buffer(buffer.read(), "")  # type: ignore
#     resp_buffer = resp.write_to_buffer(".png")
    
#     return resp_buffer
# import uuid
# import requests
# import base64
# import hashlib
# from datetime import datetime, timezone
# Img_ids_kafka= kafka_msg.data["image_ids"]
#             print(f"\n\n KAFKA: {Img_ids_kafka} \n\n")
#             test = (kafka_msg.data["image_ids"] is None)
#             print(f"\n\n IMG_IDS {test} \n\n")
#             test2 = (kafka_msg.data["image_ids"]==[])
#             print(f"\n\n IMG_IDS_LIST {test2} \n\n")
# def run_multimodal_tracer(content_bytes, field, content_type ="image/png"):
#     trace_id = str(uuid.uuid4())
#     content_sha256 = base64.b64encode(hashlib.sha256(content_bytes).digest()).decode()
#     content_length = len(content_bytes)
#     create_upload_url_body = {
#         "traceId": trace_id,
#         "contentType": content_type,
#         "contentLength": content_length,
#         "sha256Hash": content_sha256,
#         "field": field,
#     }
#     print(f"\n\n {create_upload_url_body} \n\n")
#     upload_url_request = requests.post(
#         f"{config.LANGFUSE_HOST}/api/public/media",
#         auth=(config.LANGFUSE_PUBLIC_KEY or "", config.LANGFUSE_SECRET_KEY or ""),
#         headers={"Content-Type": "application/json"},
#         json=create_upload_url_body,
#     )
#     upload_url_response = upload_url_request.json()
#     print(f"\n\n {upload_url_response} \n\n")
#     if (upload_url_response["mediaId"] is not None and 
#         upload_url_response["uploadUrl"] is not None ):
#             upload_response = requests.put(
#             upload_url_response["uploadUrl"],
#             headers={ "Content-Type": content_type,
#                 "x-amz-checksum-sha256": content_sha256,
#             },
#             data=content_bytes,)
#     if upload_response is not None:
#         requests.patch(
#             f"{config.LANGFUSE_HOST}/api/public/media/{upload_url_response['mediaId']}",
#             auth=(config.LANGFUSE_PUBLIC_KEY or "", config.LANGFUSE_SECRET_KEY or ""),
#             headers={"Content-Type": "application/json"},
#             json={
#                 "uploadedAt": datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%fZ'), # ISO 8601
#                 "uploadHttpStatus": upload_response.status_code,
#                 "uploadHttpError": upload_response.text if upload_response.status_code != 200 else None,
#             },
#         )
    
    
# # Map common extensions to MIME types (this is a simplified example)
# # mime_types = {
# #     '.txt': 'text/plain',
# #     '.jpg': 'image/jpeg',
# #     '.jpeg': 'image/jpeg',
# #     '.png': 'image/png',
# #     '.gif': 'image/gif',
# #     '.pdf': 'application/pdf',
# #     '.doc': 'application/msword',
# #     '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
# #     '.xls': 'application/vnd.ms-excel',
# #     '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
# #     # Add more mappings as needed
# # }

# # # Infer the MIME type from the extension
# # inferred_content_type = mime_types.get(file_extension, 'Unknown')
# # response = s3.head_object(Bucket="cc-playground-testing-jess-9c632", Key="c2eaa19f-b2fb-49d4-847e-e91001969ee1")
# # Extract the Content-Type
# # content_type = response.get('ContentType', 'Unknown')
# print("test")


