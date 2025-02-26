from langfuse.openai import OpenAI
import json
gpt_system_prompt ="Understand the Task: Grasp the main objective, goals, requirements, constraints, and expected output. - Minimal Changes: If an existing prompt is provided, improve it only if it's simple. For complex prompts, enhance clarity and add missing elements without altering the original structure. - Reasoning Before Conclusions: Encourage reasoning steps before any conclusions are reached. ATTENTION! If the user provides examples where the reasoning happens afterward, REVERSE the order! NEVER START EXAMPLES WITH CONCLUSIONS! - Reasoning Order: Call out reasoning portions of the prompt and conclusion parts (specific fields by name). For each, determine the ORDER in which this is done, and whether it needs to be reversed. - Conclusion, classifications, or results should ALWAYS appear last. - Examples: Include high-quality examples if helpful, using placeholders [in brackets] for complex elements. - What kinds of examples may need to be included, how many, and whether they are complex enough to benefit from placeholders. - Clarity and Conciseness: Use clear, specific language. Avoid unnecessary instructions or bland statements. - Formatting: Use markdown features for readability. DO NOT USE ``` CODE BLOCKS UNLESS SPECIFICALLY REQUESTED. - Preserve User Content: If the input task or prompt includes extensive guidelines or examples, preserve them entirely, or as closely as possible. If they are vague, consider breaking down into sub-steps. Keep any details, guidelines, examples, variables, or placeholders provided by the user. - Constants: DO include constants in the prompt, as they are not susceptible to prompt injection. Such as guides, rubrics, and examples. - Output Format: Explicitly the most appropriate output format, in detail. This should include length and syntax (e.g. short sentence, paragraph, JSON, etc.) - For tasks outputting well-defined or structured data (classification, JSON, etc.) bias toward outputting a JSON. - JSON should never be wrapped in code blocks (```) unless explicitly requested. The final prompt you output should adhere to the following structure below. Do not include any additional commentary, only output the completed system prompt. SPECIFICALLY, do not include any additional messages at the start or end of the prompt. (e.g. no ""---"")."
# gpt_system_prompt ='You are an AI assistant that explains your reasoning step by step, incorporating dynamic Chain of Thought (CoT), reflection, and verbal reinforcement learning. Follow these instructions: 1. Enclose all thoughts within <thinking> tags, exploring multiple angles and approaches. 2. Break down the solution into clear steps, providing a title and content for each step. 3. After each step, decide if you need another step or if you\'re ready to give the final answer. 4. Continuously adjust your reasoning based on intermediate results and reflections, adapting your strategy as you progress. 5. Regularly evaluate your progress, being critical and honest about your reasoning process. 6. Assign a quality score between 0.0 and 1.0 to guide your approach: - 0.8+: Continue current approach - 0.5-0.7: Consider minor adjustments - Below 0.5: Seriously consider backtracking and trying a different approach 7. If unsure or if your score is low, backtrack and try a different approach, explaining your decision. 8. For mathematical problems, show all work explicitly using LaTeX for formal notation and provide detailed proofs. 9. Explore multiple solutions individually if possible, comparing approaches in your reflections. 10. Use your thoughts as a scratchpad, writing out all calculations and reasoning explicitly. 11. Use at least 5 methods to derive the answer and consider alternative viewpoints. 12. Be aware of your limitations as an AI and what you can and cannot do. After every 3 steps, perform a detailed self-reflection on your reasoning so far, considering potential biases and alternative viewpoints. Respond in JSON format with \'title\', \'content\', \'next_action\' (either \'continue\', \'reflect\', or \'final_answer\'), and \'confidence\' (a number between 0 and 1) keys. Example of a valid JSON response: ```json { "title": "Identifying Key Information", "content": "To begin solving this problem, we need to carefully examine the given information and identify the crucial elements that will guide our solution process. This involves...", "next_action": "continue", "confidence": 0.8 }``` Your goal is to demonstrate a thorough, adaptive, and self-reflective problem-solving process, emphasizing dynamic thinking and learning from your own reasoning.'
# gpt_system_prompt = "The AI must accurately interpret or generate images based on user-provided descriptions, focusing on the relevant details, context, and objectives (creative or technical). Process: Text Interpretation: Carefully analyze any text-based descriptions provided by the user to identify the key components of the image, such as objects, actions, relationships, styles, colors, and attributes. Context Awareness: Tailor the interpretation to match the context—whether it’s technical (e.g., vehicle detection, medical images) or creative (e.g., artwork, scenes). Ensure that the analysis or generation aligns with the user’s intent. Component Breakdown: Identify and break down the visual elements described by the user: Objects: Detect or generate recognizable objects (e.g., cars, trees, faces). Actions: Recognize or simulate interactions between elements (e.g., a person running, clouds moving). Attributes: Assess or generate attributes like color, texture, light, shape, and style. Artistic/Stylistic Context: If the image involves artistic elements, ensure the style (e.g., realism, surrealism, minimalism) and emotional tone (e.g., vibrant, melancholic) reflect user preferences. Image Generation: Use detailed descriptions to generate images. Ensure all key elements described by the user are included. Adapt to creative requests while adhering to rules regarding copyrighted material or recognizable private individuals. When generating, consider: Composition: Maintain a coherent arrangement of elements. Visual Coherence: Ensure lighting, colors, and shadows are realistic or suited to the intended style. Stylistic Integrity: Follow the visual style requested by the user (e.g., natural landscapes, futuristic cityscapes, minimalistic designs). Feedback and Adjustments: When interpreting or revising an image, provide detailed feedback regarding how well the image matches the user’s expectations. Identify areas for improvement or adjustment (e.g., changing the lighting, focusing on a particular object). Be proactive in suggesting changes based on user goals—e.g., enhancing clarity, improving balance, or altering mood with color and light. Ethical Considerations: Avoid generating or interpreting images in ways that violate ethical guidelines, including respecting privacy, avoiding inappropriate content, and adhering to copyright rules. In cases of ambiguity, seek clarification from the user to ensure alignment with their needs and expectations."
# gpt_system_prompt = "Image Description Analysis: When given a text description of an image, extract key details such as objects, people, backgrounds, colors, textures, actions, and any specific artistic elements (e.g., lighting, mood, style). Pay close attention to context and requirements provided by the user, such as technical tasks (e.g., vehicle detection, medical imaging) or creative/artistic specifications (e.g., surreal, minimalist). Component Breakdown: Identify individual elements in the image description: Objects: Recognize the primary and secondary objects (e.g., cars, animals, buildings). Attributes: Consider size, color, shape, texture, lighting, shadow, and spatial relationships between objects. Actions: Recognize dynamic elements (e.g., motion, interaction between objects) where applicable. If required, apply technical domain-specific knowledge (e.g., recognize vehicle types for object detection or patterns in medical imaging). Contextual Understanding: Use contextual knowledge to guide interpretation: for example, understanding technical descriptions, artistic styles, or thematic requirements (e.g., futuristic, classical, realistic). Consider the purpose of the task (e.g., a dehazing task would focus on enhancing image clarity, while creative work may emphasize mood or style). Matching to Known Visual Concepts: Relate the objects and attributes in the description to known visual patterns (e.g., how vehicles typically look, the texture of clouds, or the structure of a landscape). For creative prompts, consider general principles of art and aesthetics (e.g., balance, contrast, color theory) and incorporate those into feedback or image generation. Generating or Interpreting Images: When generating images, construct a detailed prompt based on the description, ensuring key components, attributes, and stylistic choices are incorporated. If interpreting an existing image, describe its components, patterns, and overall structure, offering insights on whether it matches the user’s description or intended style. Providing Feedback or Suggestions: When offering feedback on images, suggest refinements based on the initial description or request. Provide clear, actionable guidance on aspects like object placement, color scheme, clarity, mood, or alignment with technical goals. Ensure Accuracy and Relevance: Double-check that the generated or interpreted image aligns with the user's objectives. For technical tasks, ensure precision in identifying or enhancing relevant features. Maintain ethical standards by avoiding generation of inappropriate content, and respect privacy and copyright boundaries."
# gpt_assistant_prompt ="Helpful: Always aim to provide useful, informative, and accurate responses. Anticipate user needs, offering clarifications or additional context where necessary. Concise yet Thorough: Deliver detailed responses that address the core of the question without being overly verbose, but still provide enough information to be thorough. Engaging: Be conversational and friendly, adapting to the user’s tone. Whether casual or formal, try to match the communication style. Context-Aware: Keep track of the conversation's context and be mindful of prior information shared. Use this to improve the relevance of responses. Problem-Solving: Focus on understanding the user’s problem or request and work toward the most practical or effective solution. If you don’t know something, admit it and suggest alternatives like further research. Ethical and Safe: Ensure all responses align with ethical standards, avoiding inappropriate content or misinformation. Respect privacy and ensure safety."
client = OpenAI(
    base_url="https://api.fireworks.ai/inference/v1/",
    api_key="iOADKSeF8E4ImIuj5Yg2256DCDmaAJ5j5Uj5AUVZGn2Xr7qU"
    )
def debug_vllm_generate_KB(image_base64, text, system_prompt, vllm_info_model, assistant_prompt=None):
    #--Generate basic prompt
    msg = []
    msg.append({"role": "system", "content":f"{system_prompt}"})
    # msg.append({"role": "assistant", "content": f"{gpt_assistant_prompt}"})
    if assistant_prompt!= None:
        msg.append({"role": "assistant", "content": f"Here is the reference information: {assistant_prompt}"})
    user_prompt = [
        {
            "type": "text",
            "text": f'{text}',
        },
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{image_base64}"
            }
        }]
    msg.append({"role": "user", "content": user_prompt})
    
    vllm_response = client.chat.completions.create(
        model=vllm_info_model,
        messages=msg,   
        temperature=0,
        max_tokens=16384,
        top_p=1,
        presence_penalty=0,
        frequency_penalty=0,
        extra_body={'top_k':1, }
    )

    return vllm_response.choices[0].message.content, [{"role": "assistant", "content": vllm_response.choices[0].message.content}], vllm_response.usage.total_tokens
def debug_vllm(image_base64, text, vllm_info_model):
    #--Generate basic prompt
    msg = []
    msg.append({"role": "system", "content":f"{gpt_system_prompt}"})
    # msg.append({"role": "assistant", "content": f"{gpt_assistant_prompt}"})
    user_prompt = [
        {
            "type": "text",
            "text": f'{text}',
        },
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{image_base64}"
            }
        }]
    msg.append({"role": "user", "content": user_prompt})
    
    vllm_response = client.chat.completions.create(
        model=vllm_info_model,
        messages=msg,
        temperature=0,
        max_tokens=16384,
        top_p=1,
        presence_penalty=0,
        frequency_penalty=0,
        extra_body={'top_k':1, }
    )
    return vllm_response.choices[0].message.content, vllm_response.usage.total_tokens
# def generate_base_prompt(image_base64, system_prompt, text):
#     messages = []
#     prompt = [
#         {
#             "type": "text",
#             "text": f'{text}',
#         },
#         {
#             "type": "image_url",
#             "image_url": {
#                 "url": f"data:image/jpeg;base64,{image_base64}"
#             }
#         }]
#     if system_prompt is not None:
#         messages.append({"role": "system", "content":f"{system_prompt}"})    
#     messages.append({"role": "user", "content": prompt})
    
#     return messages

def generate_caption_prompt(image_base64, system_prompt, user_prompt):
    messages = []
    prompt = [
        {
            "type": "text",
            "text": f'{user_prompt}',
        },
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{image_base64}"
            }
        }]
    messages.append({"role": "system", "content":f"{system_prompt}"})    
    messages.append({"role": "user", "content": prompt})
    
    return messages

def vllm_predict(msg, vllm_info_model, temperature=0, is_json=False, max_tokens=1024, top_p=0.1):
    if is_json:
        vllm_response = client.chat.completions.create(
        model=vllm_info_model,
        messages=msg,
        response_format={ "type": "json_object" },
        temperature=temperature,
        max_tokens=16384,
        top_p=1,
    )
    else:
        vllm_response = client.chat.completions.create(
            model=vllm_info_model,
            messages=msg,
            temperature=temperature,
            max_tokens=16384,
            top_p=1,
        )
    return vllm_response.choices[0].message.content, vllm_response.usage.total_tokens
