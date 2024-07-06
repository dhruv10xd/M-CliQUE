import argparse
import os
import requests
import uuid
import json
from io import BytesIO
from PIL import Image
import torch
from transformers import AutoTokenizer, AutoConfig, BitsAndBytesConfig, TextStreamer
from llava.model import LlavaLlamaForCausalLM
from llava.conversation import conv_templates, SeparatorStyle
from llava.utils import disable_torch_init
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria, get_model_name_from_path, process_images
from llava.model.builder import load_pretrained_model

# Add your Azure Translator key and endpoint
translator_key = "" # Enter your own BING Translator API key here
translator_endpoint = "https://api.cognitive.microsofttranslator.com"
translator_location = "" # Enter the region your BING Translator API key is active for
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():

    # Change the model path relative to your environment
    model_path = "checkpoints/finetuned/llava-v1.5-7b"
    model_name = get_model_name_from_path(model_path)
    model_base = None
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base, model_name, device=device)
    return model, tokenizer, image_processor

def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

def multi_modal_pipeline(model, tokenizer, image_processor, image_file, prompt):
    image = load_image(image_file)
    disable_torch_init()

    # Change the model path relative to your environment
    model_name = get_model_name_from_path("checkpoints/finetuned/llava-v1.5-7b")
    
    conv_mode = "llava_v0"

    conv = conv_templates[conv_mode].copy()
    roles = conv.roles

    image_tensor = process_images([image], image_processor, model.config)
    image_size = image.size
    if type(image_tensor) is list:
        image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
    else:
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)

    formatted_prompt = f"Summarize this hinglish text as a detailed English question: '{prompt}'. Question should be about 40-50 words and start with 'What'/'Why'/'Can'/'Is'/'Does'/'Should', should NOT have first person address (example I/my), followed by a line starting with 'The image here shows' depicting visual cue from the image."
    inp = f"{roles[0]}: {formatted_prompt}"
    inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    raw_prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(raw_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=[image_size],
            do_sample=True,
            temperature=0.5,
            max_new_tokens=1024,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
            streamer=streamer
        )

    outputs = tokenizer.decode(output_ids[0]).strip()
    conv.messages[-1][-1] = outputs
    output = outputs.rsplit('</s>', 1)[0]
    output = output[4:]

    return output

def translate_text(text):
    path = '/translate'
    constructed_url = translator_endpoint + path
    params = {
        'api-version': '3.0',
        'to': ['en']
    }
    headers = {
        'Ocp-Apim-Subscription-Key': translator_key,
        'Ocp-Apim-Subscription-Region': translator_location,
        'Content-type': 'application/json',
        'X-ClientTraceId': str(uuid.uuid4())
    }
    body = [{'text': text}]
    request = requests.post(constructed_url, params=params, headers=headers, json=body)
    response = request.json()
    translated_text = response[0]['translations'][0]['text']
    return translated_text

def main():
    model, tokenizer, image_processor = load_model()
    while True:
        image_file = input("Enter the path to the image or URL of the image (or type 'exit' to quit): ")
        if image_file.lower() == 'exit':
            break

        choice = input("Do you want to translate the text? (yes/no): ").strip().lower()
        if choice == 'yes':
            text_query = input("Enter the text query to be translated: ")
            try:
                translated_text = translate_text(text_query)
                print("Translated Text:", translated_text)
                output = multi_modal_pipeline(model, tokenizer, image_processor, image_file, translated_text)
                print(f"Output: {output}")
            except Exception as e:
                print("Translation Error:", e)
        else:
            text_query = input("Enter the text query: ")
            try:
                output = multi_modal_pipeline(model, tokenizer, image_processor, image_file, text_query)
                print(f"Output: {output}")
            except Exception as e:
                print("Error:", e)

if __name__ == '__main__':
    main()
