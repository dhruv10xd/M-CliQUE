import argparse
import os
import requests
import uuid
import json
from io import BytesIO
from PIL import Image
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig, BitsAndBytesConfig, TextStreamer
from llava.model import LlavaLlamaForCausalLM
from llava.conversation import conv_templates, SeparatorStyle
from llava.utils import disable_torch_init
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria, get_model_name_from_path, process_images
from llava.model.builder import load_pretrained_model

def is_valid_image(image_file):
    # Check if the input is a URL or a file path
    if image_file.startswith(('http://', 'https://')):
        try:
            response = requests.head(image_file, allow_redirects=True)
            # Check if the URL exists and returns a valid status code
            return response.status_code == 200
        except requests.RequestException as e:
            print(f"Invalid URL: {e}")
            return False
    else:
        # Check if the file path exists and is an image file
        if os.path.exists(image_file):
            return True
        else:
            print("File does not exist.")
            return False

def load_model():
    model_path = "Dhruv10XD/llava-v1.5-7b-bs"
    model_name = get_model_name_from_path(model_path)
    model_base = None
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base, model_name)
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

    model_name = get_model_name_from_path("checkpoints/baselines/llava-v1.5-7b")
    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
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
            # image_sizes=[image_size],
            do_sample=True,
            temperature=0.5,
            max_new_tokens=1024,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
            streamer=streamer
        )
    valid_output_ids = [token_id for token_id in output_ids[0] if token_id < tokenizer.vocab_size]
    outputs = tokenizer.decode(valid_output_ids).strip()

    # outputs = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    # conv.messages[-1][-1] = outputs
    # output = outputs.rsplit('</s>', 1)[0]
    # output = output[4:]

    return outputs

def main():
    model, tokenizer, image_processor = load_model()
    while True:
        image_file = input("Enter the path to the image or URL of the image (or type 'exit' to quit): ")
        if image_file.lower() == 'exit':
            break
        elif is_valid_image(image_file):
          choice = input("Is your query in any language other than Hindi/Hinglish/English? (yes/no): ").strip().lower()
          if choice == 'yes':
            choice2 = input("Is your query Codemixed? (yes/no): ").strip().lower()
            if choice2 == 'yes':
              # Codemixed translation through llama3-8b
              print("Please run codemixed_translate.py first! ")
              break
            elif choice2 == 'no': 
              # Multlilingual translation through indictrans2-1b
              print("Please run local_translate.py first! ")
              break
            else:
              print("Please enter a valid choice (either 'yes' or 'no')")
              continue
          elif choice == 'no':
              # No need for translation
              try:
                  text_query = input("Enter the text query: ")
                  print("\nOutput:")
                  output = multi_modal_pipeline(model, tokenizer, image_processor, image_file, text_query)
              except Exception as e:
                  print(" ")
          else:
            print("Please enter a valid choice (either 'yes' or 'no')")
        else:
          print("Invalid image file or URL. Please try again.")

if __name__ == '__main__':
    main()
