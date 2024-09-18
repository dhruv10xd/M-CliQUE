import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)
from IndicTransToolkit import IndicProcessor

# Define model and tokenizer
model_name = "ai4bharat/indictrans2-indic-en-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True)

# Initialize IndicTrans processor for preprocessing and postprocessing
ip = IndicProcessor(inference=True)

# Ask for user input
input_sentence = input("Please enter the text to translate: ")

# Ask for language input
source_language = input("Please choose a language from below options and input: \nPunjabi:Pa \nMarathi:Mr \nTamil:Ta \nTelugu:Te \nKannada:Kn\n")

# Map user input to source language codes
if source_language == 'Pa':
    source_language = "pan_Guru"
elif source_language == 'Mr':
    source_language = "mar_Deva"
elif source_language == 'Ta':
    source_language = "tam_Taml"
elif source_language == 'Te':
    source_language = "tel_Telu"
elif source_language == 'Kn':
    source_language = "kan_Knda"
else:
    print("Please enter a valid option (Pa/Mr/Ta/Te/Kn).")
    exit()

# Target language is English
tgt_lang = "eng_Latn"

# Preprocess the input sentence using IndicProcessor
batch = ip.preprocess_batch(
    [input_sentence],  # Passing input as a list
    src_lang=source_language,
    tgt_lang=tgt_lang,
)

# Check for available device (CUDA or CPU)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Tokenize the input sentences and generate input encodings
inputs = tokenizer(
    batch,
    truncation=True,
    padding="longest",
    return_tensors="pt",
    return_attention_mask=True,
).to(DEVICE)

# Generate translations using the model
with torch.no_grad():
    generated_tokens = model.generate(
        **inputs,
        use_cache=True,
        min_length=0,
        max_length=512,
        num_beams=5,
        num_return_sequences=1,
    )

# Decode the generated tokens into text
with tokenizer.as_target_tokenizer():
    generated_text = tokenizer.batch_decode(
        generated_tokens.detach().cpu().tolist(),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

# Postprocess the translations, including entity replacement
translations = ip.postprocess_batch(generated_text, lang=tgt_lang)

# Output the translated text
print(f"\nTranslated text:\n{translations[0]}")
