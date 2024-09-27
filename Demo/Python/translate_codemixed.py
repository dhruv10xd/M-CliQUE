import accelerate
from transformers import AutoTokenizer, pipeline
import torch
import re

# Define the model and tokenizer
model = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model)

# Initialize the pipeline for text generation
generation_pipeline = pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)

# Function to apply the chat template and generate the output
def generate_translation(text):
    messages = [
      {"role": "system", "content": f"You are a linguistic expert whose task is to convert the given codemixed passages into corresponding English ones. 〈Labelled Example〉: Input: {{hi, mujhe ek skin disorder hai. Mujhe exactly pata nahi kya hai. Meri labia majus par skin ki kuch problem hai, wo peels hoti hai, redness hoti hai, itchiness hoti hai. Please doctor, dekhiye is vartaman condition ko. Relief milta hai jab main scratch karti hoon lekin jab khatam ho jata hai, toh dard hota hai. Mujhe ye kaise hua aur iski kya treatment hai? Meri age 26 hai, 5 feet 4 inches ki height aur 85 kg weight hai. Ye around 3 saal se hai}} Output (English): {{Hi, I have a skin disorder. I'm not exactly sure what it is. I have some skin problems on my labia majora; it peels, gets red, and is itchy. Please, doctor, look at this current condition. I feel relief when I scratch, but it hurts once I stop. How did this happen to me, and what is the treatment for it? I am 26 years old, 5 feet 4 inches tall, and weigh 85 kg. This has been happening for around 3 years.}} . Given a codemixed passage: {{{text}}}, convert it into the corresponding English passage as shown in the 〈Labelled Example〉 and output only the English passage."},
      {"role": "assistant", "content": " 〈Output〉: Here is the converted passage:" }
    ]

    # Generate the prompt using the chat template
    prompt = generation_pipeline.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Define the terminators for the output
    terminators = [
        generation_pipeline.tokenizer.eos_token_id,
        generation_pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    # Generate the output
    outputs = generation_pipeline(
        prompt,
        max_new_tokens=512,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.2,
        top_p=0.9,
    )

    generated_text = outputs[0]["generated_text"][len(prompt):]
    
    match = re.search(r'"(.*?)"', generated_text)
    if match:
        return match.group(1)
    else:
        return generated_text
    # Return the generated text, removing the prompt part


# Ask for user input
input_text = input("Please enter the codemixed query: ")


# Generate and print the English translation
translated_text = generate_translation(input_text)
print("\nTranslated query:\n", translated_text)
print("\nPlease enter this translated query into MMSUCCESS now")
