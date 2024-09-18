# MMSUCCESS - Codemixed_Clinical_VQA
## Contents
- [Install](#Install)
- [Demo](#Demo)
  
## Install 
1. Clone the version 1.0 of LLaVA repository and navigate to LLaVA folder
```bash
git clone -b v1.0 https://github.com/camenduru/LLaVA
cd LLaVA
```
2. Clone this repository
```bash
git clone https://github.com/dhruv10xd/Codemixed_Clinical_VQA.git
```
3. Install pre-requisites
```shell
pip install -q Pillow requests xformers accelerate.
```
4. Clone and Install IndicTransTokenizer (optional)
```shell
pip install indic-nlp-library
```
```bash
git clone https://github.com/VarunGumma/IndicTransTokenizer
cd IndicTransTokenizer
```
```shell
!pip install --editable ./
cd ..
```

## Demo

### Colab Notebook
You can run this colab notebook which includes the installation and both the CLI Interface demo and Gradio demos.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ptsTwVjUCGIj-0cGXW2VGHnH0NZWnYXT?usp=sharing)


### CLI Interface
MMSUCCESS supports 7 languages - Hindi, English, Punjabi, Marathi, Tamil, Telugu, Kannada.

#### Multi-lingual Translation
If your query is multi-lingual and not codemixed, run this command
```shell
python Demo/Python/translate_multilingual.py
```
After running this command, you will be prompted to enter your query along with the language it is in. Then you will get translated query which you can enter in [Normal/LoRA Inference](#### Normal Model Inference)

#### Codemixed Translation
If your query is codemixed and is not in Hindi/English (i.e. any of the other 5 languages), please login into HuggingFace Hub first using your HF token, then run this command
```shell
python Demo/Python/translate_codemixed.py
```
After running this command, you will be prompted to enter your query, and LLaMA-3-8B will automatically detect the language present in your query. Then you will get translated query which you can enter in [Normal/LoRA Inference](#### Normal Model Inference)


#### Normal Model Inference
For running normal models, please run the command below:
```shell
python Demo/Python/mmsuccess_demo_normal.py
```
After running this command, the model will load and then you will be asked to input the path to the image. Following this, you will be prompted to input your query and the model will output a medically precise English Question summarizing your query. 
You can do this for multiple queries. To exit the demo, please enter 'exit' when asked to input path to the image. 

#### LoRA Model Inference
For running LoRA models, please run the command below:
```shell
python Demo/Python/mmsuccess_demo_lora.py
```

### Gradio Web Interface
For running normal models, please run the command below:
```shell
python Demo/Gradio/normal_vicuna7b.py
```
After running this command, you will get a local as well as public URL on your terminal. You can share the public URL with others to use as well. 

In the web Inferface, on the left side, you can drag and drop or upload the image in the image box and can enter the corresponding query in the text box. You can optionally tick the checkbox to translate the query in case it is in a language other than Hindi/English/Hinglish or their codemixed versions. In that case you'll have to append your BING Translator API key and region into the code where it is specified.

On the right side you can see the translated text in case you have selected the translation checkbox, and the output the model generates for your image and corresponding query.


