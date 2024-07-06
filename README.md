# Codemixed_Clinical_VQA
## Contents
- [Install](#Install)
- [Models](#Models)
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
pip install -q Pillow requests .
```

## Models

## Demo

### Colab Notebook
You can run this colab notebook which includes the installation and both the CLI Interface demo and Gradio demos.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1XVyr9n2KANAuA5wCkwfc0Az7d3jwIxnj?usp=sharing) Vicuna 7B model.


### CLI Interface
For running normal models, please run the command below:
```shell
python Demo/Python/normal_vicuna7b.py
```
After running this command, the model will load and then you will be asked to input the path to the image. Following this, you will be prompted with a choice to translate your query in case it is in a langauge other than Hindi/English/Hinglish or their codemixed versions. Please enter 'yes' in that case. Then you can input your query and the model will output a medically precise English Question summarizing your query. 

You can do this for multiple queries. To exit the demo, please enter 'exit' when asked to input path to the image. 
### Gradio Web Interface
For running normal models, please run the command below:
```shell
python Demo/Gradio/normal_vicuna7b.py
```
After running this command, you will get a local as well as public URL on your terminal. You can share the public URL with others to use as well. 

In the web Inferface, on the left side, you can drag and drop or upload the image in the image box and can enter the corresponding query in the text box. You can optionally tick the checkbox to translate the query in case it is in a language other than Hindi/English/Hinglish or their codemixed versions. In that case you'll have to append your BING Translator API key and region into the code where it is specified.

On the right side you can see the translated text in case you have selected the translation checkbox, and the output the model generates for your image and corresponding query.


