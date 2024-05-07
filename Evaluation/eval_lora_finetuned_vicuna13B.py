# Import libraries
!pip install deepspeed==0.14.0
!pip install gdown==4.7.3
!pip install flash-attn
!pip install wandb

# Clone LLaVA github 
!git clone https://github.com/haotian-liu/LLaVA.git
%cd LLaVA

# Make a new environment
!conda create -n llava python=3.10 -y
!conda activate llava

# Install dependencies
!python -mpip install --upgrade pip  # enable PEP 660 support
!pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu117
!pip install -e .

# Inference for a single instance
!python -m llava.serve.cli \
    --model-path "saved/llava-v1.5-13b-lora" \
    --model-base "lmsys/vicuna-13b-v1.5" \
    --image-file "https://llava-vl.github.io/static/images/view.jpg" \
    --load-4bit # optional

# Run scripts for inference on entire dataset
!python llava/eval/model_vqa.py \
    --model-path "saved/llava-v1.5-13b-lora" \
    --model-base "lmsys/vicuna-13b-v1.5" \
    --image-folder "" \
    --question-file "/content/test.jsonl" \
    --answers-file "/content/answers.jsonl"