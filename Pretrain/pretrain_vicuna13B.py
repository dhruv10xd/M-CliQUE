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

# Run pretrain scripts after adjusting parameters for single A100 GPU
!deepspeed llava/train/train_mem.py \
  --deepspeed ./scripts/zero2.json \
  --model_name_or_path lmsys/vicuna-13b-v1.5 \
  --version plain \
  --data_path 'combined_mcvqa_data.json' \
  --image_folder '' \
  --vision_tower openai/clip-vit-large-patch14-336 \
  --mm_projector_type mlp2x_gelu \
  --tune_mm_mlp_adapter True \
  --mm_vision_select_layer -2 \
  --mm_use_im_start_end False \
  --mm_use_im_patch_token False \
  --bf16 True \
  --output_dir './checkpoints/llava-v1.5-13b-pretrain' \
  --num_train_epochs 1 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 32 \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 24000 \
  --save_total_limit 1 \
  --learning_rate 1e-3 \
  --weight_decay 0. \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --tf32 True \
  --model_max_length 2048 \
  --gradient_checkpointing True \
  --dataloader_num_workers 4 \
  --lazy_preprocess True \
  --report_to wandb


# Save the projector weights from Google Colab in case running on Colab
import os
from google.colab import files

checkpoint_dir = '/content/LLaVA/checkpoints/llava-v1.5-13b-pretrain'
output_zip_file = '/content/llava_checkpoints.zip'

!zip -r {output_zip_file} {checkpoint_dir}

files.download(output_zip_file)

# Save the projector weights directly into your mounted drive
!cp -r "/content/LLaVA/checkpoints/llava-v1.5-13b-pretrain" "/content/drive/your-drive/"