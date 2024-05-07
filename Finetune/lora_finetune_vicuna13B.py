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

# Clone the github for MMCQS dataset
!git clone "https://github.com/ArkadeepAcharya/MedSumm-ECIR2024.git"

# Get the original images folder
!gdown 1HHOuu7KU22T1JGiszt8xI4D6wP8f0e8s -O Multimodal_images.zip

#Replace the sample images folder with the original folder 
!rm -rf "/content/LLaVa/MedSumm-ECIR2024/Multimodal_images"
!unzip /content/LLaVA/Multimodal_images.zip -d "/content/LLaVA/MedSumm-ECIR2024/"
!rm -rf "/content/Multimodal_images.zip"

# Run finetune scripts after adjusting parameters for single A100 GPU and corresponding lora config
!deepspeed llava/train/train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero3_offload.json \
    --model_name_or_path lmsys/vicuna-13b-v1.5 \
    --version v1 \
    --data_path '/content/MMCQS_finetune.json' \
    --image_folder '' \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter llava-v1.5-13b-pretrain/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/llava-v1.5-13b-lora \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 32 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
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

# Save the fientuned vicuna 13B weights into your drive from Google Colab in case running on Colab
!cp -r "/content/LLaVA/checkpoints/llava-v1.5-13b" "/content/drive/your-drive"