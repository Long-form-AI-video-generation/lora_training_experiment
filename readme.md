### main runing command 
make sure you are in the diffusion-pipe  folder
then run ``` CUDA_VISIBLE_DEVICES=1 NCCL_P2P_DISABLE="1" NCCL_IB_DISABLE="1" deepspeed --num_gpus=1 train.py --deepspeed --config examples/wan_14b_min_vram.toml```

a very good article : https://www.stablediffusiontutorials.com/2025/03/wan-lora-train.html


intersting article: does the same thing basically: https://huggingface.co/Remade-AI/Super-Saiyan

CUDA_VISIBLE_DEVICES=1 NCCL_P2P_DISABLE="1" NCCL_IB_DISABLE="1" deepspeed train.py --deepspeed --config examples/wan_14b_min_vram.toml

12 hour epoch 12, 



for the cakeified model that was found on hugging face
Training Details
Base Model: Wan2.1 14B I2V 480p
Training Data: 1 minute of video (13 short clips of things being cakeified, each clip captioned separately)
Epochs: 16