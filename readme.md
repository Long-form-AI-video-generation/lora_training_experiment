### main runing command 
make sure you are in the diffusion-pipe  folder
then run ``` NCCL_P2P_DISABLE="1" NCCL_IB_DISABLE="1" deepspeed --num_gpus=1 train.py --deepspeed --config examples/wan_14b_min_vram.toml```

a very good article : https://www.stablediffusiontutorials.com/2025/03/wan-lora-train.html


intersting article: does the same thing basically: https://huggingface.co/Remade-AI/Super-Saiyan