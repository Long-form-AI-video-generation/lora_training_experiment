import torch
from diffusers import DiffusionPipeline
from PIL import Image
import os
from diffusers.utils import export_to_video
import random

# --- 1. CONFIGURATION ---
# Paths remain the same
BASE_MODEL_PATH = '/home/comfy/projects/lora_training_experiment/WAN_14B_Diffusers_Format'
LORA_FOLDER_PATH = "/home/comfy/projects/lora_training_experiment/diffusion-pipe/data/output/20250715_16-17-40/epoch20"
INPUT_IMAGE_PATH = "/home/comfy/projects/lora_training_experiment/images/Gohan.webp"
PROMPT = "dbz_fight_style, a warrior unleashes a powerful punch, shockwave effect, sakuga animation"
OUTPUT_VIDEO_PATH = "lora_test_multi_gpu_2.mp4"

# --- 2. MULTI-GPU SCRIPT LOGIC ---

def main():
    print("--- Starting LoRA Evaluation Script (Multi-GPU) ---")

    # --- Step 1: Load the Pipeline with Automatic Device Mapping ---
    print(f"Loading base model from: {BASE_MODEL_PATH}")
    
    # This is the key change for multi-GPU. We load in 16-bit for full quality.
    # `device_map="auto"` will automatically split the model across all visible GPUs.
    pipe = DiffusionPipeline.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="balanced" 
    )

    # We do not need .enable_model_cpu_offload() or .to("cuda")
    # `device_map` handles all device placement.
    
    # We can still use attention slicing for extra VRAM safety
    print("Enabling attention slicing...")
    pipe.enable_attention_slicing()

    # --- Step 2: Load and Fuse the LoRA weights ---
    print(f"Loading LoRA weights from: {LORA_FOLDER_PATH}")
    
    # load_lora_weights works seamlessly with a model split across GPUs
    pipe.load_lora_weights(LORA_FOLDER_PATH, weight_name="adapter_model.safetensors")
    
    print("Fusing LoRA weights into the transformer...")
    pipe.fuse_lora()

    # --- Step 3: Prepare Inputs ---
    print(f"Loading input image from: {INPUT_IMAGE_PATH}")
    try:
        start_image = Image.open(INPUT_IMAGE_PATH).convert("RGB")
    except FileNotFoundError:
        print(f"ERROR: Input image not found at '{INPUT_IMAGE_PATH}'")
        return

    print(f"Using prompt: '{PROMPT}'")

    # --- Step 4: Generate the Video ---
    print("Generating video... This may take a few minutes.")
    
    # No changes needed here. PyTorch will handle the cross-GPU communication.
    generator = torch.Generator(device=pipe.device).manual_seed(random.randint(0, 2**32 - 1))

    video_frames = pipe(
        prompt=PROMPT,
        image=start_image,
        num_frames=2,
        num_inference_steps=50,
        guidance_scale=7.5,
        generator=generator,
    ).frames[0]

    # --- Step 5: Save the Output ---
    print(f"Saving video to: {OUTPUT_VIDEO_PATH}")
    export_to_video(video_frames, OUTPUT_VIDEO_PATH, fps=24)

    print("\n--- Multi-GPU Evaluation Complete! ---")
    print(f"Video saved successfully. Check '{OUTPUT_VIDEO_PATH}' to see the result.")

if __name__ == "__main__":
    if torch.cuda.device_count() < 2:
        print("ERROR: This script is configured for multi-GPU but found less than 2 GPUs.")
        print("Please set CUDA_VISIBLE_DEVICES correctly (e.g., 'CUDA_VISIBLE_DEVICES=2,3').")
    else:
        main()