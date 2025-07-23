import torch
from diffusers import DiffusionPipeline, UNet2DConditionModel
from peft import PeftModel
from PIL import Image
import os

# --- 1. CONFIGURATION: UPDATE THESE PATHS ---

# Path to the base Wan2.1 I2V model directory
BASE_MODEL_PATH = '/home/comfy/projects/framepack-wan/Wan2.1/Wan2.1-I2V-14B-480P'

# Path to your trained LoRA .safetensors file (from your output directory)
# Example: '/path/to/output/epoch_20/lora.safetensors'
LORA_CHECKPOINT_PATH = "/home/comfy/projects/lora_training_experiment/diffusion-pipe/data/output/20250715_16-17-40/epoch20/adapter_model.safetensors"

# Path to a starting image you want to animate
INPUT_IMAGE_PATH = "/home/comfy/projects/lora_training_experiment/images/Gohan.webp"

# The prompt to guide the video generation.
# CRITICAL: It MUST include your trigger word!
PROMPT = "dbz_fight_style, a warrior unleashes a powerful punch, shockwave effect, sakuga animation"

# Where to save the final video
OUTPUT_VIDEO_PATH = "lora_test_epoch20.mp4"

# --- 2. SCRIPT LOGIC ---

def main():
    print("--- Starting LoRA Evaluation Script ---")

    # --- Step 1: Load the Base Diffusion Pipeline ---
    print(f"Loading base model from: {BASE_MODEL_PATH}")
    # We load the base model in float16 for efficiency
    pipe = DiffusionPipeline.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.float16,
        variant="fp16" # Use fp16 variant if available for faster loading
    )
    # Move the pipeline to the GPU
    pipe = pipe.to("cuda")

    # --- Step 2: Load and Fuse the LoRA weights ---
    print(f"Loading and fusing LoRA from: {LORA_CHECKPOINT_PATH}")
    
    # The 'peft' library handles the magic of applying the LoRA.
    # We specify the subfolder where the main model (unet) is located.
    pipe.unet = PeftModel.from_pretrained(pipe.unet, os.path.dirname(LORA_CHECKPOINT_PATH), subfolder='.')
    
    # Optional but recommended: For some LoRAs, you might need to fuse them for inference
    # pipe.fuse_lora() # This merges the weights permanently for this session

    # --- Step 3: Prepare Inputs ---
    print(f"Loading input image from: {INPUT_IMAGE_PATH}")
    try:
        start_image = Image.open(INPUT_IMAGE_PATH).convert("RGB")
    except FileNotFoundError:
        print(f"ERROR: Input image not found at '{INPUT_IMAGE_PATH}'")
        return

    print(f"Using prompt: '{PROMPT}'")

    # --- Step 4: Generate the Video ---
    print("Generating video... This will take a few minutes.")
    
    # You can adjust these parameters
    video_frames = pipe(
        prompt=PROMPT,
        image=start_image,
        num_frames=81,       # Number of frames to generate
        num_inference_steps=50, # More steps can improve quality
        guidance_scale=7.5,   # How strongly to follow the prompt
    ).frames[0] # The output is a list containing a list of PIL Images

    # The output from the pipeline is a list of PIL Images.
    # We need to save them as a video file. diffusers has a helper for this.
    from diffusers.utils import export_to_video
    
    print(f"Saving video to: {OUTPUT_VIDEO_PATH}")
    export_to_video(video_frames, OUTPUT_VIDEO_PATH, fps=24)

    print("\n--- Evaluation Complete! ---")
    print(f"Video saved successfully. Check '{OUTPUT_VIDEO_PATH}' to see the result.")


if __name__ == "__main__":
    # Ensure you have a GPU available
    if not torch.cuda.is_available():
        print("ERROR: This script requires a GPU with CUDA support.")
    else:
        main()