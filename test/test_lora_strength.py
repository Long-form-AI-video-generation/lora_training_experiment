import torch
from diffusers import DiffusionPipeline
from peft import PeftModel
from PIL import Image
import os
from diffusers.utils import export_to_video

# --- 1. CONFIGURATION: UPDATE THESE ---

# Path to the base Wan2.1 I2V model directory
BASE_MODEL_PATH = '/home/comfy/projects/framepack-wan/Wan2.1/Wan2.1-I2V-14B-480P'

# Path to the FOLDER containing your LoRA checkpoint.
# peft needs the directory, not the direct .safetensors file for this method.
# Example: '/path/to/output/epoch_20/'
LORA_FOLDER_PATH = "/home/comfy/projects/lora_training/wan_14b_output/epoch_20" 

# Path to a starting image you want to animate
INPUT_IMAGE_PATH = "/home/comfy/projects/lora_training_experiment/images/Gohan.webp"

# The prompt to guide the video generation. MUST include your trigger word!
PROMPT = "dbz_fight_style, a warrior unleashes a powerful punch, shockwave effect, sakuga animation"

# Where to save the output videos. A subfolder will be created.
OUTPUT_DIR = "lora_strength_tests"

# The different LoRA weights you want to test
WEIGHTS_TO_TEST = [0.5, 0.7, 0.8, 1.0, 1.2]

# --- 2. SCRIPT LOGIC ---

def main():
    print("--- Starting LoRA Strength Test Script ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Step 1: Load the Base Diffusion Pipeline ---
    print(f"Loading base model from: {BASE_MODEL_PATH}")
    pipe = DiffusionPipeline.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.float16,
        variant="fp16"
    ).to("cuda")

    # --- Step 2: Load the LoRA model weights (but don't fuse yet) ---
    print(f"Loading LoRA from folder: {LORA_FOLDER_PATH}")
    # This loads the LoRA and attaches it to the unet, but keeps it separate
    pipe.load_lora_weights(LORA_FOLDER_PATH)

    # --- Step 3: Prepare Input Image ---
    print(f"Loading input image from: {INPUT_IMAGE_PATH}")
    try:
        start_image = Image.open(INPUT_IMAGE_PATH).convert("RGB")
    except FileNotFoundError:
        print(f"ERROR: Input image not found at '{INPUT_IMAGE_PATH}'")
        return

    # --- Step 4: Loop Through Weights and Generate Videos ---
    for weight in WEIGHTS_TO_TEST:
        print(f"\n--- Generating video with LoRA weight: {weight} ---")

        # Set the LoRA strength using a cross-attention keyword argument.
        # This is the modern way to scale LoRA influence in diffusers.
        video_frames = pipe(
            prompt=PROMPT,
            image=start_image,
            num_frames=81,
            num_inference_steps=50,
            guidance_scale=7.5,
            cross_attention_kwargs={"scale": weight} # This sets the LoRA strength
        ).frames[0]

        # Save the resulting video
        output_path = os.path.join(OUTPUT_DIR, f"lora_test_weight_{weight}.mp4")
        print(f"Saving video to: {output_path}")
        export_to_video(video_frames, output_path, fps=24)

    print("\n--- Strength Test Complete! ---")
    print(f"All videos saved in the '{OUTPUT_DIR}' folder.")

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("ERROR: This script requires a GPU with CUDA support.")
    else:
        main()