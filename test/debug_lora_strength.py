import torch
from diffusers import DiffusionPipeline
from PIL import Image
import os
from diffusers.utils import export_to_video
import random

# --- 1. CONFIGURATION ---
BASE_MODEL_PATH = '/home/comfy/projects/lora_training_experiment/WAN_14B_Diffusers_Format'
LORA_FOLDER_PATH = "/home/comfy/projects/lora_training_experiment/diffusion-pipe/data/output/20250715_16-17-40/epoch20"
INPUT_IMAGE_PATH = "/home/comfy/projects/lora_training_experiment/images/Gohan.webp"
PROMPT = "dbz_fight_style, a warrior unleashes a powerful punch, shockwave effect, sakuga animation"

# We will test two weights: one high, one low, to see if there is ANY effect.
WEIGHTS_TO_TEST = [0.0, 1.0] 
# 0.0 = No LoRA (should be a static image)
# 1.0 = Full LoRA strength (should show motion)

# Let's try to generate 21 frames, as this is a standard length.
NUM_FRAMES = 21

# --- 2. SCRIPT LOGIC ---
def main():
    print("--- Starting LoRA Debugging Script (Strength Test) ---")
    
    pipe = DiffusionPipeline.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="balanced"
    )
    pipe.enable_attention_slicing()

    # IMPORTANT: We load the LoRA weights but DO NOT fuse them.
    print(f"Loading LoRA weights from: {LORA_FOLDER_PATH}")
    pipe.load_lora_weights(LORA_FOLDER_PATH, weight_name="adapter_model.safetensors")

    start_image = Image.open(INPUT_IMAGE_PATH).convert("RGB")
    
    for weight in WEIGHTS_TO_TEST:
        print(f"\n--- Generating with LoRA weight: {weight} ---")
        
        generator = torch.Generator(device=pipe.device).manual_seed(1234) # Use a fixed seed for a fair comparison

        try:
            video_frames = pipe(
                prompt=PROMPT,
                image=start_image,
                num_frames=NUM_FRAMES,
                num_inference_steps=50,
                guidance_scale=7.5,
                generator=generator,
                cross_attention_kwargs={"scale": weight} # Directly control LoRA strength here
            ).frames[0]

            output_path = f"debug_test_weight_{weight}.mp4"
            print(f"Saving video to: {output_path}")
            export_to_video(video_frames, output_path, fps=24)
            print("Video saved successfully.")

        except Exception as e:
            print(f"An error occurred during generation with weight {weight}: {e}")
            print("This might indicate a deeper incompatibility even with this method.")

    print("\n--- Debugging Test Complete! ---")
    print("Compare 'debug_test_weight_0.0.mp4' and 'debug_test_weight_1.0.mp4'.")

if __name__ == "__main__":
    main()