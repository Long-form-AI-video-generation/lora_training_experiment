import torch
from diffusers import DiffusionPipeline
from PIL import Image
import os
from tqdm import tqdm

# --- 1. CONFIGURATION: UPDATE THESE ---

# Path to the base Wan2.1 I2V model directory
BASE_MODEL_PATH = "/home/comfy/projects/lora_training_experiment/models/WAN2.1_I2V_14B"

# Path to your trained LoRA .safetensors file
LORA_CHECKPOINT_PATH = "/home/comfy/projects/lora_training_experiment/diffusion-pipe/data/output/20250715_16-17-40/epoch20/adapter_model.safetensors"
# Path to a starting image you want to animate
INPUT_IMAGE_PATH = "/home/comfy/projects/lora_training_experiment/images/Gohan.webp"

# The prompt to guide the video generation. MUST include your trigger word!
PROMPT = "dbz_fight_style, a warrior unleashes a powerful punch, shockwave effect, sakuga animation"

# Where to save the output frames. A subfolder will be created.
OUTPUT_DIR = "latent_walk_frames"

# How many inference steps to perform in total
NUM_INFERENCE_STEPS = 50

# How often to save an intermediate image (e.g., save every 5 steps)
SAVE_EVERY_N_STEPS = 5

# --- 2. SCRIPT LOGIC ---

@torch.no_grad()
def main():
    print("--- Starting Latent Space Walk Script ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Step 1: Load the Pipeline and LoRA ---
    print("Loading base model and LoRA...")
    pipe = DiffusionPipeline.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.float16,
        variant="fp16"
    ).to("cuda")
    
    # This is the simpler method to load and fuse a LoRA for a single run
    pipe.load_lora_weights(os.path.dirname(LORA_CHECKPOINT_PATH), weight_name=os.path.basename(LORA_CHECKPOINT_PATH))
    pipe.fuse_lora() # Fuse weights for optimal inference speed
    print("Model and LoRA loaded and fused.")

    # --- Step 2: Prepare Inputs (Image, Prompt, Latents) ---
    print("Preparing inputs...")
    try:
        start_image = Image.open(INPUT_IMAGE_PATH).convert("RGB")
    except FileNotFoundError:
        print(f"ERROR: Input image not found at '{INPUT_IMAGE_PATH}'")
        return
        
    prompt_embeds, _ = pipe.encode_prompt(PROMPT, "cuda", 1, True)
    image_embeds = pipe.image_encoder(pipe.image_processor.preprocess(start_image, height=512, width=512).to("cuda", dtype=torch.float16)).image_embeds
    
    # Prepare initial random noise (the starting point of diffusion)
    latents = torch.randn(
        (1, 4, 81, 64, 64), # Shape for this model
        generator=torch.manual_seed(42), # Use a seed for reproducibility
        device="cuda",
        dtype=torch.float16
    )
    
    # Set the number of steps
    pipe.scheduler.set_timesteps(NUM_INFERENCE_STEPS, device="cuda")
    timesteps = pipe.scheduler.timesteps
    
    # --- Step 3: The Denoising Loop (The "Walk") ---
    print("Starting the denoising walk...")
    for i, t in enumerate(tqdm(timesteps, desc="Diffusion Steps")):
        # Denoise one step
        noise_pred = pipe.unet(latents, t, encoder_hidden_states=prompt_embeds, added_cond_kwargs={"image_embeds": image_embeds}).sample
        latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

        # --- Step 4: Decode and Save Intermediate Image ---
        # Check if it's time to save an image
        if (i + 1) % SAVE_EVERY_N_STEPS == 0 or i == len(timesteps) - 1:
            # Decode the current latents into a viewable image (we'll just take the first frame)
            # We scale the latents by the VAE's scaling factor before decoding
            frame_latent = latents[:, :, 0, :, :].unsqueeze(2) / pipe.vae.config.scaling_factor
            image = pipe.vae.decode(frame_latent).sample
            
            # Convert to PIL Image and save
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            image = Image.fromarray((image * 255).astype("uint8"))
            
            output_path = os.path.join(OUTPUT_DIR, f"walk_step_{i+1:03d}.png")
            image.save(output_path)
            # print(f"Saved intermediate frame to {output_path}")

    print("\n--- Latent Walk Complete! ---")
    print(f"All frames saved in the '{OUTPUT_DIR}' folder.")

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("ERROR: This script requires a GPU with CUDA support.")
    else:
        main()