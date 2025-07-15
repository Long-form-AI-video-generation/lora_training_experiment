import os
import cv2
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from tqdm import tqdm

# --- CONFIGURATION ---
# The single directory containing your .mp4 video clips.
VIDEO_DIR = "/home/comfy/projects/lora_training_experiment/data_processor/clipped_videos"

# <<-- NEW -->>
# Your unique trigger word. This will be added to the start of every caption.
TRIGGER_WORD = "dbz_fight_style"

# How many frames to sample from each video for captioning.
# Increased to 15 for potentially better, more detailed captions.
FRAMES_PER_CLIP = 30

gpu_index = int(os.environ.get("BLIP2_GPU", 1))
device = f"cuda:{gpu_index}"
# --- END OF CONFIGURATION ---


def setup_model(device):
    """Loads the BLIP-2 model and processor onto the specified device."""
    print("Loading BLIP-2 model... (This may take a moment)")
    try:
        processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b", 
            torch_dtype=torch.float16
        ).to(device)
        print("Model loaded successfully.")
        return processor, model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

def caption_videos_in_directory(video_dir, processor, model, device):
    """
    Finds all .mp4 files in a directory, generates captions with a trigger word,
    and saves them as corresponding .txt files.
    """
    if not os.path.isdir(video_dir):
        print(f"Error: Directory not found at '{video_dir}'")
        return

    video_files = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]
    if not video_files:
        print(f"No .mp4 files found in '{video_dir}'.")
        return

    print(f"Found {len(video_files)} videos to process.")

    for filename in tqdm(video_files, desc="Captioning Videos"):
        video_path = os.path.join(video_dir, filename)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"\nWarning: Could not open {filename}, skipping.")
            continue
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = max(1, total_frames // FRAMES_PER_CLIP)
        
        frames = []
        for i in range(FRAMES_PER_CLIP):
            frame_index = i * frame_interval
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
        cap.release()
        
        if not frames:
            print(f"\nWarning: Failed to read frames from {filename}, skipping.")
            continue

        try:
            # Use keyword arguments for device and dtype for clarity and correctness
            inputs = processor(images=frames, return_tensors="pt").to(device=device, dtype=torch.float16)
            generated_ids = model.generate(**inputs, max_new_tokens=50)
            captions = [processor.decode(gid, skip_special_tokens=True).strip() for gid in generated_ids]
            
            # --- IMPROVED CAPTION FORMATTING ---
            # 1. Get unique, non-empty parts from BLIP's output
            summary_parts = list(dict.fromkeys([cap for cap in captions if cap]))
            blip_summary = ". ".join(summary_parts[:3])

            # 2. Construct the final prompt with the trigger word
            if blip_summary:
                # Prepend the trigger word to the BLIP summary
                final_prompt = f"{TRIGGER_WORD}, {blip_summary}, in a Dragon Ball Z style fight, sakuga animation"
            else:
                # Fallback if BLIP returns nothing
                final_prompt = f"{TRIGGER_WORD}, action scene in a Dragon Ball Z style fight, sakuga animation"
            
            # --- Output Writing ---
            base_name, _ = os.path.splitext(filename)
            output_txt_path = os.path.join(video_dir, f"{base_name}.txt")
            
            with open(output_txt_path, 'w', encoding='utf-8') as f:
                f.write(final_prompt)

        except Exception as e:
            print(f"\nError processing {filename}: {e}")
            continue

    print("\n--- Captioning Complete ---")
    print(f"All .txt files have been saved in: {video_dir}")


if __name__ == "__main__":
    blip_processor, blip_model = setup_model(device)
    
    if blip_processor and blip_model:
        caption_videos_in_directory(VIDEO_DIR, blip_processor, blip_model, device)