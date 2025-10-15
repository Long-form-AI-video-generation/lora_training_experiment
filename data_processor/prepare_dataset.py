import os
import json
import shutil

# --- CONFIGURATION ---
# 1. Set the source directory where your videos and metadata.json are located.
SOURCE_DIR = "/home/comfy/projects/lora_training/dataset/10_dragonball"

# 2. Set the destination directory for your training data.
DEST_DIR = "/home/comfy/projects/lora_training_experiment/diffusion-pipe/data/input"

# 3. The name of your JSON metadata file.
METADATA_FILE = "updated_metadata.json"

# 4. VERY IMPORTANT: Define your unique trigger word for the LoRA.
#    Choose something unique that won't appear in normal prompts.
TRIGGER_WORD = "dbz_lora_style"
# --- END OF CONFIGURATION ---


def process_dataset():
    """
    Moves video files and creates corresponding .txt caption files
    in the format required by diffusion-pipe.
    """
    # Ensure the destination directory exists, create it if not.
    print(f"Ensuring destination directory exists: {DEST_DIR}")
    os.makedirs(DEST_DIR, exist_ok=True)

    metadata_path = os.path.join(SOURCE_DIR, METADATA_FILE)

    # --- Load Caption Data ---
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            # Assuming the JSON is a list of dictionaries.
            # If your file is not a valid list, you may need to add '[' at the
            # beginning and ']' at the end of the file.
            metadata_list = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: Metadata file not found at {metadata_path}")
        return
    except json.JSONDecodeError as e:
        print(f"ERROR: Could not decode JSON from {metadata_path}.")
        print("Please ensure it's a valid JSON file (e.g., a list of objects enclosed in []).")
        print(f"Error details: {e}")
        return
        
    # Create a fast lookup map: { "filename.mp4": "caption text" }
    # --- Safe caption map ---
    caption_map = {}
    for i, item in enumerate(metadata_list):
        file_name = item.get("file_name")
        caption = item.get("caption")
        if file_name is None or caption is None:
            print(f"Skipping invalid metadata entry {i}: {item}")
            continue
        caption_map[file_name] = caption

    print(f"Loaded {len(caption_map)} valid captions from {METADATA_FILE}")


    # --- Process Files ---
    files_processed = 0
    for filename in os.listdir(SOURCE_DIR):
        if filename.endswith(".mp4"):
            source_video_path = os.path.join(SOURCE_DIR, filename)

            if filename in caption_map:
                # 1. Get the original caption and clean it up
                original_caption = caption_map[filename]
                # Replace newlines with commas for a cleaner prompt
                cleaned_caption = original_caption.replace('\n', ', ').strip()
                
                # 2. Prepend the trigger word
                final_caption = f"{TRIGGER_WORD}, {cleaned_caption}"

                # 3. Define destination paths
                dest_video_path = os.path.join(DEST_DIR, filename)
                base_name, _ = os.path.splitext(filename)
                dest_txt_path = os.path.join(DEST_DIR, f"{base_name}.txt")

                # 4. Copy video file
                print(f"  -> Copying video: {filename}")
                shutil.copy(source_video_path, dest_video_path)

                # 5. Create .txt file with the final caption
                print(f"  -> Creating caption file: {base_name}.txt")
                with open(dest_txt_path, 'w', encoding='utf-8') as f:
                    f.write(final_caption)
                
                files_processed += 1
            else:
                print(f"WARNING: No caption found for '{filename}' in metadata. Skipping this file.")

    print("\n--- Processing Complete ---")
    print(f"Total videos processed: {files_processed}")
    print(f"Your dataset is now ready in: {DEST_DIR}")

if __name__ == "__main__":
    process_dataset()