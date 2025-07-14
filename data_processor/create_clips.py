import os
import subprocess
from tqdm import tqdm

# --- CONFIGURATION ---
# The location of your full-length movie(s) and the .txt file defining clips.
SOURCE_DIR = "/home/comfy/projects/lora_training_experiment/data_processor/full_videos"
# Where the final, small clips will be saved.
CLIPPED_VIDEOS_DIR = "/home/comfy/projects/lora_training_experiment/data_processor/clipped_videos"

# The name of the text file that lists all the clips to be created.
CLIP_LIST_FILENAME = "clips_to_cut.txt"
# --- END OF CONFIGURATION ---

def parse_clip_list(file_path):
    """Parses a file for a list of clips to be cut."""
    clips_to_make = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                # Skip comments and empty lines
                if line.strip().startswith('#') or not line.strip():
                    continue
                
                parts = line.strip().split(',')
                if len(parts) == 3:
                    start_time, duration, output_base = parts
                    clips_to_make.append({
                        "start": int(start_time),
                        "duration": int(duration),
                        "output_base": output_base.strip()
                    })
    except FileNotFoundError:
        print(f"Error: Clip list file not found at '{file_path}'")
        return None
    except Exception as e:
        print(f"Error parsing clip list file '{file_path}': {e}")
        return None
    return clips_to_make

def clip_video_accurate(input_path, output_path, start_time, duration):
    """Uses ffmpeg to create a FRAME-ACCURATE video clip by re-encoding."""
    command = [
        'ffmpeg',
        '-i', input_path,
        '-ss', str(start_time),
        '-t', str(duration),
        '-c:v', 'libx264',
        '-preset', 'fast',
        '-crf', '18',
        '-c:a', 'aac',
        '-b:a', '192k',
        '-y',
        output_path,
        '-hide_banner',
        '-loglevel', 'error'
    ]
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError:
        print("\nFATAL ERROR: ffmpeg not found. Please ensure it's in your system's PATH.")
        exit()
    except subprocess.CalledProcessError as e:
        print(f"\nError running ffmpeg for {os.path.basename(output_path)}: {e}")

def main():
    """Main function to find a video and its clip list, then create all clips."""
    print("--- Starting Curated Video Clipping Process ---")
    os.makedirs(CLIPPED_VIDEOS_DIR, exist_ok=True)
    
    # Assuming there's only one main video file we are working with
    video_files = [f for f in os.listdir(SOURCE_DIR) if f.endswith(('.mp4', '.mkv'))]
    if not video_files:
        print(f"No video files found in '{SOURCE_DIR}'. Exiting.")
        return
        
    source_video_path = os.path.join(SOURCE_DIR, video_files[0])
    source_video_basename, _ = os.path.splitext(video_files[0])
    clip_list_path = os.path.join(SOURCE_DIR, CLIP_LIST_FILENAME)

    print(f"Using source video: {video_files[0]}")
    print(f"Reading clip definitions from: {CLIP_LIST_FILENAME}")

    clips_to_make = parse_clip_list(clip_list_path)
    if not clips_to_make:
        print("No valid clips found in the list. Exiting.")
        return

    print(f"Found {len(clips_to_make)} clips to create. Starting now...")

    for clip_info in tqdm(clips_to_make, desc="Creating Clips"):
        start = clip_info["start"]
        duration = clip_info["duration"]
        output_filename = f"{source_video_basename}_{clip_info['output_base']}.mp4"
        output_path = os.path.join(CLIPPED_VIDEOS_DIR, output_filename)
        
        clip_video_accurate(source_video_path, output_path, start, duration)

    print("\n--- All Curated Clips Created ---")
    print(f"Your dataset of {len(clips_to_make)} clips is ready in: '{CLIPPED_VIDEOS_DIR}'")

if __name__ == "__main__":
    main()