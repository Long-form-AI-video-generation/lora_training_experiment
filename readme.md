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

------------------------------------------------------------------------------------------

# LoRA Training of Wan 2.1 14B 480p Video Generation Model with a Dragon Ball Anime Dataset

## Introduction & Goal

This project documents the end-to-end process of fine-tuning the powerful **Wan 2.1 Image-to-Video 14B** model to capture a specific artistic motion style. The training was conducted using the `diffusion-pipe` library, a high-performance tool designed for large-scale model training.

The primary goal was to create a **"Pure Style" LoRA**—a lightweight adapter that can animate static images with the dynamic, high-energy motion characteristic of *Dragon Ball Z/Super* fight scenes. Crucially, the LoRA was trained without specific character names in the captions, making it a flexible tool for applying this iconic animation style to any subject.

The final output of this project is a versatile LoRA file (`adapter_model.safetensors`) intended for use in popular inference engines like **ComfyUI** to generate new, stylized video content.

## Project File Structure

```lora_training_experiment/
│
├── data_processor/
│ ├── full_videos/
│ ├── clipped_videos/
│ ├── create_curated_clips.py
│ └── generate_video_summaries.py
│
├── diffusion-pipe/
│ └── ... (Gitignored - The core training library)
│
├── training/
│ ├── wan_14b_min_vram.toml
│ └── my_wan_video_dataset.toml
│
├── evaluation/
│ ├── eval.py
│ ├── test_lora_strength.py
│ └── ...
│
└── README.md
```


### Folder Breakdown:

*   📁 **`data_processor/`**
    This directory serves as the "workshop" for the entire dataset creation pipeline. It contains the scripts and source materials needed to transform long-form video into a curated set of training clips.
    -   `full_videos/`(**ignored on github**) is for storing source videos and `
    -   `clipped_videos/` (**ignored on github**) is the destination for the final, processed clips and their corresponding captions.
    - `create_clips.py` : change a full video to  5 second videos based on the seconds specified in a txt folder.
    - `generate_video_summeries.py`: generate a summery of each 5 second videos in a txt file using a model(BLIP2 model)
*   📁 **`training/`**
    This folder contains the "blueprints" for the training process. These configuration files tell `diffusion-pipe` what to train and how to train it.
    -   `wan_14b_min_vram.toml`: Defines the main training parameters, such as learning rate, number of epochs, and memory-saving optimizations.
    -   `dataset.toml`: Specifies the path to the training data and parameters for how it should be processed (e.g., resolution, frame count).

*   📁 **`evaluation/`**
    This directory holds various Python scripts used for testing and validating the trained LoRA. These scripts are used to load the base model and the LoRA to generate sample videos for quality assessment.

*   📁 **`diffusion-pipe/`**
    This directory contains the core training engine. It is a clone of the official `diffusion-pipe` repository. **This folder is included in the `.gitignore` file** and is not tracked by this repository's version control. To replicate this project, you must clone `diffusion-pipe` separately into this location.


    ## 4. The Dataset: Curation & Philosophy

The quality of a LoRA is a direct reflection of the quality of its training data. For this project, a  multi-stage curation process was employed to build a small but highly effective dataset, moving beyond simple automation to ensure every training example was purposeful.

The final dataset consists of **58 curated, 5-second video clips**.



### Clipping Strategy: From Randomness to Curation

Initial strategies involved random sampling, but to maximize the quality of the dataset, a **curated approach** was adopted. Using the `create_clips.py` script, 58 specific, high-impact moments were identified and extracted from the source movie. This ensured that every single clip in the dataset contained a meaningful, dynamic action sequence relevant to the training goal, such as:
-   Rapid punch and kick exchanges.
-   Explosive energy blasts.
-   Dramatic power-up sequences and aura flares.

The clips were extracted using `ffmpeg` in a frame-accurate mode to guarantee precision, a critical step for quality machine learning datasets.

### The Art of Captioning: The "Human-in-the-Loop" Approach

Automated captioning was used as an initial step. The `generate_video_summaries.py` script, powered by the **BLIP-2 model**, provided a baseline description for each of the 58 clips.

Therefore, the final and most crucial phase was **manual review and curation of every caption.** Each auto-generated caption was replaced with a rich, descriptive prompt following a "Pure Style" philosophy. This involved:
-   **Omitting Character Names:** Using generic descriptions like "a warrior with golden hair" instead of "Goku" to ensure the LoRA learned a flexible *motion style* rather than a specific character.
-   **Adopting a Structured Prompt Format:** Each caption was written as a comma-separated list of tags and phrases, following a logical flow.

**This manual refinement process transformed the dataset from a simple collection of clips into a targeted, high-signal training curriculum for the LoRA.**
## 5. Tools & Techniques Used

This project was made possible by a combination of powerful software and advanced memory-saving techniques.

*   **Core Library:** **`diffusion-pipe`**, which leverages Microsoft's **DeepSpeed** library to enable the training of massive models on limited hardware.

*   **Fine-Tuning Method:** **LoRA (Low-Rank Adaptation)**, a Parameter-Efficient Fine-Tuning (PEFT) technique.
    *   **How:** Freezes the 14B parameter base model and trains only tiny "adapter" layers.
    *   **Benefits:** Results in a small LoRA file, faster training times, and preserves the base model's original knowledge.

*   **Hardware & Software:**
    *   **GPU:** A single **24GB VRAM** consumer GPU.
    *   **Environment:** **Conda** with **PyTorch**, `transformers`, and `diffusers`.

*   **Key Memory Optimizations (from `.toml` config):**
    *   **`transformer_dtype = 'float8'`**: Dramatically reduces VRAM by using 8-bit precision for the model's largest layers.
    *   **`activation_checkpointing`**: Trades computation for memory by re-calculating values instead of storing them.
    *   **`AdamW8bitKahan`**: An 8-bit optimizer that uses significantly less memory than its standard counterpart.


## 6. The Training Process: Step-by-Step Guide

This section provides a streamlined guide to launching the training process.

### 1. Setup

First, prepare the environment by cloning the necessary repositories and installing dependencies inside a dedicated Conda environment.

```bash
# 1. Clone this project and the diffusion-pipe tool
git clone [your-repo-url]
cd lora_training_experiment
git clone https://github.com/huggingface/diffusion-pipe.git

# 2. Create and activate the Conda environment
conda create --name diffusion-pipe python=3.12 -y
conda activate diffusion-pipe

# 3. Install dependencies (install PyTorch first)
pip install torch torchvision torqudio --index-url https://download.pytorch.org/whl/cu121
pip install -r diffusion-pipe/requirements.txt
2. Configuration
Next, configure the training by editing the two .toml files located in the /training directory.
my_wan_video_dataset.toml: Update the path variable to the absolute path of your clipped_videos dataset folder.
wan_14b_min_vram.toml: Update the ckpt_path to the absolute path of your downloaded base model and the output_dir to your desired save location.
3. Launching
Finally, launch the training using the deepspeed command. The following command is configured for a single GPU
# Ensure you are in the root of the `lora_training_experiment` directory
# and your Conda environment is active.

CUDA_VISIBLE_DEVICES=1 NCCL_P2P_DISABLE="1" NCCL_IB_DISABLE="1" \
deepspeed diffusion-pipe/train.py --deepspeed --config training/wan_14b_min_vram.toml
