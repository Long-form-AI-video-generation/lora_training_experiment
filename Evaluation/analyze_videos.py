import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import linalg
import cv2
from tqdm import tqdm
import lpips
import decord
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.metrics import structural_similarity as ssim
from PIL import Image

# Suppress unnecessary warnings
logging.getLogger('matplotlib').setLevel(logging.ERROR)

class UnifiedVideoEvaluator:
    # ... (The entire class LoRAVideoEvaluator remains EXACTLY THE SAME as before) ...
    # It already handles both .webp and .mp4 correctly in the load_video function.
    # No changes are needed inside the class.
    
    def __init__(self, device='cuda', save_dir='evaluation_results'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        
        self.logger.info("Initializing evaluation models (LPIPS, I3D)...")
        self.lpips_fn = lpips.LPIPS(net='alex').to(self.device)
        import torchvision.models as models
        self.i3d_model = models.video.r3d_18(weights='R3D_18_Weights.DEFAULT').to(self.device)
        self.i3d_model.fc = nn.Identity()
        self.i3d_model.eval()
        self.logger.info("Evaluation models loaded.")
    
    def load_video(self, video_path: str) -> torch.Tensor:
        """Load video from .webp or .mp4 and convert to tensor."""
        if video_path.lower().endswith('.webp'):
            frames = []
            with Image.open(video_path) as img:
                for i in range(img.n_frames):
                    img.seek(i)
                    frame_np = np.array(img.convert("RGB"))
                    frames.append(torch.from_numpy(frame_np).permute(2, 0, 1))
            video = torch.stack(frames).float().permute(1, 0, 2, 3)
        else: # Assumes .mp4 or other decord-compatible formats
            decord.bridge.set_bridge('torch')
            vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
            video = vr[:].float().permute(3, 0, 1, 2)
        return video / 127.5 - 1.0

    def compute_fvd(self, videos1: List[torch.Tensor], videos2: List[torch.Tensor]) -> float:
        """Compute Fréchet Video Distance."""
        def extract_features(videos, desc):
            features = []
            with torch.no_grad():
                for video in tqdm(videos, desc=desc):
                    video = video.to(self.device)
                    if video.shape[1] < 16: video = F.pad(video, (0,0,0,0,0,16-video.shape[1]), "replicate")
                    video_norm = (video + 1) / 2
                    mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(3, 1, 1, 1)
                    std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(3, 1, 1, 1)
                    video_norm = (video_norm - mean) / std
                    feat = self.i3d_model(video_norm.unsqueeze(0)).squeeze().cpu().numpy()
                    features.append(feat)
            return np.array(features)
        
        features1 = extract_features(videos1, "Extracting Features (Set 1)")
        features2 = extract_features(videos2, "Extracting Features (Set 2)")
        mu1, sigma1 = features1.mean(0), np.cov(features1, rowvar=False)
        mu2, sigma2 = features2.mean(0), np.cov(features2, rowvar=False)
        diff = mu1 - mu2
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if np.iscomplexobj(covmean): covmean = covmean.real
        return float(diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean))

    def compute_ssim(self, video1: torch.Tensor, video2: torch.Tensor) -> float:
        video1_np = ((video1.permute(1, 2, 3, 0).cpu().numpy() + 1) / 2.0).clip(0, 1)
        video2_np = ((video2.permute(1, 2, 3, 0).cpu().numpy() + 1) / 2.0).clip(0, 1)
        return np.mean([ssim(video1_np[i], video2_np[i], channel_axis=2, data_range=1.0) for i in range(min(video1_np.shape[0], video2_np.shape[0]))])

    def compute_temporal_consistency(self, video: torch.Tensor) -> float:
        video_np = np.clip((video.permute(1, 2, 3, 0).cpu().numpy() + 1) * 127.5, 0, 255).astype(np.uint8)
        flow_mags = []
        for t in range(video_np.shape[0] - 1):
            frame1, frame2 = cv2.cvtColor(video_np[t], cv2.COLOR_RGB2GRAY), cv2.cvtColor(video_np[t+1], cv2.COLOR_RGB2GRAY)
            flow = cv2.calcOpticalFlowFarneback(frame1, frame2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            flow_mags.append(np.mean(np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)))
        return 1.0 / (1.0 + np.var(flow_mags)) if flow_mags else 1.0
    
    def run_comparison_evaluation(self, base_videos_paths: List[str], lora_videos_paths: List[str], lora_name: str):
        self.logger.info(f"--- Running COMPARISON Evaluation for {lora_name} LoRA ---")
        base_tensors = [self.load_video(p) for p in tqdm(base_videos_paths, desc="Loading Base Videos")]
        lora_tensors = [self.load_video(p) for p in tqdm(lora_videos_paths, desc="Loading LoRA Videos")]
        
        results = {
            'fvd_style_difference': self.compute_fvd(base_tensors, lora_tensors),
            'mean_ssim_vs_base': np.mean([self.compute_ssim(base_tensors[i], lora_tensors[i]) for i in range(len(base_tensors))]),
            'mean_temporal_consistency_base': np.mean([self.compute_temporal_consistency(t) for t in base_tensors]),
            'mean_temporal_consistency_lora': np.mean([self.compute_temporal_consistency(t) for t in lora_tensors]),
        }
        
        for k, v in results.items(): results[k] = float(v)
        
        report_file = self.save_dir / f'report_{lora_name}.json'
        with open(report_file, 'w') as f: json.dump(results, f, indent=2)
        self.logger.info(f"Report for {lora_name} saved to: {report_file}")
        
        self.generate_style_lora_plots(results, lora_name)
    
    def run_fidelity_evaluation(self, lora_videos_paths: List[str], lora_name: str):
        self.logger.info(f"--- Running FIDELITY Evaluation for {lora_name} Character LoRA ---")
        lora_tensors = [self.load_video(p) for p in tqdm(lora_videos_paths, desc=f"Loading {lora_name} Videos")]

        results = {
            'mean_temporal_consistency': np.mean([self.compute_temporal_consistency(t) for t in lora_tensors]),
            'num_videos': len(lora_tensors)
        }
        for k, v in results.items(): results[k] = float(v)
        
        report_file = self.save_dir / f'report_{lora_name}.json'
        with open(report_file, 'w') as f: json.dump(results, f, indent=2)
        self.logger.info(f"Report for {lora_name} saved to: {report_file}")
        
        self.generate_character_lora_plots(results, lora_name)

    def generate_style_lora_plots(self, data, lora_name):
        plt.style.use('seaborn-v0_8-darkgrid')
        
        fvd = data['fvd_style_difference']
        fig, ax = plt.subplots(figsize=(8, 5))
        fvd_color = '#2ecc71' if fvd > 50 else '#f39c12'
        bars = ax.bar(['Style Difference'], [fvd], color=fvd_color)
        ax.text(bars[0].get_x() + bars[0].get_width()/2., fvd + 5, f'{fvd:.2f}', ha='center', fontweight='bold')
        ax.set_ylabel('FVD Score')
        ax.set_title(f'FVD (Higher = More Stylistic Change) - {lora_name}', fontsize=14, weight='bold')
        plt.savefig(self.save_dir / f'plot_fvd_{lora_name}.png', dpi=150)
        plt.close()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        ssim = data['mean_ssim_vs_base']
        axes[0].bar(['Structural Similarity'], [ssim], color='#3498db')
        axes[0].set_ylim(0, 1)
        axes[0].text(0, ssim + 0.02, f'{ssim:.2%}', ha='center', fontweight='bold', fontsize=12)
        axes[0].set_title('Content Preservation (SSIM)', fontsize=14, weight='bold')
        
        temp_base = data['mean_temporal_consistency_base']
        temp_lora = data['mean_temporal_consistency_lora']
        axes[1].bar(['Base Model', 'With LoRA'], [temp_base, temp_lora], color=['#95a5a6', '#9b59b6'])
        axes[1].set_ylim(0, 1)
        axes[1].set_title('Temporal Consistency', fontsize=14, weight='bold')
        
        fig.suptitle(f'{lora_name} LoRA Quality Assessment', fontsize=18, weight='bold')
        plt.tight_layout()
        plt.savefig(self.save_dir / f'plot_quality_{lora_name}.png', dpi=150)
        plt.close()
        self.logger.info(f"Plots for {lora_name} saved to: {self.save_dir}")

    def generate_character_lora_plots(self, data, lora_name):
        plt.style.use('seaborn-v0_8-darkgrid')
        
        temp_lora = data['mean_temporal_consistency']
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar([f'{lora_name} Videos'], [temp_lora], color='#9b59b6')
        ax.set_ylim(0, 1)
        ax.text(0, temp_lora + 0.02, f'{temp_lora:.3f}', ha='center', fontweight='bold', fontsize=12)
        ax.set_ylabel('Consistency Score (Higher is Better)')
        ax.set_title(f'Temporal Consistency for {lora_name} LoRA', fontsize=14, weight='bold')
        ax.grid(True, axis='y', alpha=0.5)
        plt.savefig(self.save_dir / f'plot_consistency_{lora_name}.png', dpi=150)
        plt.close()
        self.logger.info(f"Plot for {lora_name} saved to: {self.save_dir}")

# --- NEW: main() function is now more flexible ---
def main():
    parser = argparse.ArgumentParser(description='A unified evaluator for Style and Character LoRAs.')
    parser.add_argument('--lora_dir', type=str, required=True, help='Path to the directory of LoRA-generated videos.')
    parser.add_argument('--base_dir', type=str, default=None, help='(For Style LoRAs) Path to the directory of base model videos for comparison.')
    parser.add_argument('--lora_name', type=str, required=True, help='Name of the LoRA for titles and filenames (e.g., DBZ_Style, Demetri).')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save the reports and plots.')
    # --- THIS IS THE NEW ARGUMENT ---
    parser.add_argument('--ext', type=str, default='webp', help='The file extension of the videos to look for (e.g., webp, mp4).')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    args = parser.parse_args()

    evaluator = UnifiedVideoEvaluator(device=args.device, save_dir=args.save_dir)
    
    # Use the new --ext argument to find the correct files
    wildcard = f'*.{args.ext}'
    lora_videos = sorted([str(p) for p in Path(args.lora_dir).glob(wildcard)])
    
    if not lora_videos:
        print(f"FATAL ERROR: No files with extension '.{args.ext}' found in --lora_dir '{args.lora_dir}'")
        return

    if args.base_dir:
        # --- STYLE LORA MODE ---
        base_videos = sorted([str(p) for p in Path(args.base_dir).glob(wildcard)])
        if not base_videos:
             print(f"FATAL ERROR: No files with extension '.{args.ext}' found in --base_dir '{args.base_dir}'")
             return
        evaluator.run_comparison_evaluation(base_videos, lora_videos, args.lora_name)
    else:
        # --- CHARACTER LORA MODE ---
        evaluator.run_fidelity_evaluation(lora_videos, args.lora_name)

if __name__ == "__main__":
    main()