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
# Make sure to import PIL at the top of your script
from PIL import Image
# Suppress unnecessary warnings from libraries
logging.getLogger('matplotlib').setLevel(logging.ERROR)


class LoRAVideoEvaluator:
    """Evaluator for comparing Base Model vs. LoRA-enhanced video generation."""
    
    def __init__(self, device='cuda', save_dir='evaluation_results'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        
        self.logger.info("Initializing evaluation models...")
        self.lpips_fn = lpips.LPIPS(net='alex').to(self.device)
        
        import torchvision.models as models
        # Ensure you have the new weights parameter for torchvision
        self.i3d_model = models.video.r3d_18(weights='R3D_18_Weights.DEFAULT').to(self.device)
        self.i3d_model.fc = nn.Identity()
        self.i3d_model.eval()
        self.logger.info("Evaluation models loaded.")
    
    # --- THIS IS THE CORRECTED FUNCTION ---
    def load_video(self, video_path: str) -> torch.Tensor:
        """Load video from .webp or .mp4 and convert to tensor."""
        # Check the file extension to use the correct loader
        if video_path.lower().endswith('.webp'):
            # Use Pillow for animated .webp files
            frames = []
            try:
                with Image.open(video_path) as img:
                    for i in range(img.n_frames):
                        img.seek(i)
                        frame_np = np.array(img.convert("RGB"))
                        # Convert from [H, W, C] numpy to [C, H, W] torch tensor
                        frames.append(torch.from_numpy(frame_np).permute(2, 0, 1))
                video = torch.stack(frames).float() # to [T, C, H, W]
                video = video.permute(1, 0, 2, 3)   # to [C, T, H, W]
            except Exception as e:
                self.logger.error(f"Pillow failed to load {video_path}: {e}")
                return torch.empty(0)
        else:
            # Use decord for standard video formats
            try:
                decord.bridge.set_bridge('torch')
                vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
                video = vr[:].float()
                video = video.permute(3, 0, 1, 2)  # To [C, T, H, W]
            except Exception as e:
                self.logger.error(f"Decord failed to load {video_path}: {e}")
                return torch.empty(0)

        # Normalize to [-1, 1]
        video = video / 127.5 - 1.0
        return video
    
    def compute_fvd(self, videos_base: List[torch.Tensor], videos_lora: List[torch.Tensor]) -> float:
        """Compute Fréchet Video Distance between two sets of videos."""
        def extract_features(videos):
            features = []
            with torch.no_grad():
                for video in tqdm(videos, desc="Extracting I3D Features"):
                    video = video.to(self.device)
                    C, T, H, W = video.shape
                    if T < 16:
                        video = F.pad(video, (0,0,0,0,0,16-T), "replicate")
                    
                    video_norm = (video + 1) / 2
                    mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(3, 1, 1, 1)
                    std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(3, 1, 1, 1)
                    video_norm = (video_norm - mean) / std
                    
                    feat = self.i3d_model(video_norm.unsqueeze(0)).squeeze().cpu().numpy()
                    features.append(feat)
            return np.array(features)

        self.logger.info("Extracting features from base videos...")
        features_base = extract_features(videos_base)
        self.logger.info("Extracting features from LoRA videos...")
        features_lora = extract_features(videos_lora)
        
        mu1, sigma1 = features_base.mean(0), np.cov(features_base, rowvar=False)
        mu2, sigma2 = features_lora.mean(0), np.cov(features_lora, rowvar=False)
        
        diff = mu1 - mu2
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if np.iscomplexobj(covmean): covmean = covmean.real
        
        fvd = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
        return float(fvd)
    
    def compute_ssim(self, video1: torch.Tensor, video2: torch.Tensor) -> float:
        """Compute average SSIM between two videos."""
        video1_np = ((video1.permute(1, 2, 3, 0).cpu().numpy() + 1) / 2.0).clip(0, 1)
        video2_np = ((video2.permute(1, 2, 3, 0).cpu().numpy() + 1) / 2.0).clip(0, 1)
        min_frames = min(video1_np.shape[0], video2_np.shape[0])
        return np.mean([ssim(video1_np[i], video2_np[i], channel_axis=2, data_range=1.0) for i in range(min_frames)])
    
    def compute_temporal_consistency(self, video: torch.Tensor) -> float:
        """Compute temporal consistency using optical flow variance."""
        video_np = np.clip((video.permute(1, 2, 3, 0).cpu().numpy() + 1) * 127.5, 0, 255).astype(np.uint8)
        flow_magnitudes = []
        for t in range(video_np.shape[0] - 1):
            frame1 = cv2.cvtColor(video_np[t], cv2.COLOR_RGB2GRAY)
            frame2 = cv2.cvtColor(video_np[t + 1], cv2.COLOR_RGB2GRAY)
            flow = cv2.calcOpticalFlowFarneback(frame1, frame2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            flow_magnitudes.append(np.mean(magnitude))
        return 1.0 / (1.0 + np.var(flow_magnitudes)) if flow_magnitudes else 1.0

        
    def batch_compare(self, base_videos: List[str], lora_videos: List[str]):
        """Compare multiple pairs of videos."""
        assert len(base_videos) == len(lora_videos), "Must have the same number of base and LoRA videos."
        
        all_base_tensors = [self.load_video(p) for p in tqdm(base_videos, desc="Loading Base Videos")]
        all_lora_tensors = [self.load_video(p) for p in tqdm(lora_videos, desc="Loading LoRA Videos")]
        
        self.logger.info("Computing overall FVD (Style Difference)...")
        overall_fvd = self.compute_fvd(all_base_tensors, all_lora_tensors)
        
        self.logger.info("Computing per-video metrics...")
        per_video_ssim = [self.compute_ssim(all_base_tensors[i], all_lora_tensors[i]) for i in range(len(all_base_tensors))]
        per_video_temp_base = [self.compute_temporal_consistency(t) for t in all_base_tensors]
        per_video_temp_lora = [self.compute_temporal_consistency(t) for t in all_lora_tensors]
        
        # --- THIS IS THE FIX ---
        # Convert all numpy numeric types to standard Python types before saving.
        aggregated = {
            'num_comparisons': int(len(base_videos)), # cast to int
            'fvd_style_difference': float(overall_fvd), # cast to float
            'mean_ssim_vs_base': float(np.mean(per_video_ssim)),
            'mean_temporal_consistency_base': float(np.mean(per_video_temp_base)),
            'mean_temporal_consistency_lora': float(np.mean(per_video_temp_lora)),
        }
        # --- END OF FIX ---
        
        output_file = self.save_dir / 'lora_evaluation_report.json'
        # This line will now work correctly
        with open(output_file, 'w') as f: json.dump(aggregated, f, indent=2)
        self.logger.info(f"Report saved to: {output_file}")
        
        self._generate_plots(aggregated)
        self._generate_verdict(aggregated)

    def _generate_plots(self, results: Dict):
        """Generate visualization plots for the LoRA evaluation."""
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # FVD Plot
        fvd = results['fvd_style_difference']
        fig, ax = plt.subplots(figsize=(8, 5))
        fvd_color = '#2ecc71' if fvd > 50 else '#f39c12'
        bars = ax.bar(['Style Difference'], [fvd], color=fvd_color)
        ax.text(bars[0].get_x() + bars[0].get_width()/2., fvd + 5, f'{fvd:.2f}', ha='center', fontweight='bold')
        ax.set_ylabel('FVD Score')
        ax.set_title('Fréchet Video Distance (Higher = More Stylistic Change)', fontsize=14, weight='bold')
        ax.grid(True, axis='y', alpha=0.5)
        plt.savefig(self.save_dir / 'fvd_style_difference.png', dpi=150)
        plt.close()
        
        # Comparison Plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        ssim = results['mean_ssim_vs_base']
        axes[0].bar(['Structural Similarity'], [ssim], color='#3498db')
        axes[0].set_ylim(0, 1)
        axes[0].text(0, ssim + 0.02, f'{ssim:.2%}', ha='center', fontweight='bold', fontsize=12)
        axes[0].set_title('Content Preservation (SSIM)', fontsize=14, weight='bold')
        axes[0].set_ylabel('Similarity to Base (Higher is Better)')
        
        temp_base = results['mean_temporal_consistency_base']
        temp_lora = results['mean_temporal_consistency_lora']
        axes[1].bar(['Base Model', 'With LoRA'], [temp_base, temp_lora], color=['#95a5a6', '#9b59b6'])
        axes[1].set_ylim(0, 1)
        axes[1].set_title('Temporal Consistency (Motion Stability)', fontsize=14, weight='bold')
        axes[1].set_ylabel('Consistency Score (Higher is Better)')
        
        fig.suptitle('LoRA Quality Assessment', fontsize=18, weight='bold')
        plt.tight_layout()
        plt.savefig(self.save_dir / 'lora_quality_metrics.png', dpi=150)
        plt.close()

        self.logger.info(f"Plots saved to: {self.save_dir}")

    def _generate_verdict(self, results: Dict):
        """Generate a text summary of the LoRA's performance."""
        fvd = results['fvd_style_difference']
        ssim = results['mean_ssim_vs_base']
        temp_lora = results['mean_temporal_consistency_lora']
        temp_base = results['mean_temporal_consistency_base']

        verdict = "### LoRA Performance Verdict\n\n"
        
        # FVD assessment
        if fvd > 100:
            verdict += f"✅ **Strong Stylistic Impact:** The FVD score of {fvd:.2f} is high, indicating the LoRA successfully created a significantly different visual style compared to the base model.\n"
        elif fvd > 50:
            verdict += f"✅ **Noticeable Stylistic Impact:** An FVD score of {fvd:.2f} shows a clear change in style.\n"
        else:
            verdict += f"⚠️ **Subtle Stylistic Impact:** An FVD score of {fvd:.2f} suggests the LoRA's effect is very subtle. Consider increasing LoRA strength or retraining.\n"

        # SSIM assessment
        if ssim > 0.8:
            verdict += f"✅ **Excellent Content Preservation:** The SSIM of {ssim:.2%} shows that the LoRA preserved the core structure and content of the original generation very well.\n"
        elif ssim > 0.7:
            verdict += f"✅ **Good Content Preservation:** An SSIM of {ssim:.2%} indicates the core content is mostly preserved.\n"
        else:
            verdict += f"⚠️ **Content Drift:** An SSIM of {ssim:.2%} suggests the LoRA is significantly altering the scene's content, which may be a sign of overfitting.\n"

        # Temporal Consistency assessment
        if temp_lora >= temp_base * 0.95:
             verdict += f"✅ **Excellent Motion Stability:** The LoRA maintained or improved temporal consistency ({temp_lora:.3f} vs base {temp_base:.3f}), resulting in smooth motion.\n"
        else:
            verdict += f"⚠️ **Reduced Motion Stability:** The LoRA slightly reduced temporal consistency ({temp_lora:.3f} vs base {temp_base:.3f}). This may result in minor flickering or jitter.\n"
        
        print("\n" + "="*50 + "\n" + verdict + "="*50)
        with open(self.save_dir / 'verdict.md', 'w') as f:
            f.write(verdict)


def main():
    parser = argparse.ArgumentParser(description='Compare Base vs. LoRA video generation')
    parser.add_argument('--base', type=str, nargs='+', required=True, help='Path(s) to base model videos')
    parser.add_argument('--lora', type=str, nargs='+', required=True, help='Path(s) to LoRA model videos')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    parser.add_argument('--save_dir', type=str, default='evaluation_results', help='Directory to save results')
    args = parser.parse_args()
    
    evaluator = LoRAVideoEvaluator(device=args.device, save_dir=args.save_dir)
    evaluator.batch_compare(args.base, args.lora)

if __name__ == "__main__":
    main()