#!/usr/bin/env python3
"""
Evaluate fine-tuned Stable Diffusion LoRA using LPIPS and FID metrics.

LPIPS: Measures perceptual similarity between generated and reference images
FID: Measures statistical distance between generated and real image distributions

Usage:
    python evaluate.py --lora_path path/to/lora --reference_dir path/to/dataset
"""

import argparse
import numpy as np
import torch
import yaml
from pathlib import Path
from PIL import Image
from diffusers import StableDiffusionPipeline
from pytorch_fid.fid_score import calculate_fid_given_paths

try:
    import lpips
except ImportError:
    print("ERROR: lpips not installed. Run: pip install lpips")
    exit(1)


def load_config(config_path="generation_config.yaml"):
    """Load generation config."""
    if Path(config_path).exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


def generate_images(lora_path, output_dir, config, num_images=50):
    """Generate images using trained LoRA model."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Stable Diffusion pipeline...")
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16
    ).to("cuda")

    print(f"Loading LoRA weights from {lora_path}...")
    try:
        pipe.unet.load_attn_procs(lora_path)
    except Exception as e:
        print(f"Warning: Could not load LoRA weights: {e}")

    # Use prompts from config or defaults
    prompts = config.get("prompts", [
        "photo of sks person in a blue shirt",
        "photo of sks person wearing glasses",
        "photo of sks person in a suit",
        "photo of sks person smiling",
        "full body photo of sks person standing",
        "portrait of sks person",
        "photo of sks person outdoors",
        "photo of sks person in casual clothing",
        "headshot of sks person",
        "photo of sks person with neutral expression",
    ])

    # Repeat prompts to reach desired number of images
    prompts = (prompts * (num_images // len(prompts) + 1))[:num_images]

    height = config.get("height", 512)
    width = config.get("width", 512)
    num_inference_steps = config.get("num_inference_steps", 50)
    guidance_scale = config.get("guidance_scale", 7.5)

    print(f"Generating {num_images} images...")
    for i, prompt in enumerate(prompts):
        try:
            image = pipe(
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            ).images[0]
            image.save(output_dir / f"gen_{i:03d}.png")
            
            if (i + 1) % 10 == 0:
                print(f"  Generated {i + 1}/{num_images} images")
        except Exception as e:
            print(f"  Warning: Failed to generate image {i}: {e}")

    print(f"✓ Generated images saved to {output_dir}")
    return list(output_dir.glob("*.png"))


def calculate_lpips(generated_dir, reference_dir, resize_to=512):
    """Calculate LPIPS between generated and reference images."""
    print("\nCalculating LPIPS metric...")
    
    loss_fn = lpips.LPIPS(net='alex', version='0.1')
    
    gen_images = sorted(Path(generated_dir).glob("*.png"))
    ref_images = sorted(Path(reference_dir).glob("*.jpg")) + \
                 sorted(Path(reference_dir).glob("*.jpeg")) + \
                 sorted(Path(reference_dir).glob("*.png"))
    
    if not gen_images:
        print("ERROR: No generated images found!")
        return None
    
    if not ref_images:
        print("ERROR: No reference images found!")
        return None
    
    # Match number of images
    min_len = min(len(gen_images), len(ref_images))
    gen_images = gen_images[:min_len]
    ref_images = ref_images[:min_len]
    
    lpips_scores = []
    
    print(f"  Comparing {len(gen_images)} image pairs...")
    for idx, (gen_path, ref_path) in enumerate(zip(gen_images, ref_images)):
        try:
            # Load and preprocess images
            gen_img = Image.open(gen_path).convert('RGB').resize((resize_to, resize_to))
            ref_img = Image.open(ref_path).convert('RGB').resize((resize_to, resize_to))
            
            # Convert to tensors and normalize to [-1, 1]
            gen_tensor = torch.tensor(np.array(gen_img)).float() / 127.5 - 1.0
            ref_tensor = torch.tensor(np.array(ref_img)).float() / 127.5 - 1.0
            
            # Reshape to batch format (B, C, H, W)
            gen_tensor = gen_tensor.unsqueeze(0).permute(0, 3, 1, 2).cuda()
            ref_tensor = ref_tensor.unsqueeze(0).permute(0, 3, 1, 2).cuda()
            
            # Compute LPIPS
            d = loss_fn.forward(gen_tensor, ref_tensor)
            lpips_scores.append(d.item())
            
            if (idx + 1) % 10 == 0:
                print(f"    Processed {idx + 1}/{len(gen_images)} pairs")
        except Exception as e:
            print(f"    Warning: Could not process pair {idx}: {e}")
    
    if not lpips_scores:
        return None
    
    avg_lpips = np.mean(lpips_scores)
    std_lpips = np.std(lpips_scores)
    
    print(f"✓ LPIPS calculation complete")
    print(f"  Average LPIPS: {avg_lpips:.4f} (±{std_lpips:.4f})")
    
    # Interpretation
    if avg_lpips < 0.15:
        quality = "EXCELLENT"
    elif avg_lpips < 0.25:
        quality = "GOOD"
    else:
        quality = "WEAK"
    
    print(f"  Quality: {quality} identity preservation")
    
    return avg_lpips


def calculate_fid(generated_dir, reference_dir, batch_size=50):
    """Calculate FID between generated and reference images."""
    print("\nCalculating FID metric...")
    
    gen_path = str(Path(generated_dir).absolute())
    ref_path = str(Path(reference_dir).absolute())
    
    try:
        fid_value = calculate_fid_given_paths(
            [gen_path, ref_path],
            batch_size=batch_size,
            device='cuda:0',
            dims=2048
        )
        
        print(f"✓ FID calculation complete")
        print(f"  FID Score: {fid_value:.4f}")
        
        # Interpretation
        if fid_value < 10:
            quality = "EXCELLENT"
        elif fid_value < 30:
            quality = "GOOD"
        elif fid_value < 50:
            quality = "MODERATE"
        else:
            quality = "POOR"
        
        print(f"  Quality: {quality} overall generation quality")
        
        return fid_value
    except Exception as e:
        print(f"ERROR calculating FID: {e}")
        return None


def save_results(output_dir, lpips_score, fid_score):
    """Save evaluation results to file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = output_dir / "evaluation_results.txt"
    
    with open(results_file, "w") as f:
        f.write("="*60 + "\n")
        f.write("LoRA Evaluation Results\n")
        f.write("="*60 + "\n\n")
        
        if lpips_score is not None:
            f.write(f"LPIPS (Identity Preservation): {lpips_score:.4f}\n")
            f.write(f"  ↳ Lower is better (< 0.20 = excellent)\n\n")
        else:
            f.write("LPIPS: Not calculated\n\n")
        
        if fid_score is not None:
            f.write(f"FID (Overall Quality): {fid_score:.4f}\n")
            f.write(f"  ↳ Lower is better (< 30 = good)\n\n")
        else:
            f.write("FID: Not calculated\n\n")
        
        f.write("="*60 + "\n")
        f.write("Interpretation Guide:\n")
        f.write("="*60 + "\n")
        f.write("LPIPS:\n")
        f.write("  < 0.15: Excellent identity preservation\n")
        f.write("  0.15-0.25: Good preservation\n")
        f.write("  > 0.25: Weak preservation\n\n")
        f.write("FID:\n")
        f.write("  < 10: Excellent quality\n")
        f.write("  10-30: Good quality\n")
        f.write("  30-50: Moderate quality\n")
        f.write("  > 50: Poor quality\n")
    
    print(f"\n✓ Results saved to {results_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned Stable Diffusion LoRA model"
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        required=True,
        help="Path to trained LoRA model"
    )
    parser.add_argument(
        "--reference_dir",
        type=str,
        required=True,
        help="Path to reference dataset directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation_results",
        help="Output directory for generated images and results"
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=50,
        help="Number of images to generate (default: 50)"
    )
    parser.add_argument(
        "--skip_generation",
        action="store_true",
        help="Skip image generation (use existing generated images)"
    )
    parser.add_argument(
        "--skip_lpips",
        action="store_true",
        help="Skip LPIPS calculation"
    )
    parser.add_argument(
        "--skip_fid",
        action="store_true",
        help="Skip FID calculation"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="generation_config.yaml",
        help="Path to generation config file"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    gen_dir = Path(args.output_dir) / "generated"
    
    # Step 1: Generate images
    if not args.skip_generation:
        generate_images(args.lora_path, gen_dir, config, args.num_images)
    else:
        print(f"Skipping image generation. Using images from {gen_dir}")
    
    # Step 2: Calculate LPIPS
    lpips_score = None
    if not args.skip_lpips:
        lpips_score = calculate_lpips(gen_dir, args.reference_dir)
    
    # Step 3: Calculate FID
    fid_score = None
    if not args.skip_fid:
        fid_score = calculate_fid(gen_dir, args.reference_dir)
    
    # Save results
    save_results(args.output_dir, lpips_score, fid_score)
    
    print("\n" + "="*60)
    print("Evaluation complete!")
    print("="*60)


if __name__ == "__main__":
    main()
