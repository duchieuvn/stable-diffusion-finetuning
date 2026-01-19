"""
Evaluation Script for Generated Images
Calculates LPIPS and FID metrics on generated samples
Standalone script - no dependencies on gen_full.py
"""
import os
import json
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from typing import Dict, List, Optional
import yaml
import argparse

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("⚠️  lpips not installed. Install with: pip install lpips")

try:
    from pytorch_fid.fid_score import calculate_fid_given_paths
    FID_AVAILABLE = True
except ImportError:
    FID_AVAILABLE = False
    print("⚠️  pytorch-fid not installed. Install with: pip install pytorch-fid")


def load_yaml_config(config_path="generation_config.yaml"):
    """Load configuration from external YAML file."""
    if not os.path.isabs(config_path):
        config_path = os.path.join(os.path.dirname(__file__), config_path)
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_reference_dataset_path(lora_folder_name: str, dataset_root: str = "../dataset") -> Optional[str]:
    """
    Get the static reference dataset path for jennie_v3 images.
    
    Args:
        lora_folder_name: Name of the LoRA folder (ignored, kept for compatibility)
        dataset_root: Root dataset directory
        
    Returns:
        Path to the reference dataset images (jennie_v3)
    """
    if not os.path.isabs(dataset_root):
        dataset_root = os.path.join(os.path.dirname(__file__), dataset_root)
    
    # Static reference to jennie_v3 images
    ref_path = os.path.join(dataset_root, 'jennie_v3', 'images')
    
    if os.path.exists(ref_path):
        return ref_path
    return None


def load_images_batch(image_dir: str, max_images: Optional[int] = None, resize_to: tuple = (512, 512)):
    """
    Load images from directory and return as normalized tensors.
    
    Args:
        image_dir: Directory containing images
        max_images: Maximum number of images to load
        resize_to: Target resolution (height, width)
        
    Returns:
        Batch of normalized image tensors on GPU
    """
    images = []
    paths = sorted(Path(image_dir).glob("*.png"))
    
    if max_images:
        paths = paths[:max_images]
    
    for path in paths:
        try:
            img = Image.open(path).convert('RGB').resize(resize_to)
            tensor = torch.tensor(np.array(img)).float() / 127.5 - 1.0
            tensor = tensor.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
            images.append(tensor)
        except Exception as e:
            print(f"⚠️  Error loading {path}: {e}")
            continue
    
    if not images:
        return None
    
    return torch.stack(images).cuda()


def calculate_lpips(gen_dir: str, ref_dir: str, max_images: Optional[int] = None) -> Dict:
    """
    Calculate LPIPS (Learned Perceptual Image Patch Similarity) score.
    
    Args:
        gen_dir: Directory containing generated images
        ref_dir: Directory containing reference images
        max_images: Maximum number of images to use
        
    Returns:
        Dictionary with LPIPS metrics
    """
    if not LPIPS_AVAILABLE:
        return {"error": "LPIPS not installed. Install with: pip install lpips"}
    
    print("  🔍 Calculating LPIPS...")
    
    # Initialize model and move to GPU
    loss_fn = lpips.LPIPS(net='alex', version='0.1')
    loss_fn = loss_fn.cuda()
    loss_fn.eval()
    
    lpips_scores = []
    
    gen_paths = sorted(Path(gen_dir).glob("*.png"))
    ref_paths = sorted(Path(ref_dir).glob("*.jpg")) + sorted(Path(ref_dir).glob("*.png"))
    
    if max_images:
        gen_paths = gen_paths[:max_images]
        ref_paths = ref_paths[:max_images]
    
    # Match counts
    min_count = min(len(gen_paths), len(ref_paths))
    gen_paths = gen_paths[:min_count]
    ref_paths = ref_paths[:min_count]
    
    if min_count == 0:
        return {"error": "No images found in reference or generated directory"}
    
    print(f"    Processing {min_count} image pairs...")
    
    # Disable gradient computation for inference
    with torch.no_grad():
        for i, (gen_p, ref_p) in enumerate(zip(gen_paths, ref_paths)):
            try:
                gen_img = Image.open(gen_p).convert('RGB').resize((512, 512))
                ref_img = Image.open(ref_p).convert('RGB').resize((512, 512))
                
                gen_t = torch.tensor(np.array(gen_img)).float() / 127.5 - 1.0
                ref_t = torch.tensor(np.array(ref_img)).float() / 127.5 - 1.0
                
                gen_t = gen_t.unsqueeze(0).permute(0, 3, 1, 2).cuda()
                ref_t = ref_t.unsqueeze(0).permute(0, 3, 1, 2).cuda()
                
                d = loss_fn.forward(gen_t, ref_t)
                lpips_scores.append(d.item())
                
                # Progress indicator
                if (i + 1) % 10 == 0:
                    print(f"    ✓ Processed {i + 1}/{min_count} pairs")
            except Exception as e:
                print(f"    ⚠️  Error comparing {gen_p.name} with {ref_p.name}: {e}")
                continue
    
    if not lpips_scores:
        return {"error": "No valid image pairs could be processed"}
    
    avg_lpips = np.mean(lpips_scores)
    std_lpips = np.std(lpips_scores)
    
    return {
        "score": float(avg_lpips),
        "std": float(std_lpips),
        "num_pairs": len(lpips_scores),
        "interpretation": get_lpips_interpretation(avg_lpips)
    }


def calculate_fid(gen_dir: str, ref_dir: str) -> Dict:
    """
    Calculate FID (Fréchet Inception Distance) score.
    
    Args:
        gen_dir: Directory containing generated images
        ref_dir: Directory containing reference images
        
    Returns:
        Dictionary with FID metrics
    """
    if not FID_AVAILABLE:
        return {"error": "pytorch-fid not installed. Install with: pip install pytorch-fid"}
    
    print("  🔍 Calculating FID...")
    
    try:
        fid_value = calculate_fid_given_paths(
            [gen_dir, ref_dir],
            batch_size=50,
            device='cuda:0',
            dims=2048
        )
        
        return {
            "score": float(fid_value),
            "interpretation": get_fid_interpretation(fid_value)
        }
    except Exception as e:
        return {"error": f"FID calculation failed: {str(e)}"}


def get_lpips_interpretation(score: float) -> str:
    """Provide interpretation of LPIPS score."""
    if score < 0.15:
        return "Excellent identity preservation"
    elif score < 0.25:
        return "Good identity preservation"
    elif score < 0.35:
        return "Moderate identity preservation"
    else:
        return "Weak identity preservation"


def get_fid_interpretation(score: float) -> str:
    """Provide interpretation of FID score."""
    if score < 10:
        return "Excellent quality"
    elif score < 30:
        return "Good quality"
    elif score < 50:
        return "Moderate quality"
    else:
        return "Poor quality"


def extract_base_model_from_folder(gen_folder: str) -> str:
    """
    Extract base model information from folder name.
    
    Args:
        gen_folder: Path to generation folder
        
    Returns:
        Base model name
    """
    folder_name = os.path.basename(gen_folder)
    
    if folder_name.startswith("sd15_"):
        return "runwayml/stable-diffusion-v1-5"
    else:
        return "SG161222/Realistic_Vision_V5.1_noVAE"


def evaluate_generation(gen_folder: str, lora_folder_name: str) -> Optional[Dict]:
    """
    Comprehensive evaluation of generated images.
    
    Args:
        gen_folder: Path to folder with generated images
        lora_folder_name: Name of the LoRA folder
        
    Returns:
        Dictionary with evaluation metrics, or None if evaluation failed
    """
    print(f"\n{'='*80}")
    print(f"📊 Evaluating Generation")
    print(f"{'='*80}")
    print(f"Location: {gen_folder}")
    print(f"LoRA: {lora_folder_name}")
    
    # Infer base model from folder name
    base_model = extract_base_model_from_folder(gen_folder)
    print(f"Base Model: {base_model}")
    
    # Get reference dataset
    ref_dir = get_reference_dataset_path(lora_folder_name)
    if not ref_dir or not os.path.exists(ref_dir):
        print(f"❌ Reference dataset not found for {lora_folder_name}")
        print(f"   Expected: {ref_dir}")
        return None
    
    # Check if generated images exist
    gen_images = list(Path(gen_folder).glob("*.png"))
    if not gen_images:
        print(f"❌ No generated images (.png) found in {gen_folder}")
        return None
    
    print(f"Generated Images: {len(gen_images)}")
    print(f"Reference Dataset: {ref_dir}")
    
    results = {
        "lora_folder": lora_folder_name,
        "base_model": base_model,
        "gen_folder": gen_folder,
        "reference_dir": ref_dir,
        "num_generated_images": len(gen_images),
        "timestamp": str(Path(gen_folder).stat().st_mtime)
    }
    
    # Calculate metrics
    if LPIPS_AVAILABLE:
        lpips_results = calculate_lpips(gen_folder, ref_dir)
        results["lpips"] = lpips_results
    
    if FID_AVAILABLE:
        fid_results = calculate_fid(gen_folder, ref_dir)
        results["fid"] = fid_results
    
    return results


def print_results(results: Dict):
    """Pretty print evaluation results."""
    if not results:
        return
    
    print(f"\n{'='*80}")
    print(f"📈 Evaluation Results")
    print(f"{'='*80}")
    print(f"LoRA: {results['lora_folder']}")
    print(f"Base Model: {results['base_model']}")
    print(f"Generated Images: {results['num_generated_images']}")
    print(f"Reference Dataset: {results['reference_dir']}")
    print()
    
    if 'lpips' in results:
        lpips_data = results['lpips']
        if 'error' in lpips_data:
            print(f"⚠️  LPIPS: {lpips_data['error']}")
        else:
            print(f"LPIPS Score: {lpips_data['score']:.4f} ± {lpips_data['std']:.4f}")
            print(f"  └─ Pairs: {lpips_data['num_pairs']}")
            print(f"  └─ {lpips_data['interpretation']}")
            print(f"  └─ Range: 0 (identical) to 1 (different)")
    
    if 'fid' in results:
        fid_data = results['fid']
        if 'error' in fid_data:
            print(f"⚠️  FID: {fid_data['error']}")
        else:
            print(f"FID Score: {fid_data['score']:.4f}")
            print(f"  └─ {fid_data['interpretation']}")
            print(f"  └─ Lower is better (< 30 = good quality)")
    
    print(f"{'='*80}\n")


def save_results(results: Dict, output_file: str, metrics_root: str = "../metrics"):
    """Save evaluation results to metrics directory."""
    if not os.path.isabs(metrics_root):
        metrics_root = os.path.join(os.path.dirname(__file__), metrics_root)
    
    # Extract relative path from output_file to maintain structure
    lora_folder = results['lora_folder']
    gen_folder = results['gen_folder']
    
    # Create a structured path in metrics directory
    # Extract the meaningful part of gen_folder (subfolder name if it exists)
    gen_folder_name = os.path.basename(gen_folder)
    metrics_dir = os.path.join(metrics_root, lora_folder, gen_folder_name)
    
    os.makedirs(metrics_dir, exist_ok=True)
    
    metrics_file = os.path.join(metrics_dir, "evaluation_metrics.json")
    with open(metrics_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✅ Results saved to {metrics_file}")


def evaluate_all_generations(testgen_root: str = "../testgen", metrics_root: str = "../metrics") -> List[Dict]:
    """
    Evaluate all generated images in the testgen directory.
    
    Args:
        testgen_root: Root directory containing generated images
        metrics_root: Root directory for saving metrics
        
    Returns:
        List of evaluation results for each generation
    """
    if not os.path.isabs(testgen_root):
        testgen_root = os.path.join(os.path.dirname(__file__), testgen_root)
    
    if not os.path.exists(testgen_root):
        print(f"❌ testgen directory not found: {testgen_root}")
        return []
    
    all_results = []
    
    print(f"\n{'='*80}")
    print(f"🔍 Scanning for generations in: {testgen_root}")
    print(f"{'='*80}\n")
    
    # Find all generated image folders
    for lora_folder in sorted(Path(testgen_root).glob("*")):
        if not lora_folder.is_dir():
            continue
        
        lora_name = lora_folder.name
        
        # Check for subfolders (config-based subfolders)
        subfolders = [d for d in lora_folder.iterdir() if d.is_dir()]
        
        if subfolders:
            # Evaluate each subfolder
            for subfolder in sorted(subfolders):
                results = evaluate_generation(str(subfolder), lora_name)
                if results:
                    all_results.append(results)
                    print_results(results)
                    
                    # Save individual results to metrics directory
                    save_results(results, str(subfolder), metrics_root)
        else:
            # Evaluate the folder directly
            results = evaluate_generation(str(lora_folder), lora_name)
            if results:
                all_results.append(results)
                print_results(results)
                
                # Save results to metrics directory
                save_results(results, str(lora_folder), metrics_root)
    
    return all_results


def evaluate_specific_folder(gen_folder: str, lora_name: str = None, metrics_root: str = "../metrics") -> Optional[Dict]:
    """
    Evaluate a specific generation folder.
    
    Args:
        gen_folder: Path to the generation folder
        lora_name: Optional name of the LoRA (auto-detected if not provided)
        metrics_root: Root directory for saving metrics
        
    Returns:
        Evaluation results
    """
    if not os.path.exists(gen_folder):
        print(f"❌ Folder not found: {gen_folder}")
        return None
    
    # Auto-detect lora name from parent or folder structure
    if not lora_name:
        parent = Path(gen_folder).parent.name
        if parent in ['testgen', 'generation']:
            lora_name = Path(gen_folder).parent.parent.name
        else:
            lora_name = parent
    
    results = evaluate_generation(gen_folder, lora_name)
    if results:
        print_results(results)
        
        # Save results to metrics directory
        save_results(results, gen_folder, metrics_root)
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate generated images using LPIPS and FID metrics"
    )
    parser.add_argument(
        "--folder",
        type=str,
        default=None,
        help="Specific generation folder to evaluate (default: evaluate all in testgen/)"
    )
    parser.add_argument(
        "--testgen-root",
        type=str,
        default="../testgen",
        help="Root testgen directory (default: ../testgen)"
    )
    parser.add_argument(
        "--lora-name",
        type=str,
        default=None,
        help="LoRA folder name (auto-detected if not provided)"
    )
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print(f"🎨 Stable Diffusion Generation Evaluation")
    print(f"{'='*80}")
    
    if args.folder:
        # Evaluate specific folder
        results = evaluate_specific_folder(args.folder, args.lora_name)
        if results:
            print(f"\n✅ Evaluation complete!")
        else:
            print(f"\n❌ Evaluation failed")
    else:
        # Evaluate all generations
        all_results = evaluate_all_generations(args.testgen_root)
        
        if all_results:
            print(f"\n{'='*80}")
            print(f"✅ Evaluation Summary")
            print(f"{'='*80}")
            print(f"Total evaluations: {len(all_results)}")
            print(f"{'='*80}\n")
        else:
            print(f"\n❌ No generations found or no valid evaluations completed")