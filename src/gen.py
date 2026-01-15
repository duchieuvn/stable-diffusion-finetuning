"""
Stable Diffusion Image Generation with LoRA
"""
import os
import shutil
import torch
import yaml
from pathlib import Path
from datetime import datetime
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from peft import PeftModel


def load_yaml_config(config_path="generation_config.yaml"):
    """
    Load configuration from external YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    if not os.path.isabs(config_path):
        config_path = os.path.join(os.path.dirname(__file__), config_path)
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_config(yaml_config=None):
    """
    Load configuration settings for generation.
    
    Args:
        yaml_config: Loaded YAML configuration dictionary (optional)
        
    Returns:
        dict: Configuration dictionary with paths and settings
    """
    if yaml_config is None:
        yaml_config = load_yaml_config()
    
    return {
        "base_model": yaml_config["base_model"],
        "lora_weights_path": f"../runs/{yaml_config['folder_name']}",
        "output_folder": f"../generation/{yaml_config['folder_name']}",
        "default_prompt": yaml_config["default_prompt"],
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }


def load_model(config):
    """
    Load Stable Diffusion model with LoRA weights.
    
    """
    print("Loading base UNet model...")
    unet = UNet2DConditionModel.from_pretrained(
        config["base_model"],
        subfolder="unet",
        torch_dtype=torch.float16
    )
    
    print(f"Injecting LoRA weights from {config['lora_weights_path']}...")
    unet = PeftModel.from_pretrained(unet, config["lora_weights_path"])
    unet = unet.merge_and_unload()  # Merge for faster inference
    unet.to(config["device"])
    
    print("Creating Stable Diffusion pipeline...")
    pipeline = StableDiffusionPipeline.from_pretrained(
        config["base_model"],
        unet=unet,
        torch_dtype=torch.float16,
        use_safetensors=True,
        safety_checker=None
    )
    pipeline.to(config["device"])
    
    return pipeline


def sampling(pipeline, config, sample_prompts, num_images, negative_prompt, num_inference_steps, guidance_scale):
    """
    Generate images using the pipeline.
    
    Args:
        pipeline: Loaded StableDiffusionPipeline
        config: Configuration dictionary
        sample_prompts: List of prompts to generate images for
        num_images: Number of images to generate per prompt
        negative_prompt: Negative prompt string
        num_inference_steps: Number of inference steps
        guidance_scale: Guidance scale value
        
    Returns:
        list: Generated PIL images
    """
    output_folder = config["output_folder"]
    os.makedirs(output_folder, exist_ok=True)
    
    all_images = []
    
    # Generate images for each prompt
    for prompt_idx, prompt in enumerate(sample_prompts):
        print(f"\n[{prompt_idx + 1}/{len(sample_prompts)}] Generating {num_images} images...")
        print(f"Prompt: '{prompt}'")
        
        # Generate images
        pipeline_output = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images
        )
        
        images = pipeline_output.images
        all_images.extend(images)
        
        # Save images
        print(f"Saving images to {output_folder}/")
        filename_prefix = prompt.replace(" ", "-").replace(",", "")
        
        for i, img in enumerate(images):
            filename = os.path.join(output_folder, f"{filename_prefix}_{i}.png")
            img.save(filename)
            print(f"  -> Saved: {filename}")
    
    print("\n✅ Generation complete!")
    return all_images


def get_latest_subfolder(base_folder):
    """
    Get the latest timestamped subfolder in the base folder.
    
    Args:
        base_folder: Base output folder path
        
    Returns:
        str: Path to the latest subfolder, or None if no subfolders exist
    """
    if not os.path.exists(base_folder):
        return None
    
    subfolders = [
        os.path.join(base_folder, d) 
        for d in os.listdir(base_folder) 
        if os.path.isdir(os.path.join(base_folder, d))
    ]
    
    if not subfolders:
        return None
    
    # Sort by modification time (or folder name which is timestamp-based)
    latest_subfolder = max(subfolders, key=os.path.getmtime)
    return latest_subfolder


def compare_yaml_files(file1, file2):
    """
    Compare two YAML files for equality.
    
    Args:
        file1: Path to first YAML file
        file2: Path to second YAML file
        
    Returns:
        bool: True if files are identical, False otherwise
    """
    if not os.path.exists(file2):
        return False
    
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        config1 = yaml.safe_load(f1)
        config2 = yaml.safe_load(f2)
        return config1 == config2


def generate_folder_name_from_config(yaml_config):
    """
    Generate a descriptive folder name from the configuration.
    
    Args:
        yaml_config: Loaded YAML configuration dictionary
        
    Returns:
        str: Brief descriptive folder name
    """
    gen_params = yaml_config["generation"]
    
    # Extract key parameters
    num_images = gen_params["num_images"]
    steps = gen_params["num_inference_steps"]
    guidance = gen_params["guidance_scale"]
    num_prompts = len(yaml_config["sample_prompts"])
    
    # Create brief summary
    folder_name = f"{num_prompts}prompts_{num_images}imgs_s{steps}_g{guidance}"
    
    return folder_name


def main():
    """Main execution function."""
    # Load YAML configuration
    yaml_config = load_yaml_config()
    
    # Get sample prompts and generation parameters from YAML
    sample_prompts = yaml_config["sample_prompts"]
    generation_params = yaml_config["generation"]
    
    num_images = generation_params["num_images"]
    negative_prompt = generation_params["negative_prompt"]
    num_inference_steps = generation_params["num_inference_steps"]
    guidance_scale = generation_params["guidance_scale"]
    
    # Load configuration
    config = load_config(yaml_config)
    
    # Check latest subfolder and compare configs
    base_output_folder = config["output_folder"]
    config_source = os.path.join(os.path.dirname(__file__), "generation_config.yaml")
    
    latest_subfolder = get_latest_subfolder(base_output_folder)
    
    if latest_subfolder:
        latest_config = os.path.join(latest_subfolder, "generation_config.yaml")
        if compare_yaml_files(config_source, latest_config):
            # Configs are identical, reuse the latest subfolder
            config["output_folder"] = latest_subfolder
            print(f"✓ Config unchanged, reusing folder: {latest_subfolder}")
        else:
            # Configs differ, create new subfolder with descriptive name
            folder_name = generate_folder_name_from_config(yaml_config)
            config["output_folder"] = f"{base_output_folder}/{folder_name}"
            os.makedirs(config["output_folder"], exist_ok=True)
            shutil.copy(config_source, os.path.join(config["output_folder"], "generation_config.yaml"))
            print(f"✓ Config changed, created new folder: {config['output_folder']}")
    else:
        # No existing subfolders, create first one
        folder_name = generate_folder_name_from_config(yaml_config)
        config["output_folder"] = f"{base_output_folder}/{folder_name}"
        os.makedirs(config["output_folder"], exist_ok=True)
        shutil.copy(config_source, os.path.join(config["output_folder"], "generation_config.yaml"))
        print(f"✓ Created first folder: {config['output_folder']}")
    
    # Load model with LoRA weights
    pipeline = load_model(config)
    
    # Generate images
    sampling(
        pipeline=pipeline,
        config=config,
        sample_prompts=sample_prompts,
        num_images=num_images,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale
    )


if __name__ == "__main__":
    main()