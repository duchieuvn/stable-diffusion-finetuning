"""
Stable Diffusion Image Generation with LoRA
"""
import os
import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from peft import PeftModel


# Configuration presets
CONFIG_PRESETS = {
    "config1": {
        "folder_name": "lora_sd15_baby_peft",
        "default_prompt": "a photo of a baby"
    },
    "config2": {
        "folder_name": "baby1_50epochs",
        "default_prompt": "a photo of a baby crawling on the floor"
    },
    "config3": {
        "folder_name": "baby1_150_r32a16",
        "default_prompt": "a baby crawling on the floor in a room"
    }
}


def load_config(config_name="config1"):
    """
    Load configuration settings for generation.
    
    Args:
        config_name: Name of the config preset to use
        
    Returns:
        dict: Configuration dictionary with paths and settings
    """
    config = CONFIG_PRESETS.get(config_name, CONFIG_PRESETS["config1"])
    
    return {
        "base_model": "runwayml/stable-diffusion-v1-5",
        "lora_weights_path": f"../runs/{config['folder_name']}",
        "output_folder": f"../generation/{config['folder_name']}",
        "default_prompt": config["default_prompt"],
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


def sampling(pipeline, config):
    """
    Generate images using the pipeline.
    
    Args:
        pipeline: Loaded StableDiffusionPipeline
        config: Configuration dictionary
        
    Returns:
        list: Generated PIL images
    """
    
    prompt = config["default_prompt"]
    output_folder = config["output_folder"]
    os.makedirs(output_folder, exist_ok=True)
    num_images = 4
    
    print(f"Generating {num_images} images...")
    print(f"Prompt: '{prompt}'")
    negative_prompt = "ugly, tiling, ugly hands, ugly feet, " 
    # Generate images
    pipeline_output = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=100,
        guidance_scale=7.5,
        num_images_per_prompt=num_images
    )
    
    images = pipeline_output.images
    
    # Save images
    print(f"Saving images to {output_folder}/")
    filename_prefix = prompt.replace(" ", "-")
    
    for i, img in enumerate(images):
        filename = os.path.join(output_folder, f"{filename_prefix}_{i}.png")
        img.save(filename)
        print(f"  -> Saved: {filename}")
    
    print("✅ Generation complete!")
    return images


def main():
    """Main execution function."""
    # Load configuration
    config = load_config("config3")
    
    # Load model with LoRA weights
    pipeline = load_model(config)
    
    # Generate images
    sampling(
        pipeline=pipeline,
        config=config,
    )


if __name__ == "__main__":
    main()