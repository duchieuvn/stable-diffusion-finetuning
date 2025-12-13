import torch
import os
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from peft import PeftModel

# ============================================================
# 1. Configuration
# ============================================================
BASE_MODEL = "runwayml/stable-diffusion-v1-5"
LORA_WEIGHTS_PATH = "lora_sd15_baby_peft"
OUTPUT_FOLDER = "generation"  # Folder to store results
NUM_IMAGES_TO_GENERATE = 4       # <-- How many images you want
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Create the output folder if it doesn't exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ============================================================
# 2. Load UNet and inject LoRA (Same method as before)
# ============================================================
print("1. Loading base UNet...")
unet = UNet2DConditionModel.from_pretrained(
    BASE_MODEL, 
    subfolder="unet", 
    torch_dtype=torch.float16
)

print(f"2. Injecting LoRA weights from {LORA_WEIGHTS_PATH}...")
unet = PeftModel.from_pretrained(unet, LORA_WEIGHTS_PATH)
# Merge weights for faster inference speed
unet = unet.merge_and_unload()
unet.to(DEVICE)

# ============================================================
# 3. Create Pipeline
# ============================================================
print("3. Creating Stable Diffusion pipeline...")
pipe = StableDiffusionPipeline.from_pretrained(
    BASE_MODEL,
    unet=unet,  # Use our LoRA-merged UNet
    torch_dtype=torch.float16,
    use_safetensors=True,
    safety_checker=None # Optional: disables safety checker to save VRAM
)
pipe.to(DEVICE)

# Important for saving VRAM when generating many images
pipe.enable_vae_tiling() 
# If you still run out of memory, uncomment the next line:
# pipe.enable_model_cpu_offload() 

# ============================================================
# 4. Run Batch Generation
# ============================================================
prompt = ""
negative_prompt = "bad quality, blurry, distorted, ugly, deformed hands"

print(f"4. Generating {NUM_IMAGES_TO_GENERATE} images... This might take a moment.")

# We use num_images_per_prompt to generate multiple images
pipeline_output = pipe(
    prompt=prompt, 
    # negative_prompt=negative_prompt,
    num_inference_steps=100,
    guidance_scale=7.5,
    num_images_per_prompt=NUM_IMAGES_TO_GENERATE  # <-- THE KEY CHANGE
)

# The resulting list of PIL images
images_list = pipeline_output.images

# ============================================================
# 5. Save Images Loop
# ============================================================
print(f"5. Saving images to folder: '{OUTPUT_FOLDER}/' ...")

for i, img in enumerate(images_list):
    # Generate a filename like batch_results/baby_0.png, baby_1.png, etc.
    filename = os.path.join(OUTPUT_FOLDER, f"baby_generated_{i}.png")
    img.save(filename)
    print(f"   -> Saved: {filename}")

print("✅ Batch generation complete.")