import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from diffusers import UNet2DConditionModel, DDPMScheduler, AutoencoderKL
from transformers import AutoTokenizer, CLIPTextModel
from peft import LoraConfig, get_peft_model
from dataset import BabyDataset


# ============================================================
# 2. Setup Configuration
# ============================================================
MODEL_NAME = "runwayml/stable-diffusion-v1-5"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 2  # Lower per-step memory
GRADIENT_ACCUMULATION_STEPS = 2  # Effective batch size = 4
LR = 1e-4
NUM_EPOCH = 50 # Adjust based on dataset size
OUTPUT_DIR = f"../runs/baby1_{NUM_EPOCH}epochs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 3. Load Models
# ============================================================
print(f"Loading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, subfolder="tokenizer")
noise_scheduler = DDPMScheduler.from_pretrained(MODEL_NAME, subfolder="scheduler")
text_encoder = CLIPTextModel.from_pretrained(MODEL_NAME, subfolder="text_encoder")  # stay on CPU
vae = AutoencoderKL.from_pretrained(MODEL_NAME, subfolder="vae")  # stay on CPU
unet = UNet2DConditionModel.from_pretrained(MODEL_NAME, subfolder="unet").to(DEVICE)

# Freeze core components
vae.requires_grad_(False)
text_encoder.requires_grad_(False)
unet.requires_grad_(False)

# Enable gradient checkpointing to reduce memory
unet.enable_gradient_checkpointing()

# ============================================================
# 4. Integrate PEFT (LoRA)
# ============================================================
# Identify the target modules in SD 1.5 UNet for LoRA injection.
# Usually we target the attention projection layers.
lora_config = LoraConfig(
    r=4,                  # Rank
    lora_alpha=4,         # Alpha (scaling factor)
    init_lora_weights="gaussian",
    target_modules=["to_k", "to_q", "to_v", "to_out.0"], # Targets both self and cross-attention
)

unet = get_peft_model(unet, lora_config)
unet.print_trainable_parameters()

# ============================================================
# 5. Optimizer & Loader
# ============================================================
optimizer = torch.optim.AdamW(unet.parameters(), lr=LR, weight_decay=0.01)

dataset = BabyDataset(dataset_path="../dataset/baby1")
train_dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
)

# ============================================================
# 6. Training Loop
# ============================================================
print("🚀 Starting PEFT LoRA Training...")
unet.train()
global_step = 0
progress_bar = tqdm(range(NUM_EPOCH))

while global_step < NUM_EPOCH:
    for step, batch in enumerate(train_dataloader):
        images, captions = batch
        images = images.to(DEVICE, non_blocking=True)

        # A. Encode Images to Latents (VAE temporarily on GPU)
        with torch.no_grad():
            vae.to(DEVICE)
            latents = vae.encode(images).latent_dist.sample()
            latents = latents * 0.18215  # Scaling factor for SD 1.5
            vae.to("cpu")
            torch.cuda.empty_cache()

        # B. Sample Noise & Timesteps
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=DEVICE).long()

        # C. Add Noise to Latents
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        # D. Get Text Embeddings (text encoder temporarily on GPU)
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        with torch.no_grad():
            text_encoder.to(DEVICE)
            encoder_hidden_states = text_encoder(inputs.input_ids.to(DEVICE))[0]
            text_encoder.to("cpu")
            torch.cuda.empty_cache()

        # E. Predict Noise (Forward Pass)
        model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

        # F. Loss & Backprop (with gradient accumulation)
        loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
        loss = loss / GRADIENT_ACCUMULATION_STEPS
        loss.backward()

        # G. Optimizer Step on accumulation boundary
        if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        progress_bar.update(1)
        progress_bar.set_description(f"Loss: {loss.item() * GRADIENT_ACCUMULATION_STEPS:.4f}")
        
        global_step += 1
        if global_step >= NUM_EPOCH:
            break

# ============================================================
# 7. Save LoRA Weights
# ============================================================
print(f"✅ Training done. Saving to {OUTPUT_DIR}...")

# This saves only the LoRA adapters, not the full UNet
unet.save_pretrained(OUTPUT_DIR) 

print("To use this LoRA:")
print(f"pipeline.load_lora_weights('{OUTPUT_DIR}')")