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

# ============================================================
# 1. Dataset (BabyDataset)
# ============================================================
class BabyDataset(torch.utils.data.Dataset):
    def __init__(self, split="train", dataset_path="baby_dataset"):
        self.dataset_path = dataset_path
        self.image_dir = f"{dataset_path}/baby"
        
        # Standard SD 1.5 normalization
        self.resize = transforms.Compose([
            transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]), # Map to [-1, 1]
        ])
        
        # Load filenames
        split_file = f"{dataset_path}/baby_{split}.txt"
        if not os.path.exists(split_file):
            # Fallback if text file doesn't exist: list dir
            self.image_files = [f for f in os.listdir(self.image_dir) if f.endswith(('.jpg', '.png'))]
        else:
            with open(split_file, "r") as f:
                self.image_files = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        img = Image.open(img_path).convert("RGB")
        img = self.resize(img)
        
        # You can randomize prompts here if you have a metadata file
        caption = "a photo of a baby" 
        return img, caption

# ============================================================
# 2. Setup Configuration
# ============================================================
MODEL_NAME = "runwayml/stable-diffusion-v1-5"
OUTPUT_DIR = "lora_sd15_baby_peft"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 2 # Higher if you have VRAM
LR = 1e-4
NUM_EPOCH = 30 # Adjust based on dataset size

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 3. Load Models
# ============================================================
print(f"Loading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, subfolder="tokenizer")
noise_scheduler = DDPMScheduler.from_pretrained(MODEL_NAME, subfolder="scheduler")
text_encoder = CLIPTextModel.from_pretrained(MODEL_NAME, subfolder="text_encoder").to(DEVICE)
vae = AutoencoderKL.from_pretrained(MODEL_NAME, subfolder="vae").to(DEVICE)
unet = UNet2DConditionModel.from_pretrained(MODEL_NAME, subfolder="unet").to(DEVICE)

# Freeze core components
vae.requires_grad_(False)
text_encoder.requires_grad_(False)
unet.requires_grad_(False)

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
optimizer = torch.optim.AdamW(unet.parameters(), lr=LR)

dataset = BabyDataset(split="train", dataset_path="baby_dataset")
train_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

# ============================================================
# 6. Training Loop
# ============================================================
print("🚀 Starting PEFT LoRA Training...")
unet.train()
global_step = 0
progress_bar = tqdm(range(NUM_EPOCH))

while global_step < NUM_EPOCH:
    for batch in train_dataloader:
        images, captions = batch
        images = images.to(DEVICE)

        # A. Encode Images to Latents
        with torch.no_grad():
            latents = vae.encode(images).latent_dist.sample()
            latents = latents * 0.18215 # Scaling factor for SD 1.5

        # B. Sample Noise & Timesteps
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=DEVICE).long()

        # C. Add Noise to Latents
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        # D. Get Text Embeddings
        # We tokenize the captions dynamically
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        with torch.no_grad():
            encoder_hidden_states = text_encoder(inputs.input_ids.to(DEVICE))[0]

        # E. Predict Noise (Forward Pass)
        # Note: 'unet' is now a PeftModel, but behaves like the original unet
        model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

        # F. Loss & Backprop
        loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        progress_bar.update(1)
        progress_bar.set_description(f"Loss: {loss.item():.4f}")
        
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