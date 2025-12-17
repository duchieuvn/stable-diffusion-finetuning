import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false' # To avoid warnings

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

from diffusers import UNet2DConditionModel, DDPMScheduler, AutoencoderKL, StableDiffusionPipeline
from transformers import AutoTokenizer, CLIPTextModel
from peft import LoraConfig, get_peft_model
from dataset import BabyDataset, JennieDataset


# ============================================================
# 2. Setup Configuration
# ============================================================
MODEL_NAME = "runwayml/stable-diffusion-v1-5"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 2  # Lower per-step memory
GRADIENT_ACCUMULATION_STEPS = 2  # Effective batch size = 4
LR = 1e-4
NUM_EPOCH = 80 # Adjust based on dataset size

OUTPUT_DIR = f"../runs/jennie1_{NUM_EPOCH}_r32a16"
SAMPLES_DIR = os.path.join(OUTPUT_DIR, "samples")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SAMPLES_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Configuration for sample generation during training
SAMPLE_EVERY_N_EPOCHS = 10  # Generate samples every N epochs
SAMPLE_PROMPTS = [
    "a close up selfie of Jennie Backpink",
    "an upper body view of Jennie Blackpink",
]

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
    r=32,                  # Rank
    lora_alpha=16,        # Alpha
    lora_dropout=0.1,
    init_lora_weights="gaussian",
    target_modules=["to_k", "to_q", "to_v", "to_out.0"], # Targets both self and cross-attention
)

unet = get_peft_model(unet, lora_config)
unet.print_trainable_parameters()

# ============================================================
# 5. Optimizer & Loader
# ============================================================
optimizer = torch.optim.AdamW(unet.parameters(), lr=LR, weight_decay=0.01)

dataset = JennieDataset(version='v1')
train_dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
)

# ============================================================
# 5.5 Sample Generation Function
# ============================================================
def generate_samples(unet_model, epoch, prompts=SAMPLE_PROMPTS):
    """
    Generate sample images during training to monitor progress.
    """
    print(f"\n🎨 Generating samples at epoch {epoch}...")
    
    # Save current training state
    unet_model.eval()
    
    # Create a temporary merged UNet for inference (don't modify the original)
    with torch.no_grad():
        # Merge LoRA weights into a copy for inference
        merged_unet = unet_model.merge_and_unload()
        merged_unet = merged_unet.to(torch.float16)
        
        pipeline = StableDiffusionPipeline.from_pretrained(
            MODEL_NAME,
            unet=merged_unet,
            torch_dtype=torch.float16,
            safety_checker=None
        )
        pipeline.to(DEVICE)
        
        # Generate images for each prompt
        epoch_dir = os.path.join(SAMPLES_DIR, f"epoch_{epoch:04d}")
        os.makedirs(epoch_dir, exist_ok=True)
        
        for idx, prompt in enumerate(prompts):
            try:
                image = pipeline(
                    prompt,
                    num_inference_steps=30,
                    guidance_scale=7.5,
                    generator=torch.Generator(device=DEVICE).manual_seed(42)  # Fixed seed for comparison
                ).images[0]
                
                # Save the image
                image_path = os.path.join(epoch_dir, f"sample_{idx:02d}.png")
                image.save(image_path)
                
                # Also save with prompt as metadata
                with open(os.path.join(epoch_dir, f"sample_{idx:02d}_prompt.txt"), "w") as f:
                    f.write(prompt)
                    
                print(f"  ✓ Generated: {prompt[:50]}...")
            except Exception as e:
                print(f"  ✗ Error generating sample {idx}: {e}")
        
        # Clean up pipeline
        del pipeline
        del merged_unet
        torch.cuda.empty_cache()
    
    # Restore training mode
    unet_model.train()
    print(f"✓ Samples saved to {epoch_dir}\n")

# ============================================================
# 6. Training Loop
# ============================================================
print("🚀 Starting PEFT LoRA Training...")
unet.train()

# Track all losses for detailed plotting
all_losses = []  # Loss for every iteration
epoch_losses = []  # Average loss per epoch
iteration = 0

for epoch in range(NUM_EPOCH):
    epoch_loss = 0.0
    num_batches = 0
    
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCH}")
    
    for step, batch in enumerate(progress_bar):
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

        # Track loss at iteration level
        actual_loss = loss.item() * GRADIENT_ACCUMULATION_STEPS
        all_losses.append(actual_loss)
        epoch_loss += actual_loss
        num_batches += 1
        iteration += 1
        
        progress_bar.set_postfix({"loss": f"{actual_loss:.4f}", "avg_loss": f"{epoch_loss/num_batches:.4f}"})
    
    # Record average loss for this epoch
    avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
    epoch_losses.append(avg_epoch_loss)
    print(f"Epoch {epoch+1}/{NUM_EPOCH} - Average Loss: {avg_epoch_loss:.6f}")
    
    # # Generate samples at regular intervals
    # if epoch % SAMPLE_EVERY_N_EPOCHS == 0:
    #     try:
    #         generate_samples(unet, epoch + 1)
    #     except Exception as e:
    #         print(f"Warning: Failed to generate samples at epoch {epoch+1}: {e}")
    #         import traceback
    #         traceback.print_exc()

# ============================================================
# 7. Save LoRA Weights and Training Plots
# ============================================================
print(f"✅ Training done. Saving to {OUTPUT_DIR}...")

# This saves only the LoRA adapters, not the full UNet
unet.save_pretrained(OUTPUT_DIR) 
print(f"💾 LoRA weights saved to {OUTPUT_DIR}")

# ============================================================
# 8. Save Training Metrics and Plots
# ============================================================
print("\n📊 Generating training plots...")

# Save iteration-level loss data
if all_losses:
    iterations_data_path = os.path.join(PLOTS_DIR, 'loss_per_iteration.txt')
    with open(iterations_data_path, 'w') as f:
        f.write('Iteration,Loss\n')
        for i, loss in enumerate(all_losses, 1):
            f.write(f'{i},{loss:.6f}\n')
    print(f"  ✓ Iteration-level loss data saved: {iterations_data_path}")

# Save epoch-level loss data
if epoch_losses:
    epoch_data_path = os.path.join(PLOTS_DIR, 'loss_per_epoch.txt')
    with open(epoch_data_path, 'w') as f:
        f.write('Epoch,AvgLoss\n')
        for epoch, loss in enumerate(epoch_losses, 1):
            f.write(f'{epoch},{loss:.6f}\n')
    print(f"  ✓ Epoch-level loss data saved: {epoch_data_path}")

# Create comprehensive plots
if all_losses and epoch_losses:
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Loss per iteration
    axes[0].plot(range(1, len(all_losses) + 1), all_losses, linewidth=0.8, alpha=0.7, color='#2E86AB')
    axes[0].set_xlabel('Iteration', fontsize=11)
    axes[0].set_ylabel('Loss', fontsize=11)
    axes[0].set_title('Training Loss per Iteration', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Add smoothed line (moving average)
    if len(all_losses) > 20:
        window_size = min(50, len(all_losses) // 10)
        smoothed = [sum(all_losses[max(0, i-window_size):i+1]) / min(window_size, i+1) 
                   for i in range(len(all_losses))]
        axes[0].plot(range(1, len(smoothed) + 1), smoothed, linewidth=2, 
                    color='#A23B72', label=f'Smoothed (window={window_size})')
        axes[0].legend(loc='upper right')
    
    # Plot 2: Average loss per epoch
    axes[1].plot(range(1, len(epoch_losses) + 1), epoch_losses, 
                linewidth=2, marker='o', markersize=5, color='#F18F01')
    axes[1].set_xlabel('Epoch', fontsize=11)
    axes[1].set_ylabel('Average Loss', fontsize=11)
    axes[1].set_title('Average Training Loss per Epoch', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    loss_plot_path = os.path.join(PLOTS_DIR, 'training_loss.png')
    plt.savefig(loss_plot_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Training loss plots saved: {loss_plot_path}")
    plt.close()
    
    # Also create a separate high-res epoch plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, 
            linewidth=2.5, marker='o', markersize=6, color='#F18F01', 
            markerfacecolor='#FFD166', markeredgewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Average Loss', fontsize=12)
    plt.title('Training Loss over Epochs', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    epoch_plot_path = os.path.join(PLOTS_DIR, 'epoch_loss.png')
    plt.savefig(epoch_plot_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Epoch loss plot saved: {epoch_plot_path}")
    plt.close()

print("\n" + "="*60)
print(f"✅ Training Complete!")
print("="*60)
print(f"LoRA weights: {OUTPUT_DIR}")
print(f"Training plots: {PLOTS_DIR}")
print(f"Sample images: {SAMPLES_DIR}")
print("\nTo use this LoRA:")
print(f"  pipeline.load_lora_weights('{OUTPUT_DIR}')")
print("="*60)