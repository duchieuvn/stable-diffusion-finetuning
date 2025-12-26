import os
from datetime import datetime
import shutil
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from diffusers import UNet2DConditionModel, DDPMScheduler, AutoencoderKL, StableDiffusionPipeline
from transformers import AutoTokenizer, CLIPTextModel
from peft import LoraConfig, get_peft_model, PeftModel
from dataset import JennieDataset

# To avoid warnings about parallelism
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# ============================================================ 
# 1. Configuration Setup
# ============================================================ 
def get_config(config_json=None):
    """
    Returns a dictionary of configuration parameters for the training.
    """
    if config_json is None:
        config = {
            "model_name": "SG161222/Realistic_Vision_V5.1_noVAE",
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "batch_size": 2,
            "gradient_accumulation_steps": 2,
            "lr": 1e-4,
            "num_epoch": 80,
            "output_dir_prefix": "../runs/jennie1",
            "sample_every_n_epochs": 10,
            "sample_prompts": ["J3NN13"],
            "lora_r": 32,
            "lora_alpha": 16,
            "lora_dropout": 0.1,
            "lora_target_modules": ["to_k", "to_q", "to_v", "to_out.0"],
            "vae_scaling_factor": 0.18215,
        }
    else:
        config = config_json

    date_str = datetime.now().strftime("%d-%m-%Y")
    config["output_dir"] = f"{config['output_dir_prefix']}_{config['num_epoch']}_r{config['lora_r']}a{config['lora_alpha']}_{date_str}"
    config["samples_dir"] = os.path.join(config["output_dir"], "samples")
    config["plots_dir"] = os.path.join(config["output_dir"], "plots")
    
    return config

# ============================================================ 
# 2. Model and Tokenizer Loading
# ============================================================ 
def load_models(config):
    """
    Loads all necessary models and the tokenizer from Hugging Face.
    """
    model_name = config['model_name']
    print(f"Loading models from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, subfolder="tokenizer")
    noise_scheduler = DDPMScheduler.from_pretrained(model_name, subfolder="scheduler")
    text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet").to(config['device'])

    # Freeze non-trainable components
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    
    # Enable gradient checkpointing for memory efficiency
    unet.enable_gradient_checkpointing()
    
    return tokenizer, noise_scheduler, text_encoder, vae, unet

# ============================================================ 
# 3. LoRA (PEFT) Integration
# ============================================================ 
def apply_lora(unet, config):
    """
    Applies LoRA to the UNet model for parameter-efficient fine-tuning.
    """
    lora_config = LoraConfig(
        r=config['lora_r'],
        lora_alpha=config['lora_alpha'],
        lora_dropout=config['lora_dropout'],
        init_lora_weights="gaussian",
        target_modules=config['lora_target_modules'],
    )
    
    lora_unet = get_peft_model(unet, lora_config)
    print("Trainable parameters after applying LoRA:")
    lora_unet.print_trainable_parameters()
    
    return lora_unet

# ============================================================ 
# 4. Data Loading
# ============================================================ 
def get_dataloader(config):
    """
    Creates the DataLoader for the training dataset.
    """
    dataset = JennieDataset(version='v2')
    return DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

# ============================================================ 
# 5. Sample Generation
# ============================================================ 
def generate_samples(unet_model, epoch, config):
    """
    Generates and saves sample images to monitor training progress using full precision.
    """
    print(f"\n🎨 Generating samples at epoch {epoch}...")
    
    unet_model.eval()
    
    # Create a temporary directory to save LoRA weights for clean inference
    temp_lora_dir = os.path.join(config['output_dir'], "temp_lora_for_sampling")
    unet_model.save_pretrained(temp_lora_dir)

    pipeline = None
    try:
        # Load a fresh pipeline in float16 precision
        pipeline = StableDiffusionPipeline.from_pretrained(
            config['model_name'],
            safety_checker=None,
            torch_dtype=torch.float16
        )
        pipeline.unet = PeftModel.from_pretrained(pipeline.unet, temp_lora_dir)
        pipeline.unet = pipeline.unet.merge_and_unload()
        pipeline.to(config['device'])

        epoch_dir = os.path.join(config['samples_dir'], f"epoch_{epoch:04d}")
        os.makedirs(epoch_dir, exist_ok=True)
        
        with torch.no_grad():
            for idx, prompt in enumerate(config['sample_prompts']):
                try:
                    image = pipeline(
                        prompt,
                        num_inference_steps=30,
                        guidance_scale=7.5,
                        generator=torch.Generator(device=config['device']).manual_seed(42)
                    ).images[0]
                    
                    image.save(os.path.join(epoch_dir, f"sample_{idx:02d}.png"))
                    with open(os.path.join(epoch_dir, f"sample_{idx:02d}_prompt.txt"), "w") as f:
                        f.write(prompt)
                    print(f"  ✓ Generated: {prompt[:50]}...")
                except Exception as e:
                    print(f"  ✗ Error generating sample for prompt '{prompt}': {e}")
    finally:
        # Clean up resources
        if pipeline is not None:
            del pipeline
        torch.cuda.empty_cache()
        if os.path.exists(temp_lora_dir):
            shutil.rmtree(temp_lora_dir)

    unet_model.train() # Restore model to training mode
    print(f"✓ Samples saved to {epoch_dir}\n")

# ============================================================ 
# 6. Training Metrics Saving & Plotting
# ============================================================ 
def save_training_metrics(all_losses, epoch_losses, plots_dir):
    """
    Saves training loss data to text files.
    """
    if all_losses:
        path = os.path.join(plots_dir, 'loss_per_iteration.txt')
        with open(path, 'w') as f:
            f.write('Iteration,Loss\n')
            for i, loss in enumerate(all_losses, 1):
                f.write(f'{i},{loss:.6f}\n')
        print(f"  ✓ Iteration-level loss data saved: {path}")

    if epoch_losses:
        path = os.path.join(plots_dir, 'loss_per_epoch.txt')
        with open(path, 'w') as f:
            f.write('Epoch,AvgLoss\n')
            for epoch, loss in enumerate(epoch_losses, 1):
                f.write(f'{epoch},{loss:.6f}\n')
        print(f"  ✓ Epoch-level loss data saved: {path}")

def plot_training_metrics(all_losses, epoch_losses, plots_dir):
    """
    Generates and saves plots for training loss.
    """
    if not all_losses or not epoch_losses:
        return
        
    print("\n📊 Generating training plots...")
    
    # Comprehensive plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Iteration loss
    axes[0].plot(range(1, len(all_losses) + 1), all_losses, linewidth=0.8, alpha=0.7, color='#2E86AB')
    axes[0].set(xlabel='Iteration', ylabel='Loss', title='Training Loss per Iteration')
    axes[0].grid(True, alpha=0.3)
    
    if len(all_losses) > 20:
        window = min(50, len(all_losses) // 10)
        smoothed = [sum(all_losses[max(0, i-window):i+1]) / min(window, i+1) for i in range(len(all_losses))]
        axes[0].plot(range(1, len(smoothed) + 1), smoothed, linewidth=2, color='#A23B72', label=f'Smoothed (window={window})')
        axes[0].legend()
    
    # Epoch loss
    axes[1].plot(range(1, len(epoch_losses) + 1), epoch_losses, linewidth=2, marker='o', color='#F18F01')
    axes[1].set(xlabel='Epoch', ylabel='Average Loss', title='Average Training Loss per Epoch')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'training_loss_comprehensive.png'), dpi=150)
    plt.close()

    # Separate high-res epoch plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, linewidth=2.5, marker='o', markersize=6, color='#F18F01')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Average Loss', fontsize=12)
    plt.title('Training Loss over Epochs', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'epoch_loss_high_res.png'), dpi=150)
    plt.close()
    
    print("  ✓ Training plots saved.")

# ============================================================ 
# 7. Main Training Loop
# ============================================================ 
def main(config_json=None):
    """
    Main function to orchestrate the LoRA fine-tuning process.
    """
    config = get_config(config_json)
    
    # Create directories
    os.makedirs(config['output_dir'], exist_ok=True)
    os.makedirs(config['samples_dir'], exist_ok=True)
    os.makedirs(config['plots_dir'], exist_ok=True)
    
    # Load components
    tokenizer, noise_scheduler, text_encoder, vae, unet = load_models(config)
    lora_unet = apply_lora(unet, config)
    train_dataloader = get_dataloader(config)
    optimizer = torch.optim.AdamW(lora_unet.parameters(), lr=config['lr'], weight_decay=0.01)
    
    # Training state
    all_losses, epoch_losses = [], []
    
    print("\n🚀 Starting LoRA Training...")
    lora_unet.train()

    for epoch in range(config['num_epoch']):
        epoch_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{config['num_epoch']}")
        
        for step, batch in enumerate(progress_bar):
            images, captions = batch
            images = images.to(config['device'], dtype=torch.float32, non_blocking=True)

            # Process batch
            with torch.no_grad():
                vae.to(config['device'])
                latents = vae.encode(images).latent_dist.sample() * config['vae_scaling_factor']
                vae.to("cpu")
                
                text_encoder.to(config['device'])
                input_ids = tokenizer(captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids
                encoder_hidden_states = text_encoder(input_ids.to(config['device']))[0]
                text_encoder.to("cpu")
                torch.cuda.empty_cache()

            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=config['device']).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            # Forward & backward pass
            model_pred = lora_unet(noisy_latents, timesteps, encoder_hidden_states).sample
            loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
            loss = loss / config['gradient_accumulation_steps']
            loss.backward()

            # Optimizer step
            if (step + 1) % config['gradient_accumulation_steps'] == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            # Loss tracking
            actual_loss = loss.item() * config['gradient_accumulation_steps']
            all_losses.append(actual_loss)
            epoch_loss += actual_loss
            progress_bar.set_postfix({"loss": f"{actual_loss:.4f}"})
        
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        epoch_losses.append(avg_epoch_loss)
        print(f"Epoch {epoch+1} Avg Loss: {avg_epoch_loss:.6f}")
        
        # Intermediate evaluation
        if (epoch + 1) % config['sample_every_n_epochs'] == 0:
            try:
                generate_samples(lora_unet, epoch + 1, config)
            except Exception as e:
                print(f"  ✗ Error generating samples: {e}")

    # ============================================================ 
    # 8. Save Final Model and Metrics
    # ============================================================ 
    print(f"\n✅ Training done. Saving artifacts to {config['output_dir']}...")
    lora_unet.save_pretrained(config['output_dir'])
    print(f"💾 LoRA weights saved.")
    
    save_training_metrics(all_losses, epoch_losses, config['plots_dir'])
    plot_training_metrics(all_losses, epoch_losses, config['plots_dir'])

    print("\n" + "="*60)
    print("🎉 Training Complete!")
    print(f"   - LoRA weights: {config['output_dir']}")
    print(f"   - Plots: {config['plots_dir']}")
    print(f"   - Samples: {config['samples_dir']}")
    print("\nTo use this LoRA:")
    print(f"  pipeline.load_lora_weights('{config['output_dir']}')")
    print("="*60)

if __name__ == "__main__":
    config = {
            "model_name": "SG161222/Realistic_Vision_V5.1_noVAE",
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "batch_size": 2,
            "gradient_accumulation_steps": 2,
            "lr": 5e-5,
            "num_epoch": 100,
            "output_dir_prefix": "../runs/jennie2",
            "sample_every_n_epochs": 2,
            "sample_prompts": ["J3NN13"],
            "lora_r": 32,
            "lora_alpha": 16,
            "lora_dropout": 0.1,
            "lora_target_modules": ["to_k", "to_q", "to_v", "to_out.0"],
            "vae_scaling_factor": 0.18215,
        }
    main(config)