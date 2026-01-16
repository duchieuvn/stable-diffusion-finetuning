# Evaluating SD 1.5 LoRA Quality Using LPIPS and FID Metrics

After training your LoRA, you should evaluate the quality of generated images using LPIPS (perceptual similarity to training data) and FID (overall generation quality relative to a reference dataset). Here's the complete workflow.

## Overview of Metrics

**LPIPS (Learned Perceptual Image Patch Similarity):**

- Measures perceptual similarity between generated images and reference images
- Lower scores = more similar to reference (better identity preservation)
- Range: 0 (identical) to 1 (completely different)
- Best for: Evaluating identity fidelity in person/celebrity LoRAs[^1]

**FID (Fréchet Inception Distance):**

- Measures statistical distance between generated and real image distributions
- Lower scores = better quality (typically 0-100, lower is better)
- Range: 0 (identical distributions) to higher values (poor quality)
- Best for: Overall generation quality and diversity[^2]

## Step 1: Generate Images from Trained LoRA

Create a set of generated images using your trained LoRA for evaluation:

```python
from diffusers import StableDiffusionPipeline
import torch
from pathlib import Path

# Load base model with trained LoRA
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

# Load your trained LoRA weights
lora_path = "path/to/lora_output"
pipe.unet.load_attn_procs(lora_path)

# Generate evaluation images
output_dir = Path("generated_images/")
output_dir.mkdir(exist_ok=True)

prompts = [
    "photo of sks person in a blue shirt",
    "photo of sks person wearing glasses",
    "photo of sks person in a suit",
    "photo of sks person smiling",
    "full body photo of sks person standing",
    # ... repeat 20-50 times with varied prompts
]

for i, prompt in enumerate(prompts):
    image = pipe(
        prompt=prompt,
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=7.5
    ).images[^0]
    image.save(f"{output_dir}/generated_{i:03d}.png")

print(f"✓ Generated {len(prompts)} images")
```

Generate **20-100 diverse images** with varied prompts to ensure statistical validity of metrics.

## Step 2: Calculate LPIPS Score

LPIPS measures how perceptually similar your generated images are to training reference images (identity preservation).

### Installation

```bash
pip install lpips
```

### Method A: Compare Generated vs. Training Images (Recommended)

```python
import lpips
import torch
from PIL import Image
from pathlib import Path

# Initialize LPIPS model (AlexNet is fastest and best for forward scoring)
loss_fn = lpips.LPIPS(net='alex', version='0.1')

# Load image pairs
gen_dir = Path("generated_images/")
ref_dir = Path("path/to/dataset/")  # Original training images

gen_images = sorted(gen_dir.glob("*.png"))
ref_images = sorted(ref_dir.glob("*.jpg"))[:len(gen_images)]  # Match counts

lpips_scores = []

for gen_path, ref_path in zip(gen_images, ref_images):
    # Load images
    gen_img = Image.open(gen_path).convert('RGB')
    ref_img = Image.open(ref_path).convert('RGB')

    # Resize to same dimensions (e.g., 512x512)
    gen_img = gen_img.resize((512, 512))
    ref_img = ref_img.resize((512, 512))

    # Convert to tensors, normalize to [-1, 1]
    gen_tensor = torch.tensor(np.array(gen_img)).float() / 127.5 - 1.0
    ref_tensor = torch.tensor(np.array(ref_img)).float() / 127.5 - 1.0

    # Reshape to batch format (B, C, H, W)
    gen_tensor = gen_tensor.unsqueeze(0).permute(0, 3, 1, 2).cuda()
    ref_tensor = ref_tensor.unsqueeze(0).permute(0, 3, 1, 2).cuda()

    # Compute LPIPS
    d = loss_fn.forward(gen_tensor, ref_tensor)
    lpips_scores.append(d.item())

avg_lpips = sum(lpips_scores) / len(lpips_scores)
print(f"Average LPIPS: {avg_lpips:.4f}")
print(f"  Lower is better (0 = identical)")
```

### Method B: Batch Processing (Faster)

```python
import lpips
import torch
from PIL import Image
import numpy as np
from pathlib import Path

def load_images_batch(image_dir, max_images=100):
    """Load images and return as normalized tensors."""
    images = []
    paths = sorted(Path(image_dir).glob("*.png"))[:max_images]

    for path in paths:
        img = Image.open(path).convert('RGB').resize((512, 512))
        tensor = torch.tensor(np.array(img)).float() / 127.5 - 1.0
        tensor = tensor.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
        images.append(tensor)

    return torch.stack(images).cuda()

loss_fn = lpips.LPIPS(net='alex', version='0.1')

gen_images = load_images_batch("generated_images/")
ref_images = load_images_batch("path/to/dataset/", max_images=len(gen_images))

# Compute all at once
d = loss_fn.forward(gen_images, ref_images)
avg_lpips = d.mean().item()
print(f"Average LPIPS: {avg_lpips:.4f}")
```

**Interpretation:**

- **LPIPS < 0.15**: Excellent identity preservation (generated faces very similar to training)
- **LPIPS 0.15-0.25**: Good preservation (recognizable identity)
- **LPIPS > 0.25**: Weak preservation (less similar to reference)[^1]

## Step 3: Calculate FID Score

FID compares the statistical distribution of generated images to real images (overall quality).

### Installation

```bash
pip install pytorch-fid
```

### Method A: Command Line (Simplest)

```bash
python -m pytorch_fid path/to/generated_images path/to/reference_images --device cuda:0
```

Example:

```bash
python -m pytorch_fid generated_images/ path/to/dataset/ --device cuda:0
```

Output:

```
FID:  15.234
```

### Method B: Python API (More Control)

```python
from pytorch_fid.fid_score import calculate_fid_given_paths
import torch

# Paths to generated and reference image directories
generated_path = "generated_images/"
reference_path = "path/to/dataset/"

# Calculate FID
fid_value = calculate_fid_given_paths(
    [generated_path, reference_path],
    batch_size=50,
    device='cuda:0',
    dims=2048  # Use Inception pool3 layer (default, most comparable)
)

print(f"FID Score: {fid_value:.4f}")
```

### Method C: Pre-compute Reference Statistics (For Multiple LoRA Comparisons)

If testing multiple LoRAs against the same reference dataset:

```python
from pytorch_fid.fid_score import calculate_fid_given_paths

# Save reference dataset statistics (one-time)
import subprocess
subprocess.run([
    "python", "-m", "pytorch_fid",
    "--save-stats",
    "path/to/dataset/",
    "reference_stats.npz",
    "--device", "cuda:0"
], check=True)

# Later: Compare any generated set against pre-computed stats
fid_value = calculate_fid_given_paths(
    ["generated_images/", "reference_stats.npz"],
    batch_size=50,
    device='cuda:0',
    dims=2048
)

print(f"FID Score: {fid_value:.4f}")
```

**Interpretation:**

- **FID < 10**: Excellent quality (closely matches real image distribution)
- **FID 10-30**: Good quality (comparable to well-trained GANs)
- **FID 30-50**: Moderate quality (noticeable artifacts)
- **FID > 50**: Poor quality (significant degradation)[^2]

## Complete Evaluation Pipeline

Combine both metrics in a single workflow:

```python
import lpips
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from pytorch_fid.fid_score import calculate_fid_given_paths
from diffusers import StableDiffusionPipeline

def evaluate_lora(lora_path, reference_dir, output_dir="evaluation_results/"):
    """Comprehensive LoRA evaluation."""

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    gen_dir = output_dir / "generated"
    gen_dir.mkdir(exist_ok=True)

    # Step 1: Generate images
    print("Generating images...")
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16
    ).to("cuda")
    pipe.unet.load_attn_procs(lora_path)

    prompts = [
        "photo of sks person in a blue shirt",
        "photo of sks person wearing glasses",
        "photo of sks person in a suit",
        "full body photo of sks person",
        "portrait of sks person smiling",
    ] * 10  # 50 total images

    for i, prompt in enumerate(prompts):
        image = pipe(prompt, height=512, width=512, num_inference_steps=50).images[^0]
        image.save(f"{gen_dir}/gen_{i:03d}.png")

    # Step 2: Calculate LPIPS
    print("Calculating LPIPS...")
    loss_fn = lpips.LPIPS(net='alex')
    lpips_scores = []

    gen_images = sorted(gen_dir.glob("*.png"))
    ref_images = sorted(Path(reference_dir).glob("*.jpg"))[:len(gen_images)]

    for gen_p, ref_p in zip(gen_images, ref_images):
        gen_img = Image.open(gen_p).convert('RGB').resize((512, 512))
        ref_img = Image.open(ref_p).convert('RGB').resize((512, 512))

        gen_t = torch.tensor(np.array(gen_img)).float() / 127.5 - 1.0
        ref_t = torch.tensor(np.array(ref_img)).float() / 127.5 - 1.0

        gen_t = gen_t.unsqueeze(0).permute(0, 3, 1, 2).cuda()
        ref_t = ref_t.unsqueeze(0).permute(0, 3, 1, 2).cuda()

        lpips_scores.append(loss_fn.forward(gen_t, ref_t).item())

    avg_lpips = np.mean(lpips_scores)

    # Step 3: Calculate FID
    print("Calculating FID...")
    fid_score = calculate_fid_given_paths(
        [str(gen_dir), reference_dir],
        batch_size=50,
        device='cuda:0'
    )

    # Report results
    print(f"\n{'='*50}")
    print(f"LoRA Evaluation Results")
    print(f"{'='*50}")
    print(f"LPIPS (Identity Preservation): {avg_lpips:.4f}")
    print(f"  ↳ Lower is better (< 0.20 = excellent)")
    print(f"FID (Overall Quality): {fid_score:.4f}")
    print(f"  ↳ Lower is better (< 30 = good)")
    print(f"{'='*50}")

    # Save results
    with open(output_dir / "metrics.txt", "w") as f:
        f.write(f"LPIPS: {avg_lpips:.4f}\n")
        f.write(f"FID: {fid_score:.4f}\n")

    return {"lpips": avg_lpips, "fid": fid_score}

# Usage
results = evaluate_lora(
    lora_path="path/to/lora_output",
    reference_dir="path/to/dataset/"
)
```

## Best Practices for Evaluation

| Step              | Guideline                | Rationale                      |
| :---------------- | :----------------------- | :----------------------------- |
| Generate images   | 50-100+ diverse prompts  | Ensures statistical validity   |
| LPIPS comparison  | Use 512×512 resolution   | Matches training resolution    |
| FID batch size    | 50-100                   | Balance speed and memory       |
| Reference set     | 50+ high-quality images  | Represents target distribution |
| Repeat evaluation | Test 3+ LoRA checkpoints | Find optimal epoch             |

**Checkpoints to Compare:**
Save LoRA snapshots at epochs 6, 8, 10, 12 during training; evaluate each with LPIPS/FID to identify the best-performing epoch (often 8-10 for person LoRAs).[^3]

---

This workflow provides quantitative evidence of your LoRA's identity preservation (LPIPS) and overall image quality (FID), enabling data-driven model selection and comparison.

<div align="center">⁂</div>

[^1]: https://github.com/richzhang/PerceptualSimilarity
[^2]: https://github.com/mseitzer/pytorch-fid
[^3]: https://docs.swanlab.cn/en/examples/stable_diffusion.html
