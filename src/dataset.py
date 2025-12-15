import torch
from torchvision import transforms
from PIL import Image
import os

class BabyDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.image_dir = f"{dataset_path}/images"
        self.caption_dir = f"{dataset_path}/captions"
        
        # Standard SD 1.5 normalization
        self.resize = transforms.Compose([
            transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]), # Map to [-1, 1]
        ])
        
        # Load all image filenames from the images directory
        self.image_files = [f for f in os.listdir(self.image_dir) if f.endswith(('.jpg', '.png'))]
        self.image_files.sort()  # Sort for consistency

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        img = Image.open(img_path).convert("RGB")
        img = self.resize(img)
        
        # Load caption from corresponding text file
        # Remove .jpg or .png extension and add .txt
        img_basename = os.path.splitext(self.image_files[idx])[0]
        caption_path = os.path.join(self.caption_dir, f"{img_basename}.txt")
        
        # Read caption, fallback to default if not found
        if os.path.exists(caption_path):
            with open(caption_path, "r") as f:
                caption = f.read().strip()
        else:
            caption = "a photo of a baby"
        

        return img, caption
