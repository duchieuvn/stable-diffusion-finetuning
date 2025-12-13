import os
import numpy as np
from PIL import Image
import torchvision
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Config
out = "stl10_512_sd"
os.makedirs(out + "/train", exist_ok=True)
os.makedirs(out + "/val",   exist_ok=True)
os.makedirs(out + "/test",  exist_ok=True)

classes = ["airplane","bird","car","cat","deer","dog","horse","monkey","ship","truck"]

# Load
train_ds = torchvision.datasets.STL10(".", split="train", download=True)
test_ds  = torchvision.datasets.STL10(".", split="test",  download=True)

# Chia train → train + val (90/10)
imgs, lbls = zip(*train_ds)
train_imgs, val_imgs, train_lbls, val_lbls = train_test_split(
    imgs, lbls, test_size=0.1, random_state=42, stratify=lbls)

# Save function
def save_split(name, imgs, lbls):
    for i, (img, lbl) in enumerate(tqdm(zip(imgs, lbls), total=len(imgs), desc=name)):
        # # normalize to PIL Image (handle numpy arrays, PIL Images, and other array-like inputs)
        # if isinstance(img, np.ndarray):
        #     pil_img = Image.fromarray(img)
        # else:
        #     # PIL Image has .save; otherwise try to convert via numpy
        #     if hasattr(img, "save"):
        #         pil_img = img
        #     else:
        #     pil_img = Image.fromarray(np.array(img))
        pil_img = Image.fromarray(np.array(img))

        pil_img.save(f"{out}/{name}/{i:06d}.png")

        with open(f"{out}/{name}/{i:06d}.txt", "w") as f:
            f.write(classes[lbl])

# Save
save_split("train", train_imgs, train_lbls)
save_split("val",   val_imgs,   val_lbls)
save_split("test",  test_ds.data, test_ds.labels)   # test giữ nguyên

print("Done! Dataset ready:")
print("   train:", len(train_imgs))
print("   val:  ", len(val_imgs))
print("   test: ", len(test_ds))