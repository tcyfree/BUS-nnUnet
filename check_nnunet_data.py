from pathlib import Path
from PIL import Image
import numpy as np
import random


img_dir = Path("nnUNet_raw/Dataset001_BUSI/imagesTr")
label_dir = Path("nnUNet_raw/Dataset001_BUSI/labelsTr")

img_files = sorted(img_dir.glob("*.png"))

print("Total images:", len(img_files))

sample_files = random.sample(img_files, min(20, len(img_files)))

for img_path in sample_files:
    case_id = img_path.name.replace("_0000.png", "")
    label_path = label_dir / f"{case_id}.png"

    img = np.array(Image.open(img_path).convert("L"))
    mask = np.array(Image.open(label_path).convert("L"))

    print("=" * 80)
    print("Image:", img_path.name)
    print("Label:", label_path.name)

    print("Image shape:", img.shape)
    print("Mask shape:", mask.shape)

    print("Image min/max:", img.min(), img.max())
    print("Image unique count:", len(np.unique(img)))
    print("Image unique first 20:", np.unique(img)[:20])

    print("Mask min/max:", mask.min(), mask.max())
    print("Mask unique:", np.unique(mask))
    print("Mask foreground pixels:", np.sum(mask > 0))