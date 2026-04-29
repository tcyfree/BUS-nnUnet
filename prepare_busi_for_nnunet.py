import os
import json
import shutil
from pathlib import Path
from PIL import Image
import numpy as np


src_root = Path("datasets/BUSI")
img_dir = src_root / "img"
label_dir = src_root / "label"

out_root = Path("nnUNet_raw/Dataset001_BUSI")
imagesTr = out_root / "imagesTr"
labelsTr = out_root / "labelsTr"

imagesTr.mkdir(parents=True, exist_ok=True)
labelsTr.mkdir(parents=True, exist_ok=True)

# 保存病例信息，方便以后按 benign/malignant 分析
case_info = []

image_files = sorted([p for p in img_dir.glob("*.png")])

case_id = 1

for img_path in image_files:
    label_path = label_dir / img_path.name

    if not label_path.exists():
        print(f"Warning: label not found for {img_path.name}")
        continue

    # 判断良恶性来源
    name_lower = img_path.name.lower()
    if name_lower.startswith("benign"):
        diagnosis = "benign"
    elif name_lower.startswith("malignant"):
        diagnosis = "malignant"
    else:
        diagnosis = "unknown"

    new_id = f"BUSI_{case_id:04d}"

    # nnU-Net image 命名：caseid_0000.png
    new_img_name = f"{new_id}_0000.png"

    # nnU-Net label 命名：caseid.png
    new_label_name = f"{new_id}.png"


    # 保存原图为单通道灰度图
    img = Image.open(img_path).convert("L")
    img.save(imagesTr / new_img_name)

    # 处理 mask：确保是 0 和 1，而不是 0 和 255
    mask = Image.open(imagesTr / new_img_name).convert("L")
    mask_np = np.array(mask)
    mask_bin = (mask_np > 0).astype(np.uint8)
    Image.fromarray(mask_bin).save(imagesTr / new_img_name)


    case_info.append({
        "case_id": new_id,
        "original_name": img_path.name,
        "diagnosis": diagnosis
    })

    case_id += 1


dataset_json = {
    "channel_names": {
        "0": "Ultrasound"
    },
    "labels": {
        "background": 0,
        "lesion": 1
    },
    "numTraining": len(case_info),
    "file_ending": ".png"
}

with open(out_root / "dataset.json", "w", encoding="utf-8") as f:
    json.dump(dataset_json, f, indent=4)

with open(out_root / "case_info.json", "w", encoding="utf-8") as f:
    json.dump(case_info, f, indent=4, ensure_ascii=False)

print(f"Done. Converted {len(case_info)} cases.")
print(f"Output folder: {out_root}")