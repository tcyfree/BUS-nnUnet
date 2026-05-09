import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


def save_image_as_grayscale(img_path, save_path):
    """
    原始超声图像：只转为单通道灰度图，不做二值化。
    正常保存后像素值应该是 0~255，unique values 很多。
    """
    img = Image.open(img_path).convert("L")
    img.save(save_path)


def save_mask_as_binary(mask_path, save_path):
    """
    分割标签：转成 0/1。
    """
    mask = Image.open(mask_path).convert("L")
    mask_np = np.array(mask)

    mask_bin = (mask_np > 0).astype(np.uint8)

    Image.fromarray(mask_bin).save(save_path)


def check_pair_size(img_path, mask_path):
    img = Image.open(img_path)
    mask = Image.open(mask_path)

    if img.size != mask.size:
        raise ValueError(
            f"Size mismatch: {img_path.name}, image={img.size}, mask={mask.size}"
        )


def main():
    # src_root = Path("datasets/BUSI")
    # src_root = Path("datasets/BUSI_3M")
    src_root = Path("datasets/BUSI_sw_all")
    img_dir = src_root / "img"
    label_dir = src_root / "label"

    # out_root = Path("nnUNet_raw/Dataset001_BUSI")
    # out_root = Path("nnUNet_raw/Dataset002_BUSI_3M")
    out_root = Path("nnUNet_raw/Dataset003_BUSI_sw_all")
    imagesTr = out_root / "imagesTr"
    labelsTr = out_root / "labelsTr"

    imagesTr.mkdir(parents=True, exist_ok=True)
    labelsTr.mkdir(parents=True, exist_ok=True)

    image_files = sorted(img_dir.glob("*.png"))

    case_records = []
    case_id = 1

    for img_path in tqdm(image_files, desc="Converting BUSI"):
        mask_path = label_dir / img_path.name

        if not mask_path.exists():
            print(f"[Warning] mask not found: {img_path.name}")
            continue

        check_pair_size(img_path, mask_path)

        name_lower = img_path.name.lower()

        if name_lower.startswith("benign"):
            diagnosis = "benign"
        elif name_lower.startswith("malignant"):
            diagnosis = "malignant"
        elif name_lower.startswith("normal"):
            diagnosis = "normal"
        else:
            diagnosis = "unknown"

        new_case_id = f"BUSI_{case_id:04d}"

        dst_img_path = imagesTr / f"{new_case_id}_0000.png"
        dst_mask_path = labelsTr / f"{new_case_id}.png"

        # 关键：image 不二值化
        save_image_as_grayscale(img_path, dst_img_path)

        # 关键：label 才二值化
        save_mask_as_binary(mask_path, dst_mask_path)

        case_records.append(
            {
                "case_id": new_case_id,
                "original_name": img_path.name,
                "diagnosis": diagnosis,
                "image_path": str(dst_img_path),
                "label_path": str(dst_mask_path),
            }
        )

        case_id += 1

    dataset_json = {
        "channel_names": {
            "0": "Ultrasound"
        },
        "labels": {
            "background": 0,
            "lesion": 1
        },
        "numTraining": len(case_records),
        "file_ending": ".png"
    }

    with open(out_root / "dataset.json", "w", encoding="utf-8") as f:
        json.dump(dataset_json, f, indent=4, ensure_ascii=False)

    pd.DataFrame(case_records).to_csv(
        out_root / "case_info.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("Done.")
    print(f"Total converted: {len(case_records)}")
    print(pd.DataFrame(case_records)["diagnosis"].value_counts())


if __name__ == "__main__":
    main()