#!/bin/bash

set -e

# =========================
# 0. 项目路径
# =========================
cd /home/network/Desktop/Project/BUS-nnUnet

export nnUNet_raw="/home/network/Desktop/Project/BUS-nnUnet/nnUNet_raw"
export nnUNet_preprocessed="/home/network/Desktop/Project/BUS-nnUnet/nnUNet_preprocessed"
export nnUNet_results="/home/network/Desktop/Project/BUS-nnUnet/nnUNet_results"

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# =========================
# 1. 参数设置
# =========================
DATASET_ID=3
CONFIG=2d
FOLD=0
CHECKPOINT="checkpoint_best.pth"

# 输入：prepare_predict_from_excel.py 已经生成的 nnU-Net 输入目录
# INPUT_DIR="nnUNet_predict_input/3_M"
# INPUT_DIR="nnUNet_predict_input/3_B"
INPUT_DIR="nnUNet_predict_input/4_M"

# nnU-Net 原始预测输出目录
# PRED_DIR="nnUNet_predict_output/3_M"
# PRED_DIR="nnUNet_predict_output/3_B"
PRED_DIR="nnUNet_predict_output/4_M"

# 映射表：prepare_predict_from_excel.py 生成的 mapping csv
# MAPPING_CSV="nnUNet_predict_input/3_M/3_M_predict_mapping.csv"
# MAPPING_CSV="nnUNet_predict_input/3_B/3_B_predict_mapping.csv"
MAPPING_CSV="nnUNet_predict_input/4_M/4_M_predict_mapping.csv"

# 按 patient_id 导出的 mask
# MASK_BY_PATIENT_DIR="masks_by_patient_id/3_M"
# MASK_BY_PATIENT_DIR="masks_by_patient_id/3_B"
MASK_BY_PATIENT_DIR="masks_by_patient_id/4_M"

# overlay 可视化输出目录
# OVERLAY_DIR="prediction_overlays/3_M"
# OVERLAY_DIR="prediction_overlays/3_B"
OVERLAY_DIR="prediction_overlays/4_M"

mkdir -p "${PRED_DIR}"
mkdir -p "${MASK_BY_PATIENT_DIR}"
mkdir -p "${OVERLAY_DIR}"

echo "=========================================="
echo "Step 1/3: Running nnU-Net prediction"
echo "Input dir: ${INPUT_DIR}"
echo "Output dir: ${PRED_DIR}"
echo "Checkpoint: ${CHECKPOINT}"
echo "=========================================="

nnUNetv2_predict \
    -i "${INPUT_DIR}" \
    -o "${PRED_DIR}" \
    -d ${DATASET_ID} \
    -c ${CONFIG} \
    -f ${FOLD} \
    -chk ${CHECKPOINT}

echo ""
echo "=========================================="
echo "Step 2/3: Exporting masks by patient_id and image name"
echo "=========================================="

python - << 'PYCODE'
from pathlib import Path
import shutil
import pandas as pd


# mapping_csv = Path("nnUNet_predict_input/3_M/3_M_predict_mapping.csv")
# pred_dir = Path("nnUNet_predict_output/3_M")
# out_dir = Path("masks_by_patient_id/3_M")
# out_csv = Path("prediction_results_with_patient_id_3_M.csv")

# mapping_csv = Path("nnUNet_predict_input/3_B/3_B_predict_mapping.csv")
# pred_dir = Path("nnUNet_predict_output/3_B")
# out_dir = Path("masks_by_patient_id/3_B")
# out_csv = Path("prediction_results_with_patient_id_3_B.csv")

mapping_csv = Path("nnUNet_predict_input/4_M/4_M_predict_mapping.csv")
pred_dir = Path("nnUNet_predict_output/4_M")
out_dir = Path("masks_by_patient_id/4_M")
out_csv = Path("prediction_results_with_patient_id_4_M.csv")

out_dir.mkdir(parents=True, exist_ok=True)


def safe_filename(name):
    name = str(name)
    name = name.replace("/", "_")
    name = name.replace("\\", "_")
    name = name.replace(" ", "_")
    name = name.replace("(", "_")
    name = name.replace(")", "_")
    name = name.replace(":", "_")
    return name


if not mapping_csv.exists():
    raise FileNotFoundError(f"Mapping CSV not found: {mapping_csv}")

df = pd.read_csv(mapping_csv)

records = []

for _, row in df.iterrows():
    case_id = str(row["case_id"])
    patient_id = str(row["patient_id"])
    image_path = Path(row["original_b_mode_image"])

    pred_mask = pred_dir / f"{case_id}.png"

    if not pred_mask.exists():
        print(f"[Warning] Missing prediction: {pred_mask}")
        continue

    image_stem = image_path.stem  # FILE4 / FILE2

    new_name = f"{safe_filename(case_id)}_{safe_filename(patient_id)}_{safe_filename(image_stem)}_mask.png"
    dst_path = out_dir / new_name

    shutil.copy(pred_mask, dst_path)

    records.append({
        "case_id": case_id,
        "patient_id": patient_id,
        "original_b_mode_image": str(image_path),
        "nnunet_mask": str(pred_mask),
        "mask_by_patient_id": str(dst_path),
        "image_stem": image_stem,
    })

pd.DataFrame(records).to_csv(out_csv, index=False, encoding="utf-8-sig")

print(f"Exported masks: {len(records)}")
print(f"Mask folder: {out_dir.resolve()}")
print(f"Result CSV: {out_csv.resolve()}")
PYCODE


echo ""
echo "=========================================="
echo "Step 3/3: Generating overlay images"
echo "=========================================="

python - << 'PYCODE'
from pathlib import Path
import os

os.environ["QT_QPA_PLATFORM"] = "offscreen"

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt


# result_csv = Path("prediction_results_with_patient_id_3_M.csv")
# out_dir = Path("prediction_overlays/3_M")
# out_dir.mkdir(parents=True, exist_ok=True)

# result_csv = Path("prediction_results_with_patient_id_3_B.csv")
# out_dir = Path("prediction_overlays/3_B")
# out_dir.mkdir(parents=True, exist_ok=True)

result_csv = Path("prediction_results_with_patient_id_4_M.csv")
out_dir = Path("prediction_overlays/4_M")
out_dir.mkdir(parents=True, exist_ok=True)


def load_gray(path):
    return np.array(Image.open(path).convert("L"))


def normalize_to_uint8(img):
    img = img.astype(np.float32)
    min_val = img.min()
    max_val = img.max()

    if max_val - min_val < 1e-8:
        return np.zeros_like(img, dtype=np.uint8)

    img = (img - min_val) / (max_val - min_val)
    img = (img * 255).astype(np.uint8)
    return img


def make_overlay(image, mask, alpha=0.35):
    image_uint8 = normalize_to_uint8(image)
    rgb = np.stack([image_uint8, image_uint8, image_uint8], axis=-1)

    mask_bin = mask > 0

    overlay = rgb.copy()
    overlay[mask_bin] = [255, 0, 0]

    blended = alpha * overlay.astype(np.float32) + (1 - alpha) * rgb.astype(np.float32)
    return blended.astype(np.uint8)


def safe_filename(name):
    name = str(name)
    name = name.replace("/", "_")
    name = name.replace("\\", "_")
    name = name.replace(" ", "_")
    name = name.replace("(", "_")
    name = name.replace(")", "_")
    name = name.replace(":", "_")
    return name


if not result_csv.exists():
    raise FileNotFoundError(f"Result CSV not found: {result_csv}")

df = pd.read_csv(result_csv)

saved = 0

for _, row in df.iterrows():
    case_id = str(row["case_id"])
    patient_id = str(row["patient_id"])
    image_path = Path(row["original_b_mode_image"])
    mask_path = Path(row["mask_by_patient_id"])

    image_stem = image_path.stem

    if not image_path.exists():
        print(f"[Warning] image not found: {image_path}")
        continue

    if not mask_path.exists():
        print(f"[Warning] mask not found: {mask_path}")
        continue

    image = load_gray(image_path)
    mask = load_gray(mask_path)
    mask = (mask > 0).astype(np.uint8)

    if image.shape != mask.shape:
        print(
            f"[Warning] shape mismatch: "
            f"case_id={case_id}, patient_id={patient_id}, "
            f"image={image.shape}, mask={mask.shape}"
        )
        continue

    overlay = make_overlay(image, mask)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("B-mode Image")
    axes[0].axis("off")

    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Predicted Mask")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    fig.suptitle(f"{case_id} | {patient_id} | {image_stem}")

    save_name = (
        f"{safe_filename(case_id)}_"
        f"{safe_filename(patient_id)}_"
        f"{safe_filename(image_stem)}_overlay.png"
    )

    save_path = out_dir / save_name

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    saved += 1

print(f"Saved overlays: {saved}")
print(f"Overlay folder: {out_dir.resolve()}")
PYCODE


echo ""
echo "=========================================="
echo "All done."
echo "Prediction masks: ${PRED_DIR}"
echo "Masks by patient_id: ${MASK_BY_PATIENT_DIR}"
echo "Overlays: ${OVERLAY_DIR}"
echo "CSV: prediction_results_with_patient_id_4_M.csv"
echo "=========================================="