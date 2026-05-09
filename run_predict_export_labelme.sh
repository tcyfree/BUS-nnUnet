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
echo "Step 2/3: Exporting masks + labelme json"
echo "=========================================="

python - << 'PYCODE'
from pathlib import Path
import shutil
import json
import base64

import cv2
import numpy as np
import pandas as pd
from PIL import Image


# =========================
# 路径
# =========================

mapping_csv = Path("nnUNet_predict_input/4_M/4_M_predict_mapping.csv")
pred_dir = Path("nnUNet_predict_output/4_M")
out_mask_dir = Path("masks_by_patient_id/4_M")
out_mask_dir.mkdir(parents=True, exist_ok=True)

# 最终输出目录
out_dir = Path("labelme_annotations/4_M")

out_csv = Path("prediction_results_with_patient_id_4_M.csv")

out_dir.mkdir(parents=True, exist_ok=True)


# =========================
# 工具函数
# =========================

def safe_filename(name):
    name = str(name)

    for x in ["/", "\\", " ", "(", ")", ":"]:
        name = name.replace(x, "_")

    return name


def image_to_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def mask_to_polygons(mask):
    """
    mask -> polygon list
    """

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    polygons = []

    for contour in contours:

        area = cv2.contourArea(contour)

        # 去除很小噪声
        if area < 20:
            continue

        epsilon = 0.002 * cv2.arcLength(contour, True)

        contour = cv2.approxPolyDP(
            contour,
            epsilon,
            True
        )

        points = contour.squeeze(1)

        if len(points) < 3:
            continue

        polygons.append(points.tolist())

    return polygons


def create_labelme_json(
    image_path,
    mask_path,
    json_path,
    label_name="lesion"
):

    image = np.array(Image.open(image_path).convert("L"))

    mask = np.array(Image.open(mask_path).convert("L"))

    mask = (mask > 0).astype(np.uint8)

    polygons = mask_to_polygons(mask)

    shapes = []

    for poly in polygons:

        shapes.append({
            "label": label_name,
            "points": poly,
            "group_id": None,
            "shape_type": "polygon",
            "flags": {}
        })

    h, w = image.shape

    data = {
        "version": "5.5.0",
        "flags": {},
        "shapes": shapes,
        "imagePath": image_path.name,
        "imageData": image_to_base64(image_path),
        "imageHeight": int(h),
        "imageWidth": int(w)
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2
        )


# =========================
# 主流程
# =========================

if not mapping_csv.exists():
    raise FileNotFoundError(mapping_csv)

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

    image_stem = image_path.stem

    # =========================
    # 一个病人一个文件夹
    # =========================

    patient_dir = out_dir / safe_filename(patient_id)
    patient_dir_mask = out_mask_dir / safe_filename(patient_id)

    patient_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # =========================
    # 文件名
    # =========================

    base_name = (
        f"{safe_filename(case_id)}_"
        f"{safe_filename(image_stem)}"
    )

    dst_image = patient_dir / f"{base_name}.png"

    dst_json = patient_dir / f"{base_name}.json"

    dst_mask = patient_dir_mask / f"{base_name}_mask.png"

    # =========================
    # 复制原图
    # =========================

    shutil.copy(image_path, dst_image)

    # =========================
    # 生成labelme json
    # =========================

    create_labelme_json(
        dst_image,
        pred_mask,
        dst_json
    )

    records.append({
        "case_id": case_id,
        "patient_id": patient_id,
        "image": str(dst_image),
        "original_b_mode_image": str(image_path),
        "mask_by_patient_id": str(dst_mask),
        "json": str(dst_json),
    })

    print(f"[OK] {patient_id} | {image_stem}")

pd.DataFrame(records).to_csv(
    out_csv,
    index=False,
    encoding="utf-8-sig"
)

print("")
print("===================================")
print(f"Exported: {len(records)}")
print(f"Output: {out_dir.resolve()}")
print(f"CSV: {out_csv.resolve()}")
print("===================================")

PYCODE


# echo ""
# echo "=========================================="
# echo "Step 3/3: Generating overlay images"
# echo "=========================================="

# python - << 'PYCODE'
# from pathlib import Path
# import os

# os.environ["QT_QPA_PLATFORM"] = "offscreen"

# import matplotlib
# matplotlib.use("Agg")

# import numpy as np
# import pandas as pd
# from PIL import Image
# import matplotlib.pyplot as plt


# # result_csv = Path("prediction_results_with_patient_id_3_M.csv")
# # out_dir = Path("prediction_overlays/3_M")
# # out_dir.mkdir(parents=True, exist_ok=True)

# # result_csv = Path("prediction_results_with_patient_id_3_B.csv")
# # out_dir = Path("prediction_overlays/3_B")
# # out_dir.mkdir(parents=True, exist_ok=True)

# result_csv = Path("prediction_results_with_patient_id_4_M.csv")
# out_dir = Path("prediction_overlays/4_M")
# out_dir.mkdir(parents=True, exist_ok=True)


# def load_gray(path):
#     return np.array(Image.open(path).convert("L"))


# def normalize_to_uint8(img):
#     img = img.astype(np.float32)
#     min_val = img.min()
#     max_val = img.max()

#     if max_val - min_val < 1e-8:
#         return np.zeros_like(img, dtype=np.uint8)

#     img = (img - min_val) / (max_val - min_val)
#     img = (img * 255).astype(np.uint8)
#     return img


# def make_overlay(image, mask, alpha=0.35):
#     image_uint8 = normalize_to_uint8(image)
#     rgb = np.stack([image_uint8, image_uint8, image_uint8], axis=-1)

#     mask_bin = mask > 0

#     overlay = rgb.copy()
#     overlay[mask_bin] = [255, 0, 0]

#     blended = alpha * overlay.astype(np.float32) + (1 - alpha) * rgb.astype(np.float32)
#     return blended.astype(np.uint8)


# def safe_filename(name):
#     name = str(name)
#     name = name.replace("/", "_")
#     name = name.replace("\\", "_")
#     name = name.replace(" ", "_")
#     name = name.replace("(", "_")
#     name = name.replace(")", "_")
#     name = name.replace(":", "_")
#     return name


# if not result_csv.exists():
#     raise FileNotFoundError(f"Result CSV not found: {result_csv}")

# df = pd.read_csv(result_csv)

# saved = 0

# for _, row in df.iterrows():
#     case_id = str(row["case_id"])
#     patient_id = str(row["patient_id"])
#     image_path = Path(row["original_b_mode_image"])
#     mask_path = Path(row["mask_by_patient_id"])

#     image_stem = image_path.stem

#     if not image_path.exists():
#         print(f"[Warning] image not found: {image_path}")
#         continue

#     if not mask_path.exists():
#         print(f"[Warning] mask not found: {mask_path}")
#         continue

#     image = load_gray(image_path)
#     mask = load_gray(mask_path)
#     mask = (mask > 0).astype(np.uint8)

#     if image.shape != mask.shape:
#         print(
#             f"[Warning] shape mismatch: "
#             f"case_id={case_id}, patient_id={patient_id}, "
#             f"image={image.shape}, mask={mask.shape}"
#         )
#         continue

#     overlay = make_overlay(image, mask)

#     fig, axes = plt.subplots(1, 3, figsize=(14, 5))

#     axes[0].imshow(image, cmap="gray")
#     axes[0].set_title("B-mode Image")
#     axes[0].axis("off")

#     axes[1].imshow(mask, cmap="gray")
#     axes[1].set_title("Predicted Mask")
#     axes[1].axis("off")

#     axes[2].imshow(overlay)
#     axes[2].set_title("Overlay")
#     axes[2].axis("off")

#     fig.suptitle(f"{case_id} | {patient_id} | {image_stem}")

#     save_name = (
#         f"{safe_filename(case_id)}_"
#         f"{safe_filename(patient_id)}_"
#         f"{safe_filename(image_stem)}_overlay.png"
#     )

#     save_path = out_dir / save_name

#     plt.tight_layout()
#     plt.savefig(save_path, dpi=150, bbox_inches="tight")
#     plt.close()

#     saved += 1

# print(f"Saved overlays: {saved}")
# print(f"Overlay folder: {out_dir.resolve()}")
# PYCODE


# echo ""
# echo "=========================================="
# echo "All done."
# echo "Prediction masks: ${PRED_DIR}"
# echo "Masks by patient_id: ${MASK_BY_PATIENT_DIR}"
# echo "Overlays: ${OVERLAY_DIR}"
# echo "CSV: prediction_results_with_patient_id_4_M.csv"
# echo "=========================================="