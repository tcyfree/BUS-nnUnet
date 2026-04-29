#!/bin/bash

set -e

# =========================
# 1. 设置 nnU-Net 路径
# =========================
export nnUNet_raw="$(pwd)/nnUNet_raw"
export nnUNet_preprocessed="$(pwd)/nnUNet_preprocessed"
export nnUNet_results="$(pwd)/nnUNet_results"

echo "nnUNet_raw=${nnUNet_raw}"
echo "nnUNet_preprocessed=${nnUNet_preprocessed}"
echo "nnUNet_results=${nnUNet_results}"

# =========================
# 2. 数据集编号
# =========================
DATASET_ID=1

# =========================
# 3. 预处理和数据完整性检查
# =========================
nnUNetv2_plan_and_preprocess \
    -d ${DATASET_ID} \
    --verify_dataset_integrity

# =========================
# 4. 训练 2D nnU-Net，5 折交叉验证
# =========================

nnUNetv2_train ${DATASET_ID} 2d 0
nnUNetv2_train ${DATASET_ID} 2d 1
nnUNetv2_train ${DATASET_ID} 2d 2
nnUNetv2_train ${DATASET_ID} 2d 3
nnUNetv2_train ${DATASET_ID} 2d 4

# =========================
# 5. 自动寻找最佳配置
# =========================
nnUNetv2_find_best_configuration ${DATASET_ID} -c 2d

echo "Training finished."