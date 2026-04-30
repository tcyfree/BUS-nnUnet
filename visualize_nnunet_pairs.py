from pathlib import Path
import random
import os

os.environ["QT_QPA_PLATFORM"] = "offscreen"

import matplotlib
matplotlib.use("Agg")

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def load_gray_image(path):
    """
    Load image as grayscale numpy array.
    """
    img = Image.open(path).convert("L")
    return np.array(img)


def normalize_to_uint8(img):
    """
    Normalize image to 0~255 uint8 for visualization.
    """
    img = img.astype(np.float32)

    min_val = img.min()
    max_val = img.max()

    if max_val - min_val < 1e-8:
        return np.zeros_like(img, dtype=np.uint8)

    img = (img - min_val) / (max_val - min_val)
    img = (img * 255).astype(np.uint8)

    return img


def get_mask_contour(mask):
    """
    Get contour image from binary mask.
    """
    mask_bin = (mask > 0).astype(np.uint8)

    contours, _ = cv2.findContours(
        mask_bin,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    contour_img = np.zeros_like(mask_bin, dtype=np.uint8)
    cv2.drawContours(contour_img, contours, -1, 1, thickness=2)

    return contour_img


def make_overlay(image, mask, alpha=0.35):
    """
    Create image + mask overlay.
    Red area = lesion mask.
    """
    image_uint8 = normalize_to_uint8(image)

    rgb = cv2.cvtColor(image_uint8, cv2.COLOR_GRAY2RGB)

    mask_bin = (mask > 0).astype(np.uint8)

    overlay = rgb.copy()
    overlay[mask_bin == 1] = [255, 0, 0]

    blended = cv2.addWeighted(overlay, alpha, rgb, 1 - alpha, 0)

    return blended


def make_contour_overlay(image, mask):
    """
    Create image + mask contour overlay.
    Green line = lesion boundary.
    """
    image_uint8 = normalize_to_uint8(image)
    rgb = cv2.cvtColor(image_uint8, cv2.COLOR_GRAY2RGB)

    contour = get_mask_contour(mask)
    rgb[contour > 0] = [0, 255, 0]

    return rgb


def visualize_one_case(img_path, label_path, save_path):
    """
    Visualize one image-mask pair.
    """
    image = load_gray_image(img_path)
    mask = load_gray_image(label_path)
    mask_bin = (mask > 0).astype(np.uint8)

    if image.shape != mask_bin.shape:
        raise ValueError(
            f"Shape mismatch: {img_path.name}, image={image.shape}, mask={mask_bin.shape}"
        )

    overlay = make_overlay(image, mask_bin, alpha=0.35)
    contour_overlay = make_contour_overlay(image, mask_bin)

    image_show = normalize_to_uint8(image)

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    axes[0].imshow(image_show, cmap="gray")
    axes[0].set_title(
        f"Image\nmin={image.min()}, max={image.max()}, unique={len(np.unique(image))}"
    )
    axes[0].axis("off")

    axes[1].imshow(mask_bin, cmap="gray")
    axes[1].set_title(
        f"GT Mask\nfg pixels={int(mask_bin.sum())}"
    )
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Image + Mask Overlay")
    axes[2].axis("off")

    axes[3].imshow(contour_overlay)
    axes[3].set_title("Image + Mask Contour")
    axes[3].axis("off")

    fig.suptitle(
        f"{img_path.name}  <->  {label_path.name}",
        fontsize=12
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    # =========================
    # 1. 路径设置
    # =========================
    img_dir = Path("nnUNet_raw/Dataset001_BUSI/imagesTr")
    label_dir = Path("nnUNet_raw/Dataset001_BUSI/labelsTr")
    out_dir = Path("check_pairs")

    out_dir.mkdir(parents=True, exist_ok=True)

    # =========================
    # 2. 参数设置
    # =========================
    num_samples = 30
    random_seed = 2026

    random.seed(random_seed)

    # =========================
    # 3. 收集图像
    # =========================
    img_files = sorted(img_dir.glob("*_0000.png"))

    if len(img_files) == 0:
        raise RuntimeError(f"No images found in {img_dir}")

    print(f"Total images found: {len(img_files)}")

    sample_files = random.sample(
        img_files,
        min(num_samples, len(img_files))
    )

    # =========================
    # 4. 可视化
    # =========================
    valid_count = 0
    missing_count = 0
    mismatch_count = 0

    for img_path in sample_files:
        case_id = img_path.name.replace("_0000.png", "")
        label_path = label_dir / f"{case_id}.png"

        if not label_path.exists():
            print(f"[Missing label] {img_path.name}")
            missing_count += 1
            continue

        try:
            save_path = out_dir / f"{case_id}_pair.png"
            visualize_one_case(img_path, label_path, save_path)
            valid_count += 1
            print(f"[Saved] {save_path}")
        except Exception as e:
            mismatch_count += 1
            print(f"[Error] {img_path.name}: {e}")

    print("\nDone.")
    print(f"Saved visualizations: {valid_count}")
    print(f"Missing labels: {missing_count}")
    print(f"Errors: {mismatch_count}")
    print(f"Output folder: {out_dir.resolve()}")


if __name__ == "__main__":
    main()