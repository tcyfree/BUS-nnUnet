from pathlib import Path
import pandas as pd
from PIL import Image
from tqdm import tqdm


def convert_to_grayscale_png(src_path, dst_path):
    """
    B超原图：只转灰度，不二值化。
    保存为 png，方便 nnU-Net 预测。
    """
    img = Image.open(src_path).convert("L")
    img.save(dst_path)


def main():
    # =========================
    # 1. 修改这里：你的 Excel 路径
    # =========================
    # excel_path = Path("./datasets/3_Malignant_all.xlsx")
    # excel_path = Path("./datasets/3_Benign_all.xlsx")
    excel_path = Path("./datasets/4_Malignant_all.xlsx")

    # 如果 Excel 和图像路径是相对路径，例如 ./datasets/...
    # 这里设置为项目根目录
    project_root = Path(".").resolve()

    # =========================
    # 2. 输出目录
    # =========================
    # output_dir = Path("nnUNet_predict_input/3_M")
    # output_dir = Path("nnUNet_predict_input/3_B")
    output_dir = Path("nnUNet_predict_input/4_M")
    output_dir.mkdir(parents=True, exist_ok=True)

    # mapping_csv = Path("nnUNet_predict_input/3_M/3_M_predict_mapping.csv")
    # mapping_csv = Path("nnUNet_predict_input/3_B/3_B_predict_mapping.csv")
    mapping_csv = Path("nnUNet_predict_input/4_M/4_M_predict_mapping.csv")

    # =========================
    # 3. 读取 Excel
    # =========================
    df = pd.read_excel(excel_path)

    required_cols = ["patient_id", "b_mode_image"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Excel 缺少必要列: {col}")

    records = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Preparing nnU-Net input"):
        patient_id = str(row["patient_id"])
        b_mode_path = str(row["b_mode_image"])

        if pd.isna(b_mode_path) or b_mode_path.strip() == "":
            print(f"[Warning] Empty b_mode_image for patient_id={patient_id}")
            continue

        src_path = Path(b_mode_path)

        # 处理 ./datasets/... 这种相对路径
        if not src_path.is_absolute():
            src_path = project_root / src_path

        if not src_path.exists():
            print(f"[Warning] Image not found: {src_path}")
            continue

        case_id = f"PRED_{len(records) + 1:04d}"

        # nnU-Net 预测输入命名：case_id_0000.png
        dst_name = f"{case_id}_0000.png"
        dst_path = output_dir / dst_name

        convert_to_grayscale_png(src_path, dst_path)

        records.append(
            {
                "case_id": case_id,
                "patient_id": patient_id,
                "original_b_mode_image": str(src_path),
                "nnunet_input": str(dst_path),
                "expected_mask_name": f"{case_id}.png",
            }
        )

    mapping_df = pd.DataFrame(records)
    mapping_df.to_csv(mapping_csv, index=False, encoding="utf-8-sig")

    print("\nDone.")
    print(f"Prepared images: {len(records)}")
    print(f"nnU-Net input folder: {output_dir.resolve()}")
    print(f"Mapping saved to: {mapping_csv.resolve()}")


if __name__ == "__main__":
    main()