import os
import json
import glob
import numpy as np
import cv2

# ================= 配置区域 =================
INPUT_ROOT = "/home/network/Desktop/Project/BreastUS-GPT/segment/3M"       # 原始数据根目录
OUTPUT_IMG_DIR = "datasets/sw/3M/img"       # 原图输出目录
OUTPUT_LABEL_DIR = "datasets/sw/3M/label"   # Mask输出目录

# 标签映射：修改为颜色元组 (B, G, R) - 因为 OpenCV 默认是 BGR 通道顺序
# 例如："lesion" 标记为纯红色 (0, 0, 255)
# 如果想标为纯白色，可以用 (255, 255, 255)
CLASS_MAPPING = {
    "lesion": (255, 255, 255) 
}
# ============================================

def convert_labelme_to_color_mask():
    # 创建输出目录
    os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
    os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)

    # 查找所有 JSON 文件
    json_files = glob.glob(os.path.join(INPUT_ROOT, "*/*.json"))
    print(f"📌 找到 {len(json_files)} 个标注文件，准备转换...")

    for json_path in json_files:
        # 1. 解析路径信息，构建新文件名
        path_parts = os.path.normpath(json_path).split(os.sep)
        patient_id = path_parts[-2]
        base_name = os.path.basename(json_path).replace(".json", "")
        save_name = f"{patient_id}_{base_name}.png"
        
        # 原图路径检查
        tif_path = json_path.replace(".json", ".tif")
        if not os.path.exists(tif_path):
            print(f"⚠️ 跳过：找不到原图 {tif_path}")
            continue

        # 2. 拷贝/转换原图为 PNG
        img = cv2.imread(tif_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"❌ 错误：无法读取图片 {tif_path}")
            continue
        cv2.imwrite(os.path.join(OUTPUT_IMG_DIR, save_name), img)

        # 3. 处理 JSON 生成三通道彩色 Mask
        with open(json_path, 'r', encoding='utf-8') as f:
            label_data = json.load(f)

        h, w = label_data.get('imageHeight'), label_data.get('imageWidth')
        if h is None: 
            h, w = img.shape[:2]

        # 关键修改：初始化全黑的 3 通道 (RGB/BGR) 图像矩阵
        mask = np.zeros((h, w, 3), dtype=np.uint8)

        for shape in label_data['shapes']:
            label = shape['label']
            points = shape['points']
            
            # 获取对应的颜色，不在字典里的标签默认涂成白色
            color = CLASS_MAPPING.get(label, (255, 255, 255))
            
            # 转换为 OpenCV 识别的多边形坐标格式并填充颜色
            pts = np.array(points, np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(mask, [pts], color)

        # 4. 保存彩色 Mask
        cv2.imwrite(os.path.join(OUTPUT_LABEL_DIR, save_name), mask)

    print(f"✅ 全部完成！")
    print(f"📂 原图已保存在: {OUTPUT_IMG_DIR}")
    print(f"📂 Mask已保存在: {OUTPUT_LABEL_DIR}")

if __name__ == "__main__":
    convert_labelme_to_color_mask()