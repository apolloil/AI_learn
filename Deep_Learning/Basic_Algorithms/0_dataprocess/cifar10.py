import os
import pandas as pd
import shutil
from tqdm import tqdm

# 设置路径
train_dir = 'train'  # 原始训练图像文件夹
csv_path = 'trainLabels.csv'  # 标签文件
output_dir = 'train_sorted'  # 输出目录

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# 读取CSV标签文件
labels_df = pd.read_csv(csv_path)

# 创建类别子文件夹
class_names = labels_df['label'].unique()
for class_name in class_names:
    os.makedirs(os.path.join(output_dir, class_name), exist_ok=True)

# 移动图片到对应类别文件夹
for idx, row in tqdm(labels_df.iterrows(), total=len(labels_df)):
    image_id = row['id']
    class_name = row['label']

    # 源文件路径 (假设图片是PNG格式)
    src_path = os.path.join(train_dir, f"{image_id}.png")

    # 目标路径
    dest_path = os.path.join(output_dir, class_name, f"{image_id}.png")

    # 复制文件（保留原始文件）
    shutil.copy2(src_path, dest_path)

print("预处理完成！图像已分类存储在:", output_dir)
