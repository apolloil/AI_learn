import os
import gzip
import struct
from PIL import Image
import numpy as np

'''
把raw的二进制文件，转化为ImageFolder格式的数据集
'''

def convert_to_images(data_path, label_path, output_dir):
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 加载图像数据
    with gzip.open(data_path, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)

    # 加载标签数据
    with gzip.open(label_path, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)

    # 保存为图像文件
    for i in range(num):
        img = Image.fromarray(images[i])
        class_dir = os.path.join(output_dir, str(labels[i]))
        os.makedirs(class_dir, exist_ok=True)
        img.save(os.path.join(class_dir, f"image_{i}.png"))

    print(f"已保存 {num} 张图像到 {output_dir}")


# 转换训练集
convert_to_images(
    '../data/FashionMNIST/raw/train-images-idx3-ubyte.gz',
    './raw/train-labels-idx1-ubyte.gz',
    './images/train'
)

# 转换测试集
convert_to_images(
    '../data/FashionMNIST/raw/t10k-images-idx3-ubyte.gz',
    './raw/t10k-labels-idx1-ubyte.gz',
    './images/test'
)