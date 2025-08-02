# check_environment.py
import sys
import numpy as np
import torch

print("===== 环境报告 =====")
print(f"Python: {sys.version}")
print(f"NumPy: {np.__version__}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.version.cuda}")
print(f"CUDA可用: {torch.cuda.is_available()}")
print(f"设备: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# 测试张量转换
tensor = torch.tensor([1, 2, 3]).cuda()
array = tensor.cpu().numpy()
print(f"张量转NumPy测试: {array}")