# test_gpu.py
import torch
import cv2
import numpy as np
from transformers import AutoModel

print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU设备: {torch.cuda.get_device_name(0)}")
    print(f"CUDA计算能力: {torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor}")

# 测试其他库
print(f"OpenCV版本: {cv2.__version__}")
print(f"NumPy版本: {np.__version__}")

# 测试transformers
try:
    model = AutoModel.from_pretrained("facebook/dinov3_vitb14")
    print("✓ Transformers库工作正常")
except Exception as e:
    print(f"Transformers错误: {e}")