import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

# モデルの読み込み
model_name = "LiheYoung/depth-anything-small-hf"
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModelForDepthEstimation.from_pretrained(model_name)

# 画像の読み込み
image_path = "2.png"  # 入力画像
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # OpenCV は BGR なので RGB に変換

# 画像を前処理
inputs = processor(images=image, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
    
# 深度マップを取得し、128x128 にリサイズ
depth = outputs.predicted_depth
depth = depth.squeeze().cpu().numpy()
depth = cv2.resize(depth, (128, 128))

# 深度を正規化（可視化のため）
depth_min, depth_max = depth.min(), depth.max()
depth_vis = (depth - depth_min) / (depth_max - depth_min)  # 0-1 に正規化

# 深度マップを画像として保存
depth_uint8 = (depth_vis * 255).astype(np.uint8)
cv2.imwrite("3.png", depth_uint8)

print("深度推定結果を '3.png' に保存しました。")
