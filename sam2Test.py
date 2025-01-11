import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

# 定义绘制掩码的函数
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    image = cv2.imread("./notebooks/images/cars.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 将图像从BGR转换为RGB

    startTime = time.time()
    predictor.set_image(image)
    # 调用 predict 方法，不传递任何点或框提示
    masks, _, _ = predictor.predict()
    print(f"Time: {time.time() - startTime} seconds")

    # 显示结果
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    for mask in masks:
        show_mask(mask, plt.gca(), random_color=True)
    plt.axis("off")
    plt.show()