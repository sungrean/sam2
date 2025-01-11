import torch
import cv2
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import numpy as np
import matplotlib.pyplot as plt
import threading
import queue
import time

# 定义绘制掩码的函数
def show_mask(mask, frame, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])    
    # 确保 mask 是二维数组
    mask = mask.squeeze()    
    # 创建一个与 frame 同尺寸的掩码图像
    mask_image = np.zeros_like(frame, dtype=np.float32)    
    # 将 mask 应用到 mask_image 上
    mask_image[mask > 0] = color[:3]    
    # 将 mask_image 转换为 uint8 类型
    mask_image = (mask_image * 255).astype(np.uint8)    
    # 使用 cv2.addWeighted 进行加权叠加
    return cv2.addWeighted(frame, 1, mask_image, 0.6, 0)

# 加载模型和配置
checkpoint = "./checkpoints/sam2.1_hiera_base_plus.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
model = build_sam2(model_cfg, checkpoint)
predictor = SAM2ImagePredictor(model)

# 创建一个队列来存储从相机获取的图像
frame_queue = queue.Queue(maxsize=100)
# 创建一个队列来存储处理后的图像
processed_frame_queue = queue.Queue(maxsize=100)
# 创建一个线程来从相机获取图像
def capture_frames():
    cap = cv2.VideoCapture(0)
    last_time = time.time()
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法获取图像")
            break
        frame_queue.put(frame)
        frame_count += 1
        if frame_count % 10 == 0:
            current_time = time.time()
            fps = 10 / (current_time - last_time)
            print(f"Capture FPS: {fps:.2f}")
            last_time = current_time
    cap.release()

# 创建一个线程来处理图像并绘制掩码
def process_frames():
    last_time = time.time()
    processed_count = 0
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                predictor.set_image(frame_rgb)
                masks, _, _ = predictor.predict()
            for mask in masks:
                frame = show_mask(mask, frame, random_color=True)
            processed_frame_queue.put(frame)
            processed_count += 1
            if processed_count % 10 == 0:
                current_time = time.time()
                fps = 10 / (current_time - last_time)
                print(f"Process FPS: {fps:.2f}")
                print(f"Processed: {processed_count}, Remaining: {frame_queue.qsize()}")
                last_time = current_time
        else:
            time.sleep(0.001)

# 主线程中显示处理后的图像
def display_frames():
    plt.ion()  # 开启交互模式
    while True:
        if not processed_frame_queue.empty():
            frame = processed_frame_queue.get()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            plt.imshow(frame_rgb)
            plt.axis("off")
            plt.show()
            plt.pause(0.001)
            plt.clf()
        else:
            time.sleep(0.001)

# 创建并启动线程
capture_thread = threading.Thread(target=capture_frames)
process_thread = threading.Thread(target=process_frames)

capture_thread.start()
process_thread.start()

# 在主线程中显示处理后的图像
display_frames()

# 等待线程结束
capture_thread.join()
process_thread.join()

plt.close()