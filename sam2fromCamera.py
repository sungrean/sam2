import torch
import cv2
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# 加载模型和配置
checkpoint = "checkpoints/sam2.1_hiera_small.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
model = build_sam2(model_cfg, checkpoint)
predictor = SAM2ImagePredictor(model)

# 打开相机
cap = cv2.VideoCapture(0)

while True:
    # 读取相机图像
    ret, frame = cap.read()

    if not ret:
        print("无法获取图像")
        break

    # 将图像从BGR格式转换为RGB格式（假设模型需要RGB格式）
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        # 设置图像到预测器
        predictor.set_image(frame_rgb)

        # 这里假设使用一个简单的点提示作为示例，你可以根据实际需求修改
        input_prompts = [(100, 100)]  # 示例点坐标，可调整
        masks, _, _ = predictor.predict(input_prompts)

    # 可视化掩码（这里只是简单地将掩码绘制在图像上，你可以根据需要改进可视化效果）
    for mask in masks:
        mask = mask.squeeze().cpu().numpy()
        mask = (mask * 255).astype('uint8')
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)

    # 显示结果图像
    cv2.imshow('Result', frame)

    # 按下 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放相机资源并关闭窗口
cap.release()
cv2.destroyAllWindows()