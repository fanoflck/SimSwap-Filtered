import cv2
import matplotlib.pyplot as plt
import os

# 视频路径
video_path = "/mnt/data1/szl/OPPO/FF++/raw_mp4_only/000.mp4"

# 输出图像保存目录
save_dir = "./output_frames"
os.makedirs(save_dir, exist_ok=True)

# 打开视频并读取第一帧
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cap.release()

if not ret:
    raise ValueError("无法读取视频帧")

# 转换 BGR → RGB
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# ✅ 保存图像
cv2.imwrite(os.path.join(save_dir, "raw_bgr_frame.jpg"), frame)  # OpenCV 默认保存为 BGR
cv2.imwrite(os.path.join(save_dir, "corrected_rgb_frame.jpg"), cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
