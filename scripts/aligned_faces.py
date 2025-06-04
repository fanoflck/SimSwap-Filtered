import os
import cv2
import torch
from PIL import Image
from tqdm import tqdm
from facenet_pytorch import MTCNN

# 输入输出目录
input_dir = "/mnt/data1/szl/OPPO/FF++/raw_mp4_only"
output_dir = "/mnt/data1/szl/OPPO/FF++/raw_faces"
os.makedirs(output_dir, exist_ok=True)

mtcnn = MTCNN(keep_all=True, device='cuda' if torch.cuda.is_available() else 'cpu')

def extract_middle_face(video_path, output_image_path, expand_ratio=0.2):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        return False
    cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return False

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)

    # 获取所有人脸框和 landmarks（这里只用框）
    boxes, _ = mtcnn.detect(img_pil)

    if boxes is None or len(boxes) == 0:
        return False

    # 取第一个人脸框
    x1, y1, x2, y2 = boxes[0]
    h, w = frame.shape[:2]

    # 扩展 bbox（例如扩大 30%）
    bw = x2 - x1
    bh = y2 - y1
    x1 = max(int(x1 - bw * expand_ratio), 0)
    y1 = max(int(y1 - bh * expand_ratio), 0)
    x2 = min(int(x2 + bw * expand_ratio), w)
    y2 = min(int(y2 + bh * expand_ratio), h)

    # 裁剪并保存（可 resize 到标准尺寸）
    face_crop = frame[y1:y2, x1:x2]
    face_crop = cv2.resize(face_crop, (224, 224))  # 可选
    cv2.imwrite(output_image_path, face_crop)

    return True

video_files = [f for f in os.listdir(input_dir) if f.endswith(".mp4")]

success_count = 0
for filename in tqdm(video_files, desc="Extracting faces"):
    video_path = os.path.join(input_dir, filename)
    image_path = os.path.join(output_dir, filename.replace(".mp4", ".jpg"))
    if extract_middle_face(video_path, image_path):
        success_count += 1
