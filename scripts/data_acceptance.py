import os
import cv2
import torch
from facenet_pytorch import MTCNN
from tqdm import tqdm

input_root = "/mnt/data1/szl/OPPO/output"
output_root = "/mnt/data1/szl/OPPO/output_acceptance"
os.makedirs(output_root, exist_ok=True)

MIN_FACE_FRAMES = 15   # 至少多少帧有人脸才算一个有效片段（可调）
SAMPLE_DURATION = 3    # 每个切割视频长度（秒）
FPS = 25               # 默认帧率（可自动获取）

device = "cuda" if torch.cuda.is_available() else "cpu"
mtcnn = MTCNN(keep_all=False, device=device)

def detect_faces_in_frame(frame):
    boxes, _ = mtcnn.detect(frame)
    return boxes is not None

def extract_valid_segments(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or FPS
    frames = []
    valid_segments = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()

    total_frames = len(frames)
    current = 0
    segment = []

    while current < total_frames:
        frame = frames[current]
        if detect_faces_in_frame(frame):
            segment.append(frame)
        else:
            if len(segment) >= MIN_FACE_FRAMES:
                valid_segments.append(segment)
            segment = []
        current += 1

    if len(segment) >= MIN_FACE_FRAMES:
        valid_segments.append(segment)

    return valid_segments, int(fps)

def save_segments(segments, fps, base_name, save_dir):
    count = 1
    for segment in segments:
        total_frames = len(segment)
        clip_len = fps * SAMPLE_DURATION

        for i in range(0, total_frames, clip_len):
            clip = segment[i:i+clip_len]
            if len(clip) < clip_len:
                continue

            h, w, _ = clip[0].shape
            out_path = os.path.join(save_dir, f"{base_name}_{count}.mp4")
            out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

            for frame in clip:
                out.write(frame)
            out.release()
            print(f"✅ 保存片段：{out_path}")
            count += 1

for folder in sorted(os.listdir(input_root)):
    folder_path = os.path.join(input_root, folder)
    if not os.path.isdir(folder_path):
        continue

    for file in os.listdir(folder_path):
        if not file.endswith(".mp4"):
            continue
        video_path = os.path.join(folder_path, file)
        base_name = os.path.splitext(file)[0]

        print(f"🔍 处理视频：{video_path}")
        segments, fps = extract_valid_segments(video_path)
        if not segments:
            print(f"⚠️ 无有效人脸片段：{video_path}")
            continue

        save_segments(segments, fps, base_name, output_root)
