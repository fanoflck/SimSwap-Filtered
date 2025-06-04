import os
import cv2
from tqdm import tqdm

# Input and output directories
input_dir = "/mnt/data1/szl/OPPO/FF++/raw_mp4_only"
output_dir = "/mnt/data1/szl/OPPO/FF++/raw_frames"
os.makedirs(output_dir, exist_ok=True)

def extract_middle_frame(video_path, output_image_path):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        return False
    cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return False

    # Save the original frame directly
    cv2.imwrite(output_image_path, frame)
    return True

video_files = [f for f in os.listdir(input_dir) if f.endswith(".mp4")]

success_count = 0
for filename in tqdm(video_files, desc="Extracting frames"):
    video_path = os.path.join(input_dir, filename)
    image_path = os.path.join(output_dir, filename.replace(".mp4", ".jpg"))
    if extract_middle_frame(video_path, image_path):
        success_count += 1

print(f"Successfully extracted {success_count} frames out of {len(video_files)} videos.")