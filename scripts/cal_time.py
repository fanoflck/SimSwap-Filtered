import os
import subprocess
from tqdm import tqdm

def get_video_duration(filepath):
    """使用 ffprobe 获取视频时长（单位：秒）"""
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
             '-of', 'default=noprint_wrappers=1:nokey=1', filepath],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return float(result.stdout.strip())
    except Exception as e:
        print(f"⚠️ 获取 {filepath} 时长失败: {e}")
        return 0.0

def get_total_duration_recursive(root_dir):
    total_duration = 0.0
    mp4_files = []

    # 递归遍历所有子目录
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(".mp4"):
                full_path = os.path.join(dirpath, filename)
                mp4_files.append(full_path)

    for filepath in tqdm(mp4_files, desc="统计视频时长", unit="个"):
        duration = get_video_duration(filepath)
        total_duration += duration

    return total_duration

if __name__ == "__main__":
    directory = "output"
    total_sec = get_total_duration_recursive(directory)
    hours = int(total_sec // 3600)
    minutes = int((total_sec % 3600) // 60)
    seconds = int(total_sec % 60)
    print(f"\n📊 总视频时长：{hours} 小时 {minutes} 分钟 {seconds} 秒")

    