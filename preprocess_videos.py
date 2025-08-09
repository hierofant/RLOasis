import os
import cv2
import shutil
from glob import glob

INPUT_DIR = "./your_dataset"
OUTPUT_DIR = "./processed_dataset"
TARGET_SIZE = (96, 64)  # (width, height)
TARGET_FPS = 15

os.makedirs(OUTPUT_DIR, exist_ok=True)

video_files = sorted(glob(os.path.join(INPUT_DIR, "*.mp4")))
if not video_files:
    print(f"No videos found in {INPUT_DIR}")

for vpath in video_files:
    base = os.path.basename(vpath)
    stem, _ = os.path.splitext(base)
    csv_in = os.path.join(INPUT_DIR, f"{stem}_Joystick.csv")
    out_video = os.path.join(OUTPUT_DIR, base)

    cap = cv2.VideoCapture(vpath)
    if not cap.isOpened():
        print(f"[!] Cannot open {vpath}")
        continue
    input_fps = cap.get(cv2.CAP_PROP_FPS) or TARGET_FPS
    frame_skip = max(1, int(round(input_fps / TARGET_FPS)))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_video, fourcc, TARGET_FPS, TARGET_SIZE)

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % frame_skip == 0:
            frame = cv2.resize(frame, TARGET_SIZE)
            writer.write(frame)
        idx += 1

    cap.release()
    writer.release()

    if os.path.exists(csv_in):
        shutil.copy(csv_in, os.path.join(OUTPUT_DIR, os.path.basename(csv_in)))

print(f"[✓] Preprocessing complete. Output saved to {OUTPUT_DIR}")

