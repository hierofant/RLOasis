import os, cv2
from glob import glob

VIDEO_DIR = "./your_dataset"
IMG_SIZE = (128, 72)
PREVIEW_DIR = "./preview"

os.makedirs(PREVIEW_DIR, exist_ok=True)
video_files = sorted(glob(os.path.join(VIDEO_DIR, "*.mp4")))
assert video_files, "Нет видео в папке"

video_path = video_files[0]
cap = cv2.VideoCapture(video_path)

print(f"[+] Сохраняю превью из: {os.path.basename(video_path)}")
for i in range(10):
    ret, frame = cap.read()
    if not ret:
        print("[-] Кадров меньше 10!")
        break
    frame = cv2.resize(frame, IMG_SIZE)
    out_path = os.path.join(PREVIEW_DIR, f"frame_{i:02}.jpg")
    cv2.imwrite(out_path, frame)
    print(f"  └─ {out_path}")

cap.release()
print(f"[✓] Готово. Кадры в: {PREVIEW_DIR}")
