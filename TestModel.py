import os
import re
import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torchvision import transforms

VIDEO_DIR = "./your_dataset"
MODEL_PATH = "model_epoch_10.pth"
OUTPUT_VIDEO = "generated.mp4"
IMG_SIZE = (256, 144)
FPS = 15
FRAMES_TO_GENERATE = 60 * FPS
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class FramePredictor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder1 = torch.nn.Sequential(torch.nn.Conv2d(3, 64, 4, 2, 1), torch.nn.ReLU())     # 256x144 -> 128x72
        self.encoder2 = torch.nn.Sequential(torch.nn.Conv2d(64, 128, 4, 2, 1), torch.nn.ReLU())   # 128x72 -> 64x36
        self.encoder3 = torch.nn.Sequential(torch.nn.Conv2d(128, 256, 4, 2, 1), torch.nn.ReLU())  # 64x36 -> 32x18

        self.joystick_fc = torch.nn.Sequential(
            torch.nn.Linear(2, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 256 * 18 * 32),
            torch.nn.ReLU()
        )

        self.decoder1 = torch.nn.Sequential(torch.nn.ConvTranspose2d(512, 128, 4, 2, 1), torch.nn.ReLU())  # 18x32 -> 36x64
        self.decoder2 = torch.nn.Sequential(torch.nn.ConvTranspose2d(256, 64, 4, 2, 1), torch.nn.ReLU())   # 36x64 -> 72x128
        self.decoder3 = torch.nn.Sequential(torch.nn.ConvTranspose2d(128, 3, 4, 2, 1), torch.nn.Sigmoid()) # 72x128 -> 144x256

    def forward(self, frame_t, joystick):
        e1 = self.encoder1(frame_t)  # [B, 64, 72, 128]
        e2 = self.encoder2(e1)       # [B, 128, 36, 64]
        e3 = self.encoder3(e2)       # [B, 256, 18, 32]

        z_js = self.joystick_fc(joystick).view(-1, 256, 18, 32)
        z = torch.cat([e3, z_js], dim=1)  # [B, 512, 18, 32]

        d1 = self.decoder1(z)             # [B, 128, 36, 64]
        d1 = torch.cat([d1, e2], dim=1)   # [B, 256, 36, 64]
        d2 = self.decoder2(d1)            # [B, 64, 72, 128]
        d2 = torch.cat([d2, e1], dim=1)   # [B, 128, 72, 128]
        return self.decoder3(d2)          # [B, 3, 144, 256]

# === Получить первый кадр и путь к CSV
video = sorted([f for f in os.listdir(VIDEO_DIR) if f.endswith(".mp4")])[0]
timestamp = re.search(r"(\d{8}_\d{6})", video).group(1)
csv_path = os.path.join(VIDEO_DIR, f"{timestamp}_Joystick.csv")
video_path = os.path.join(VIDEO_DIR, video)

print(f"[+] Видео: {video}")
print(f"[+] CSV:   {os.path.basename(csv_path)}")

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cap.release()

if not ret:
    raise RuntimeError("Не удалось прочитать первый кадр")

frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), IMG_SIZE)
frame_tensor = transforms.ToTensor()(frame).unsqueeze(0).to(DEVICE)

# === Загрузка джойстика
js = pd.read_csv(csv_path)
js['t'] -= js['t'].min()
yaw = np.interp(np.arange(FRAMES_TO_GENERATE) / FPS, js['t'], js['yaw_norm'])
acc = np.interp(np.arange(FRAMES_TO_GENERATE) / FPS, js['t'], js['acc_norm'])

# === Загрузка модели
model = FramePredictor().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# === Запись видео
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, FPS, IMG_SIZE)

print("[*] Генерация...")

with torch.no_grad():
    current_frame = frame_tensor
    for i in tqdm(range(FRAMES_TO_GENERATE)):
        joystick_input = torch.tensor([[yaw[i], acc[i]]], dtype=torch.float32).to(DEVICE)
        next_frame = model(current_frame, joystick_input).clamp(0, 1)

        img = (next_frame.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        out.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        current_frame = next_frame

out.release()
print(f"[✓] Сгенерировано видео: {OUTPUT_VIDEO}")
