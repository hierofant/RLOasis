import os
import re
import cv2
import torch
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

VIDEO_DIR = "./processed_dataset"
IMG_SIZE = (96, 64)
EPOCHS = 10
BATCH_SIZE = 16
PREDICT_AHEAD = 15
TARGET_FPS = 15
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class FramePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder1 = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),  # W,H → /2
            nn.ReLU()
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1),  # /4
            nn.ReLU()
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(128, 256, 4, 2, 1),  # /8
            nn.ReLU(),
            nn.Conv2d(256, 256, 4, 2, 1),  # /16
            nn.ReLU()
        )

        self.joystick_fc = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU()
        )

        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(384, 128, 4, 2, 1),
            nn.ReLU()
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, 4, 2, 1),
            nn.ReLU()
        )
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 3, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, frame_t, joystick):
        B, C, H, W = frame_t.shape
        e1 = self.encoder1(frame_t)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)

        h16, w16 = e3.shape[2], e3.shape[3]
        js = self.joystick_fc(joystick)
        z_js = js.view(B, 128, 1, 1).expand(B, 128, h16, w16)
        z = torch.cat([e3, z_js], dim=1)

        d1 = self.decoder1(z)
        e2_resized = torch.nn.functional.interpolate(e2, size=d1.shape[2:], mode='bilinear', align_corners=False)
        d1 = torch.cat([d1, e2_resized], dim=1)

        d2 = self.decoder2(d1)
        e1_resized = torch.nn.functional.interpolate(e1, size=d2.shape[2:], mode='bilinear', align_corners=False)
        d2 = torch.cat([d2, e1_resized], dim=1)

        pred_frame = self.decoder3(d2)
        return pred_frame



class VideoJoystickDataset(Dataset):
    def __init__(self, video_dir, img_size=(96, 64)):
        self.index = []
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        self.img_size = img_size
        self.video_dir = video_dir
        self.fps_cache = {}

        video_files = glob(os.path.join(video_dir, "*.mp4"))

        for vid_path in video_files:
            base_name = os.path.basename(vid_path)
            match = re.search(r"(\d{8}_\d{6})", base_name)
            if not match:
                continue

            timestamp = match.group(1)
            csv_path = os.path.join(video_dir, f"{timestamp}_Joystick.csv")
            if not os.path.exists(csv_path):
                continue

            print(f"\n[+] Анализ {base_name}")
            cap = cv2.VideoCapture(vid_path)
            input_fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()

            self.fps_cache[vid_path] = input_fps
            total_frames = int(cv2.VideoCapture(vid_path).get(cv2.CAP_PROP_FRAME_COUNT))
            frame_skip = int(round(input_fps / TARGET_FPS)) if input_fps > 0 else 2
            usable_frames = total_frames // frame_skip

            if usable_frames < PREDICT_AHEAD + 1:
                continue

            joystick = pd.read_csv(csv_path)
            joystick['t'] -= joystick['t'].min()
            t_csv = joystick['t'].to_numpy()
            yaw = joystick['yaw_norm'].to_numpy()
            acc = joystick['acc_norm'].to_numpy()

            duration = usable_frames / TARGET_FPS
            t_video = np.linspace(0, duration, usable_frames)
            yaw_interp = np.interp(t_video, t_csv, yaw)
            acc_interp = np.interp(t_video, t_csv, acc)

            for i in range(usable_frames - PREDICT_AHEAD):
                self.index.append({
                    "vid_path": vid_path,
                    "yaw": yaw_interp[i],
                    "acc": acc_interp[i],
                    "frame_idx": i * frame_skip,
                    "frame_skip": frame_skip
                })

        print(f"\n[✓] Всего пар: {len(self.index)}")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        record = self.index[i]
        cap = cv2.VideoCapture(record["vid_path"])
        f1 = self.read_frame(cap, record["frame_idx"])
        f2 = self.read_frame(cap, record["frame_idx"] + record["frame_skip"] * PREDICT_AHEAD)
        cap.release()

        f1 = self.transform(f1)
        f2 = self.transform(f2)
        js = torch.tensor([record["yaw"], record["acc"]], dtype=torch.float32)
        return f1, js, f2

    def read_frame(self, cap, idx):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            frame = np.zeros((self.img_size[1], self.img_size[0], 3), dtype=np.uint8)
        else:
            frame = cv2.resize(frame, self.img_size)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame


def train():
    print("[*] Загрузка данных...")
    dataset = VideoJoystickDataset(VIDEO_DIR, IMG_SIZE)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    model = FramePredictor().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    mse_loss = nn.MSELoss()
    l1_loss = nn.L1Loss()

    print("\n[*] Начинаем обучение...\n")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for f1, js, f2 in tqdm(loader, desc=f"Эпоха {epoch+1}/{EPOCHS}"):
            f1, js, f2 = f1.to(DEVICE), js.to(DEVICE), f2.to(DEVICE)

            pred = model(f1, js)
            loss = 0.5 * mse_loss(pred, f2) + 0.5 * l1_loss(pred, f2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"[{epoch+1}/{EPOCHS}] Loss: {total_loss / len(loader):.6f}")
        torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    print("[✓] Запуск обучения с предсказанием вперёд на", PREDICT_AHEAD, "кадров")
    train()
