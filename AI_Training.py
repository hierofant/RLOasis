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
        # совместим обе версии, чтобы ничего не упало
        self.index = []          # из main
        self.samples = self.index  # алиас для старого кода
        self.sequences = []      # из другой ветки (если где-то используется)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        self.img_size = img_size


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
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_skip = int(round(input_fps / TARGET_FPS)) if input_fps > 0 else 1
            usable_frames = total_frames // frame_skip

            if usable_frames < PREDICT_AHEAD + 1:
                cap.release()
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

            frames = []
            for i in range(usable_frames):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i * frame_skip)
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, self.img_size)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = self.transform(frame)
                frames.append(frame)
            cap.release()

            self.sequences.append({"frames": frames, "yaw": yaw_interp, "acc": acc_interp})

            for i in range(len(frames) - PREDICT_AHEAD):
                f1 = frames[i]
                f2 = frames[i + PREDICT_AHEAD]
                js = torch.tensor([yaw_interp[i], acc_interp[i]], dtype=torch.float32)
                self.samples.append((f1, js, f2))

        print(f"\n[✓] Всего пар: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


def save_epoch_video(model, dataset, epoch, seq_idx=0, max_frames=120, out_dir="generated"):
    model.eval()
    os.makedirs(out_dir, exist_ok=True)
    seq = dataset.sequences[seq_idx]
    frames = seq["frames"]
    yaw = seq["yaw"]
    acc = seq["acc"]
    current = frames[0].unsqueeze(0).to(DEVICE)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path = os.path.join(out_dir, f"epoch_{epoch+1}.mp4")
    writer = cv2.VideoWriter(out_path, fourcc, TARGET_FPS, IMG_SIZE)
    steps = min(len(frames) - 1, max_frames)
    for i in range(steps):
        js = torch.tensor([yaw[i], acc[i]], dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            pred = model(current, js)
        frame = pred[0].cpu()
        img = (frame * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).numpy()
        img = (img * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        writer.write(img)
        current = pred
    writer.release()
    model.train()
    print(f"[✓] Сохранён пример работы модели: {out_path}")


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
        save_epoch_video(model, dataset, epoch)

if __name__ == "__main__":
    print("[✓] Запуск обучения с предсказанием вперёд на", PREDICT_AHEAD, "кадров")
    train()
