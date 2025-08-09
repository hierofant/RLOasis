import cv2
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog
import time
import os

# ---------- Диалоги выбора ----------
def choose_file(title, save=False, default_ext=""):
    root = tk.Tk()
    root.withdraw()
    if save:
        return filedialog.asksaveasfilename(
            title=title,
            defaultextension=default_ext,
            filetypes=[("MP4-файл", "*.mp4"), ("AVI-файл", "*.avi"), ("Все файлы", "*.*")]
        )
    else:
        return filedialog.askopenfilename(title=title)

video_path  = choose_file("Выберите исходное видео")
gyro_path   = choose_file("Выберите CSV-файл с гироскопом")
accel_path  = choose_file("Выберите CSV-файл с акселерометром")
out_path    = choose_file("Куда сохранить обработанное видео", save=True, default_ext=".mp4")  # NEW

# ---------- Загрузка сенсорных данных ----------
gyro  = pd.read_csv(gyro_path,  header=None, names=['gx', 'gy', 'gz', 'timestamp'])
accel = pd.read_csv(accel_path, header=None, names=['ax', 'ay', 'az', 'timestamp'])

# перевод наносекунд → секунд и синхронизация нуля
gyro['timestamp']  /= 1e9
accel['timestamp'] /= 1e9
t0 = min(gyro['timestamp'].min(), accel['timestamp'].min())
gyro['timestamp']  -= t0
accel['timestamp'] -= t0

# ---------- Видео вход/выход ----------
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
if fps < 1 or fps > 240:   # fallback
    fps = 29.97

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')                             # NEW
writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))              # NEW
if not writer.isOpened():
    raise IOError("Не удалось открыть файл для записи: " + out_path)

# ---------- Параметры отрисовки ----------
frame_index = 0
prev_az = prev_gy = None

pos = np.array([0.0, 0.0])
radius = 100
smoothness = 0.2
vec_smoothness = 0.5

smoothed_delta = np.array([0.0, 0.0])
prev_time = time.time()

duration_limit = 1 * 20                # 5 минут в секундах  # NEW

# ---------- Основной цикл ----------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    timestamp = frame_index / fps
    if timestamp > duration_limit:      # стоп после 5 минут   # NEW
        break

    frame_index += 1

    # ближайшие строки сенсоров во времени
    gyro_row  = gyro.iloc[(gyro['timestamp']  - timestamp).abs().argsort()[:1]]
    accel_row = accel.iloc[(accel['timestamp'] - timestamp).abs().argsort()[:1]]

    gx = float(gyro_row ['gx'])
    gy = float(gyro_row ['gy'])
    gz = float(gyro_row ['gz'])
    ax = float(accel_row['ax'])
    ay = float(accel_row['ay'])
    az = float(accel_row['az'])

    if prev_az is not None and prev_gy is not None:
        delta_accel = az - prev_az
        delta_gyro  = gy - prev_gy

        accel_change = delta_accel / (abs(prev_az) + 1e-5)
        gyro_change  = delta_gyro  / (abs(prev_gy) + 1e-5)

        accel_change = np.clip(accel_change, -1, 1)
        gyro_change  = np.clip(gyro_change,  -1, 1)

        raw_delta = np.array([gyro_change, accel_change])
        smoothed_delta = vec_smoothness * raw_delta + (1 - vec_smoothness) * smoothed_delta
        pos += smoothed_delta * smoothness

        if np.linalg.norm(pos) > 1:
            pos /= np.linalg.norm(pos)

    prev_az, prev_gy = az, gy

    # ---------- Визуализация ----------
    center = (w // 2, h // 2)
    point  = (int(center[0] + pos[0] * radius),
              int(center[1] - pos[1] * radius))

    cv2.circle(frame, center, radius, (200, 200, 200), 2)
    cv2.circle(frame, point,   6,      (0,   0, 255),  -1)
    cv2.putText(frame,
                f"Δgyro: {smoothed_delta[0]: .3f}   Δaccel: {smoothed_delta[1]: .3f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # ---------- Запись и показ ----------
    writer.write(frame)               # NEW – сохраняем кадр
    cv2.imshow("Сглаженный вектор движения", frame)

    # корректный delay для realtime-воспроизведения
    elapsed = time.time() - prev_time
    delay = max(1, int((1 / fps - elapsed) * 1000))
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break
    prev_time = time.time()

# ---------- Завершение ----------
cap.release()
writer.release()                       # NEW
cv2.destroyAllWindows()

print(f"Готово! Видео сохранено в:\n{os.path.abspath(out_path)}")
