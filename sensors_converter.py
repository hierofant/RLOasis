import cv2
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter.filedialog import askopenfilename, asksaveasfilename   # ← NEW
import time
from collections import deque

# ─────────── Выбор файлов ────────────
def choose_file(title):
    root = tk.Tk(); root.withdraw()
    return askopenfilename(title=title)

video_path = choose_file("Выбери видео")
gyro_path  = choose_file("Выбери CSV с гироскопом")
accel_path = choose_file("Выбери CSV с акселерометром")

# ─────────── Куда сохранить результат ──
root = tk.Tk(); root.withdraw()
out_path = asksaveasfilename(
    title="Сохранить итоговое видео как…",
    defaultextension=".mp4",
    filetypes=[("MP4-файл", "*.mp4"), ("AVI-файл", "*.avi")])
if not out_path:
    raise SystemExit("Не выбрано имя выходного файла!")

# ─────────── Загрузка датчиков ─────────
gyro  = pd.read_csv(gyro_path,  header=None, names=['gx','gy','gz','ts'])
accel = pd.read_csv(accel_path, header=None, names=['ax','ay','az','ts'])
gyro['ts']  /= 1e9; accel['ts'] /= 1e9
t0 = min(gyro['ts'].min(), accel['ts'].min())
gyro['ts']  -= t0;  accel['ts'] -= t0

# ─────────── Калибровка (1-я секунда) ─
first_acc = accel[accel['ts'] < 1.0][['ax','ay','az']].values
g0 = np.mean(first_acc, axis=0)
roll0  = np.arctan2(g0[1], g0[2])
pitch0 = np.arctan2(-g0[0], np.hypot(g0[1], g0[2]))
cr0, sr0, cp0, sp0 = np.cos(roll0), np.sin(roll0), np.cos(pitch0), np.sin(pitch0)
R0 = np.array([[   cp0, sr0*sp0, cr0*sp0],
               [     0,     cr0,    -sr0],
               [  -sp0, sr0*cp0, cr0*cp0]])
acc_earth0 = (R0 @ first_acc.T).T
lin0       = acc_earth0 - np.array([0,0,9.81])
bias_acc   = np.mean(lin0[:,0])
bias_gz    = np.mean(gyro[gyro['ts'] < 1.0]['gz'])

# ─────────── Подготовка видео ──────────
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
dt  = 1.0 / fps
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# —— VideoWriter (mp4v ≈ кросс-платформенный) ——
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
if not writer.isOpened():
    raise RuntimeError("Не удалось открыть VideoWriter!")

# ─────────── Параметры сглаживания ─────
alpha_g = 0.95
ma_win  = 10
ema_acc = 0.2
ema_yaw = 0.2
buf_acc = deque(maxlen=ma_win)
buf_yaw = deque(maxlen=ma_win)
vec     = np.zeros(2)
alpha_v = 0.1

# ─────────── Основной цикл ─────────────
g_est = first_acc[0].copy()
frame_idx = 0
max_frames = int(fps * 300)        # 5 минут = 300 с

while cap.isOpened() and frame_idx < max_frames:
    ok, frame = cap.read()
    if not ok: break
    t = frame_idx / fps
    frame_idx += 1

    # ---------- ближайшие образцы сенсоров
    i_g = (gyro['ts'] - t).abs().idxmin()
    i_a = (accel['ts'] - t).abs().idxmin()
    gx, gy, gz = gyro.loc[i_g, ['gx','gy','gz']]
    ax, ay, az = accel.loc[i_a, ['ax','ay','az']]

    # ---------- гравитация → roll/pitch
    g_est = alpha_g * g_est + (1-alpha_g) * np.array([ax,ay,az])
    g_n   = g_est / np.linalg.norm(g_est)
    roll  = np.arctan2( g_n[1],  g_n[2])
    pitch = np.arctan2(-g_n[0], np.hypot(g_n[1], g_n[2]))
    cr, sr, cp, sp = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch)
    R = np.array([[   cp, sr*sp, cr*sp],
                  [    0,    cr,   -sr],
                  [  -sp, sr*cp, cr*cp]])

    # ---------- прод. ускорение & yaw-rate
    acc_earth = R @ np.array([ax,ay,az])
    lin_acc   = acc_earth - np.array([0,0,9.81])
    acc_x     = lin_acc[0] - bias_acc
    yaw_rate  = (R @ np.array([gx,gy,gz]))[2] - bias_gz

    # ---------- мягкое сглаживание сигнала
    buf_acc.append(acc_x); buf_yaw.append(yaw_rate)
    ma_acc  = np.mean(buf_acc)
    ma_yaw  = np.mean(buf_yaw)
    s_acc   = ema_acc * ma_acc + (1-ema_acc) * locals().get('s_acc', ma_acc)
    s_yaw   = ema_yaw * ma_yaw + (1-ema_yaw) * locals().get('s_yaw', ma_yaw)

    # ---------- нормализация до [-1,1]
    acc_norm = np.tanh(s_acc / 2.0)
    yaw_norm = np.tanh(s_yaw / 2.0)

    # ---------- финальное сглаживание вектора
    raw_vec = np.array([ yaw_norm, acc_norm ])
    vec     = alpha_v * raw_vec + (1-alpha_v) * vec
    if np.linalg.norm(vec) > 1:
        vec /= np.linalg.norm(vec)

    # ────────── Отрисовка поверх видео ───
    h, w = frame.shape[:2]
    cx, cy, Rjoy = w//2, h//2, 100
    px = int(cx + vec[0] * Rjoy * -1)
    py = int(cy - vec[1] * Rjoy * -1)

    cv2.circle(frame, (cx,cy), Rjoy, (200,200,200), 2)
    cv2.circle(frame, (px,py), 6, (0,0,255), -1)
    cv2.putText(frame, f"ACC  (↑/↓): {vec[1]:+.2f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    cv2.putText(frame, f"YAW (←/→): {vec[0]:+.2f}",
                (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    # ---------- вывод и запись
    cv2.imshow("Realtime-джойстик", frame)
    writer.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
writer.release()
cv2.destroyAllWindows()
print(f"Готово! Сохранено в {out_path}")
