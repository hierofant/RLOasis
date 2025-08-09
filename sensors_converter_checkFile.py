# joystick_overlay.py
import cv2
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter.filedialog import askopenfilename, asksaveasfilename
from collections import deque

# ─────────── Выбор файлов ────────────
def choose_file(title, filetypes):
    root = tk.Tk(); root.withdraw()
    return askopenfilename(title=title, filetypes=filetypes)

video_path = choose_file("Выберите видео", [("Все видео", "*.mp4;*.avi;*.mov"), ("MP4", "*.mp4"), ("AVI", "*.avi")])
joy_path   = choose_file("Выберите CSV с готовым джой-стиком", [("CSV файлы", "*.csv")])

if not video_path or not joy_path:
    raise SystemExit("Файл(ы) не выбраны!")

# ─────────── Куда сохранить результат ──
root = tk.Tk(); root.withdraw()
out_path = asksaveasfilename(
    title="Сохранить итоговое видео как…",
    defaultextension=".mp4",
    filetypes=[("MP4-файл", "*.mp4"), ("AVI-файл", "*.avi")])
if not out_path:
    raise SystemExit("Не выбрано имя выходного файла!")

# ─────────── Загрузка джой-стика ───────
# Ожидаем заголовок t,yaw_norm,acc_norm
joy = pd.read_csv(joy_path)
if set(joy.columns) != {"t", "yaw_norm", "acc_norm"}:
    raise ValueError("CSV должен содержать колонки: t, yaw_norm, acc_norm")

# ─────────── Подготовка видео ──────────
cap = cv2.VideoCapture(video_path)
fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
dt     = 1.0 / fps

# —— VideoWriter (mp4v ≈ кросс-платформенный) ——
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
if not writer.isOpened():
    raise RuntimeError("Не удалось открыть VideoWriter!")

# ─────────── Параметры сглаживания (опц.) ─
alpha_v = 0.15          # 0 → мгновенно, 1 → сильное сглаживание
vec     = np.zeros(2)   # стартовая точка по центру

# ─────────── Основной цикл ─────────────
frame_idx   = 0
max_frames  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or int(fps*3600)  # по длительности видео

while cap.isOpened() and frame_idx < max_frames:
    ok, frame = cap.read()
    if not ok:
        break

    t = frame_idx / fps            # время текущего кадра
    frame_idx += 1

    # ---------- ближайшее значение из CSV
    j = joy.iloc[(joy["t"] - t).abs().idxmin()]
    raw_vec = np.array([j["yaw_norm"], j["acc_norm"]], dtype=float)

    # ---------- (необязательно) ограничим длину и сгладим
    if np.linalg.norm(raw_vec) > 1:
        raw_vec /= np.linalg.norm(raw_vec)
    vec = alpha_v * raw_vec + (1 - alpha_v) * vec

    # ────────── Отрисовка поверх видео ───
    h, w = frame.shape[:2]
    cx, cy, Rjoy = w // 2, h // 2, 100

    # Знак такой же, как в изначальном скрипте
    px = int(cx - vec[0] * Rjoy)   # yaw: +влево, –вправо
    py = int(cy + vec[1] * Rjoy)   # acc: +вниз,  –вверх

    cv2.circle(frame, (cx, cy), Rjoy, (200, 200, 200), 2)
    cv2.circle(frame, (px, py), 6, (0, 0, 255), -1)
    cv2.putText(frame, f"ACC  (↑/↓): {vec[1]:+.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"YAW (←/→): {vec[0]:+.2f}", (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # ---------- вывод и запись
    cv2.imshow("Просмотр джой-стика", frame)
    writer.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
writer.release()
cv2.destroyAllWindows()
print(f"Готово! Сохранено в: {out_path}")
