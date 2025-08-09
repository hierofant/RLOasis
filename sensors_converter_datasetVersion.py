#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Быстрая обработка логов IMU → нормированный джойстик (yaw, acc).
Минимальное сглаживание: быстрый, но мягкий отклик.
"""
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter.filedialog import askopenfilename, asksaveasfilename

# ─────────── Функции выбора файлов ──────────
def choose_file(title):
    root = tk.Tk(); root.withdraw()
    return askopenfilename(title=title)

gyro_path  = choose_file("Выбери CSV с гироскопом")
accel_path = choose_file("Выбери CSV с акселерометром")

root = tk.Tk(); root.withdraw()
out_path = asksaveasfilename(
    title="Сохранить CSV с джойстиком как…",
    defaultextension=".csv",
    filetypes=[("CSV-файл", "*.csv")])
if not out_path:
    raise SystemExit("Не выбрано имя выходного файла!")

# ─────────── Загрузка в numpy ───────────────
gyro_df  = pd.read_csv(gyro_path,  header=None, names=['gx','gy','gz','ts'])
accel_df = pd.read_csv(accel_path, header=None, names=['ax','ay','az','ts'])

gyro  = gyro_df.to_numpy(dtype=np.float64)
accel = accel_df.to_numpy(dtype=np.float64)

# ─────────── Нормализация времени ───────────
gyro[:,3]  /= 1e9
accel[:,3] /= 1e9
t0 = min(gyro[:,3].min(), accel[:,3].min())
gyro[:,3]  -= t0
accel[:,3] -= t0
t_max = min(gyro[:,3].max(), accel[:,3].max())

# ─────────── Калибровка по первой секунде ──
mask1 = accel[:,3] < 1.0
first_acc = accel[mask1, :3]

g0 = first_acc.mean(axis=0)
roll0  = np.arctan2(g0[1], g0[2])
pitch0 = np.arctan2(-g0[0], np.hypot(g0[1], g0[2]))
cr0, sr0, cp0, sp0 = np.cos(roll0), np.sin(roll0), np.cos(pitch0), np.sin(pitch0)
R0 = np.array([[   cp0, sr0*sp0, cr0*sp0],
               [     0,     cr0,    -sr0],
               [  -sp0, sr0*cp0, cr0*cp0]])
acc_earth0 = (R0 @ first_acc.T).T
lin0       = acc_earth0 - np.array([0,0,9.81])
bias_acc   = lin0[:,0].mean()
bias_gz    = gyro[gyro[:,3] < 1.0, 2].mean()

# ─────────── Интерполяция на сетку 30 Гц ──
TARGET_FPS = 30.0
dt = 1.0 / TARGET_FPS
ts_grid = np.arange(0, t_max, dt)

gx = np.interp(ts_grid, gyro[:,3],  gyro[:,0])
gy = np.interp(ts_grid, gyro[:,3],  gyro[:,1])
gz = np.interp(ts_grid, gyro[:,3],  gyro[:,2])
ax = np.interp(ts_grid, accel[:,3], accel[:,0])
ay = np.interp(ts_grid, accel[:,3], accel[:,1])
az = np.interp(ts_grid, accel[:,3], accel[:,2])

# ─────────── Параметры фильтра ────────────
alpha_g = 0.5      # быстро адаптируется к наклону
alpha_v = 0.2      # лёгкое сглаживание итогового вектора
vec = np.zeros(2)

# ─────────── Основной цикл ──────────
g_est = first_acc[0].copy()
records = []

for step, t in enumerate(ts_grid):
    gx_, gy_, gz_ = gx[step], gy[step], gz[step]
    ax_, ay_, az_ = ax[step], ay[step], az[step]

    # --- гравитация → roll/pitch
    g_est = alpha_g * g_est + (1 - alpha_g) * np.array([ax_, ay_, az_])
    g_n = g_est / np.linalg.norm(g_est)
    roll  = np.arctan2(g_n[1], g_n[2])
    pitch = np.arctan2(-g_n[0], np.hypot(g_n[1], g_n[2]))
    cr, sr, cp, sp = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch)
    R = np.array([[   cp, sr*sp, cr*sp],
                  [    0,    cr,   -sr],
                  [  -sp, sr*cp, cr*cp]])

    # --- ускорение и вращение в земной системе
    acc_earth = R @ np.array([ax_, ay_, az_])
    lin_acc = acc_earth - np.array([0, 0, 9.81])
    acc_x = lin_acc[0] - bias_acc
    yaw_rate = (R @ np.array([gx_, gy_, gz_]))[2] - bias_gz

    # --- tanh-нормализация
    acc_norm = np.tanh(acc_x / 2.0)
    yaw_norm = np.tanh(yaw_rate / 2.0)

    # --- лёгкое сглаживание финального вектора
    raw_vec = np.array([yaw_norm, acc_norm])
    vec = alpha_v * raw_vec + (1 - alpha_v) * vec
    if vec.dot(vec) > 1.0:
        vec /= np.linalg.norm(vec)

    records.append((t, vec[0], vec[1]))

# ─────────── Сохранение результата ────────
pd.DataFrame(records, columns=['t','yaw_norm','acc_norm']) \
  .to_csv(out_path, index=False, float_format="%.6f")

print(f"Готово! Сохранено в {out_path}")
