import cv2
import pandas as pd

video_path = "dataset/001.mp4"
gyro = pd.read_csv("dataset/001_gyro.csv", header=None, names=["x", "y", "z", "timestamp"])
accel = pd.read_csv("dataset/001_accel.csv", header=None, names=["x", "y", "z", "timestamp"])

# Длительность видео
cap = cv2.VideoCapture(video_path)
frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
fps = cap.get(cv2.CAP_PROP_FPS)
video_duration = frame_count / fps

# Длительность сенсоров
gyro_duration = (gyro['timestamp'].iloc[-1] - gyro['timestamp'].iloc[0]) / 1e9
accel_duration = (accel['timestamp'].iloc[-1] - accel['timestamp'].iloc[0]) / 1e9

print(f"🎞️  Видео:     {video_duration:.2f} секунд")
print(f"📈 Гироскоп:  {gyro_duration:.2f} секунд")
print(f"📉 Акселером: {accel_duration:.2f} секунд")
