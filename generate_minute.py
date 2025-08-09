# generate_minute.py
# Генерация минуты видео из модели FramePredictor.
# Стартовый кадр: первый кадр видео (--video) или картинка (--image).
# Джойстик: CSV (--csv) с колонками t,yaw_norm,acc_norm. Если не задан, едем на нулях.
#
# Пример:
# python generate_minute.py --ckpt ./checkpoints_shards/fp_mm_epoch010.pt ^
#   --video your_dataset/20250805_174049.mp4 --csv your_dataset/20250805_174049.csv ^
#   --out minute.mp4 --fps 15 --seconds 60 --size 256x144

import os, cv2, argparse, numpy as np, pandas as pd, torch
from model_unified import FramePredictor

def parse_size(s):
    W,H = s.lower().split("x")
    return int(W), int(H)

def read_csv(csv_path):
    if not csv_path or not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    cols = {c.lower(): c for c in df.columns}
    tcol  = cols.get('t') or cols.get('time') or cols.get('timestamp')
    yawc  = cols.get('yaw_norm') or cols.get('yaw')
    accc  = cols.get('acc_norm') or cols.get('acc')
    if tcol is None or yawc is None or accc is None:
        raise ValueError("CSV должен содержать колонки t,yaw_norm,acc_norm")
    t   = df[tcol].to_numpy(dtype=np.float32)
    yaw = df[yawc].to_numpy(dtype=np.float32)
    acc = df[accc].to_numpy(dtype=np.float32)
    return t, yaw, acc

def resize_to_tensor(bgr_img, size):
    # bgr -> rgb, к [0,1], resize до (W,H)
    img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    H, W = size[1], size[0]
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
    ten = torch.from_numpy(img).permute(2,0,1).float() / 255.0
    return ten.unsqueeze(0)  # [1,3,H,W]

def first_frame_from_video(vpath, size):
    cap = cv2.VideoCapture(vpath)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Не могу прочитать первый кадр из {vpath}")
    return resize_to_tensor(frame, size)

def first_frame_from_image(ipath, size):
    img = cv2.imread(ipath, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Не могу прочитать картинку {ipath}")
    return resize_to_tensor(img, size)

def build_joystick(csv_triplet, fps, seconds):
    T = int(fps * seconds)
    if csv_triplet is None:
        return torch.zeros(T, 2)  # нули
    t, yaw, acc = csv_triplet
    t_end = (T - 1) / float(fps)
    t_video = np.arange(0, t_end + 1e-6, 1.0/fps, dtype=np.float32)
    if t_video.shape[0] > T:
        t_video = t_video[:T]
    elif t_video.shape[0] < T:
        pad = T - t_video.shape[0]
        t_video = np.concatenate([t_video, np.full((pad,), t_video[-1] if t_video.size else 0, dtype=np.float32)], 0)
    yaw_i = np.interp(t_video, t, yaw).astype(np.float32)
    acc_i = np.interp(t_video, t, acc).astype(np.float32)
    joys = np.stack([yaw_i, acc_i], axis=1)
    return torch.from_numpy(joys)  # [T,2]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="путь к чекпоинту .pt (fp_mm_epochXXX.pt)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--video", help="видео для первого кадра")
    g.add_argument("--image", help="картинка для первого кадра")
    ap.add_argument("--csv", help="CSV с t,yaw_norm,acc_norm")
    ap.add_argument("--out",   default="minute.mp4")
    ap.add_argument("--fps",   type=int, default=15)
    ap.add_argument("--seconds", type=int, default=60)
    ap.add_argument("--size",  default="256x144", help="WxH, должна совпадать с трейном")
    # антидрейф параметры
    ap.add_argument("--state_decay", type=float, default=0.98, help="затухание скрытого состояния GRU [0..1]")
    ap.add_argument("--anchor_strength", type=float, default=1.0, help="сила якоря яркости (1.0 норм)")
    ap.add_argument("--blend_prev", type=float, default=0.05, help="примесь прошлого кадра [0..1]")
    args = ap.parse_args()

    W,H = parse_size(args.size)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # модель
    ckpt = torch.load(args.ckpt, map_location=dev)
    model = FramePredictor(img_ch=3, base=96, joy_dim=2).to(dev)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    # стартовый кадр
    if args.video:
        cur = first_frame_from_video(args.video, (W,H)).to(dev)
    else:
        cur = first_frame_from_image(args.image, (W,H)).to(dev)

    # джойстик
    csv_triplet = read_csv(args.csv) if args.csv else None
    joys = build_joystick(csv_triplet, args.fps, args.seconds).to(dev)  # [T,2]

    # фотометрический якорь по первому кадру
    with torch.no_grad():
        m0 = cur.mean(dim=(2,3), keepdim=True)  # [1,3,1,1]

    # видеорайтер
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(args.out, fourcc, args.fps, (W, H))

    # генерация
    with torch.no_grad():
        h = None
        T = joys.size(0)
        eps = 1e-6
        for i in range(T):
            joy = joys[i].unsqueeze(0).float()
            nxt, h = model(cur, joy, h)

            # антидрейф: 1) якорь яркости к m0, 2) затухание состояния, 3) лёгкий бленд с прошлым
            if args.anchor_strength > 0:
                mn = nxt.mean(dim=(2,3), keepdim=True)
                scale = (m0 / (mn + eps)).clamp(0.5, 2.0) ** args.anchor_strength
                nxt = (nxt * scale).clamp(0.0, 1.0)

            if h is not None and 0 < args.state_decay < 1:
                h = h * args.state_decay

            if args.blend_prev > 0:
                alpha = float(np.clip(args.blend_prev, 0.0, 1.0))
                nxt = (1 - alpha) * nxt + alpha * cur

            nxt = nxt.clamp(0.01, 0.99)  # чтобы не налипало на идеальные 0/1

            img = (nxt[0] * 255).permute(1,2,0).byte().cpu().numpy()
            vw.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            cur = nxt

    vw.release()
    print(f"[✓] Сохранено: {args.out} | {args.seconds}s @ {args.fps}fps, size {W}x{H}")

if __name__ == "__main__":
    main()
