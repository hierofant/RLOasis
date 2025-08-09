"""
prepare_shards_mm.py  (verbose)
Создаёт memory-mapped шарды-папки для быстрого обучения без раздувания RAM.

Структура:
  C:/cache_shards_mm/shard_0000/
      prevs.npy  [N,T,3,H,W] uint8|fp16|fp32
      nexts.npy  [N,T,3,H,W] uint8|fp16|fp32
      joys.npy   [N,T,2] float32
      meta.json  {"img_size":[W,H], "clip_len":T, "dtype":"uint8"}

Логи:
- Для каждого видео показывает прогресс по кадрам и по клипам с ETA.
- Лог на каждый флаш шарда.
"""

import os, cv2, json, time, math, numpy as np, pandas as pd
from glob import glob

# ------- CONFIG -------
VIDEO_DIR     = "./your_dataset"
SHARDS_DIR    = "C:/cache_shards_mm"   # положи на SSD
IMG_SIZE      = (256, 144)             # (W,H)
TARGET_FPS    = 15
CLIP_LEN      = 12
CLIP_HOP      = 6
SHARD_CAP     = 256                    # клипов в одном шарде
DEFAULT_DTYPE = "uint8"                # "uint8" | "fp16" | "fp32"
LOG_EVERY_FR  = 200                    # как часто логать по кадрам
LOG_EVERY_CL  = 200                    # как часто логать по клипам

os.makedirs(SHARDS_DIR, exist_ok=True)

def read_csv_align(csv_path: str):
    df = pd.read_csv(csv_path)
    cols = {c.lower(): c for c in df.columns}
    tcol  = cols.get('t') or cols.get('time') or cols.get('timestamp')
    yawc  = cols.get('yaw_norm') or cols.get('yaw')
    accc  = cols.get('acc_norm') or cols.get('acc')
    assert tcol and yawc and accc, f"CSV must have t,yaw_norm,acc_norm: {csv_path}"
    t   = df[tcol].to_numpy(dtype=np.float32)
    yaw = df[yawc].to_numpy(dtype=np.float32)
    acc = df[accc].to_numpy(dtype=np.float32)
    return t, yaw, acc

def resize_to_tensor(img_np, size):
    import torch
    img = torch.from_numpy(img_np).permute(2,0,1).float()/255.0
    img = torch.nn.functional.interpolate(img[None], size=(size[1], size[0]), mode="bilinear", align_corners=False)[0]
    return img.numpy()  # [3,H,W] float32 0..1

def quantize(x, mode):
    if mode == "uint8":
        return (np.clip(x,0,1)*255.0 + 0.5).astype(np.uint8)
    if mode == "fp16":
        return x.astype(np.float16)
    return x.astype(np.float32)

def fmt_time(sec):
    if sec < 1: return f"{sec*1000:.0f} ms"
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    if h: return f"{h}h {m}m {s}s"
    if m: return f"{m}m {s}s"
    return f"{s}s"

def flush_shard(si, buf_prevs, buf_nexts, buf_joys, dtype):
    if not buf_prevs: 
        return 0
    N = len(buf_prevs)
    outdir = os.path.join(SHARDS_DIR, f"shard_{si:04d}")
    os.makedirs(outdir, exist_ok=True)

    t0 = time.perf_counter()
    prevs = quantize(np.stack(buf_prevs, axis=0), dtype)  # [N,T,3,H,W]
    nexts = quantize(np.stack(buf_nexts, axis=0), dtype)
    joys  = np.stack(buf_joys, axis=0).astype(np.float32) # [N,T,2]
    t1 = time.perf_counter()

    np.save(os.path.join(outdir, "prevs.npy"), prevs)
    np.save(os.path.join(outdir, "nexts.npy"), nexts)
    np.save(os.path.join(outdir, "joys.npy"),  joys)
    with open(os.path.join(outdir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({"img_size": IMG_SIZE, "clip_len": CLIP_LEN, "dtype": dtype}, f)

    buf_prevs.clear(); buf_nexts.clear(); buf_joys.clear()
    t2 = time.perf_counter()
    print(f"[#] Wrote {outdir}  clips={N} dtype={dtype} | stack {fmt_time(t1-t0)} | save {fmt_time(t2-t1)}", flush=True)
    return N

def process_video(vpath, si_start, dtype):
    base = os.path.splitext(vpath)[0]
    csv  = base + ".csv"
    if not os.path.exists(csv):
        print(f"[!] Skip (no CSV): {vpath}")
        return si_start, 0
    cap = cv2.VideoCapture(vpath)
    if not cap.isOpened():
        print(f"[!] Cannot open: {vpath}")
        return si_start, 0
    nf  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or TARGET_FPS

    duration = nf / max(1.0, fps)
    t_video  = np.arange(0, duration, 1.0/TARGET_FPS, dtype=np.float32)
    t_csv, yaw, acc = read_csv_align(csv)
    yaw_i = np.interp(t_video, t_csv, yaw).astype(np.float32)
    acc_i = np.interp(t_video, t_csv, acc).astype(np.float32)

    print(f"\n[+] {os.path.basename(vpath)} | frames={nf} @ {fps:.2f}fps -> target {len(t_video)} | size={IMG_SIZE} | dtype={dtype}")
    t_video_start = time.perf_counter()

    frames = []
    t_count = min(len(t_video), nf)
    # чтение кадров с логом прогресса
    for i in range(t_count):
        ok, frame = cap.read()
        if not ok: 
            print(f"[!] Early EOF at frame {i}/{t_count}")
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(resize_to_tensor(frame, IMG_SIZE))

        if (i+1) % LOG_EVERY_FR == 0 or i+1 == t_count:
            elapsed = time.perf_counter() - t_video_start
            rate = (i+1) / max(elapsed, 1e-6)
            eta = (t_count - (i+1)) / max(rate, 1e-6)
            pct = (i+1) * 100.0 / t_count
            print(f"[{os.path.basename(vpath)}] Frame {i+1}/{t_count} ({pct:.1f}%) | {rate:.1f} fps | ETA {fmt_time(eta)}", flush=True)

    cap.release()
    if not frames:
        return si_start, 0
    frames = np.stack(frames, axis=0)  # [F,3,H,W]
    yaw_i  = yaw_i[:len(frames)]
    acc_i  = acc_i[:len(frames)]

    # нарезка на клипы
    buf_prevs, buf_nexts, buf_joys = [], [], []
    si = si_start
    made = 0
    slice_start = time.perf_counter()
    total_clips = max(0, (len(frames) - (CLIP_LEN+1) + 1 + (CLIP_HOP-1)) // CLIP_HOP)

    for s in range(0, len(frames) - (CLIP_LEN+1) + 1, CLIP_HOP):
        prevs = frames[s:s+CLIP_LEN]
        nexts = frames[s+1:s+CLIP_LEN+1]
        joys  = np.stack([yaw_i[s:s+CLIP_LEN], acc_i[s:s+CLIP_LEN]], axis=1)
        buf_prevs.append(prevs)
        buf_nexts.append(nexts)
        buf_joys.append(joys)
        made += 1

        if made % LOG_EVERY_CL == 0:
            elapsed = time.perf_counter() - slice_start
            rate = made / max(elapsed, 1e-6)
            eta = (total_clips - made) / max(rate, 1e-6) if total_clips else 0
            print(f"[{os.path.basename(vpath)}] Clip {made}/{total_clips} | {rate:.1f} clips/s | ETA {fmt_time(eta)}", flush=True)

        if len(buf_prevs) >= SHARD_CAP:
            flush_shard(si, buf_prevs, buf_nexts, buf_joys, dtype); si += 1

    # финальный флаш для этого видео
    if buf_prevs:
        flush_shard(si, buf_prevs, buf_nexts, buf_joys, dtype); si += 1

    total = time.perf_counter() - t_video_start
    print(f"[=] {os.path.basename(vpath)} done | clips={made} | time {fmt_time(total)} | avg {(len(frames)/max(total,1e-6)):.1f} fps", flush=True)
    return si, made

def main():
    vids = sorted(glob(os.path.join(VIDEO_DIR, "*.mp4")) + glob(os.path.join(VIDEO_DIR, "*.avi")))
    if not vids:
        print(f"No videos in {VIDEO_DIR}")
        return
    si = 0
    total_clips = 0
    wall_start = time.perf_counter()
    print(f"[*] Found {len(vids)} videos. Output dir: {SHARDS_DIR}")

    for k, v in enumerate(vids, 1):
        print(f"[{k}/{len(vids)}] Processing {v}")
        si, made = process_video(v, si, DEFAULT_DTYPE)
        total_clips += made

    with open(os.path.join(SHARDS_DIR, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"img_size": IMG_SIZE, "target_fps": TARGET_FPS, "clip_len": CLIP_LEN, "clip_hop": CLIP_HOP,
                   "dtype": DEFAULT_DTYPE, "num_shards": si, "num_clips": total_clips}, f, ensure_ascii=False, indent=2)

    wall = time.perf_counter() - wall_start
    print(f"[✓] Done. Shards: {si}, Clips: {total_clips}, dtype={DEFAULT_DTYPE} | total {fmt_time(wall)}", flush=True)

if __name__ == "__main__":
    main()