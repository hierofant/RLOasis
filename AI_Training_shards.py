"""
AI_Training_shards_mm.py
Обучение из шардов-папок (prevs.npy/nexts.npy/joys.npy), созданных prepare_shards_mm.py.
Читает через mmap, не жрет RAM. Работает в [0,1], без Normalize.
"""

import os, json, time, numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from model_unified import FramePredictor, ssim

# ---- CONFIG ----
SHARDS_DIR   = "C:/cache_shards_mm"   # где лежат shard_xxxx/
OUT_DIR      = "./checkpoints_shards" # куда класть веса
BATCH_SIZE   = 12
EPOCHS       = 10
LR           = 2e-4
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS  = 0                      # на SSD можешь 2-4 попробовать
PREFETCH     = None                   # например 2, если включишь воркеры

os.makedirs(OUT_DIR, exist_ok=True)

def temporal_l1(pred_seq, tgt_seq):
    dp = pred_seq[:,1:] - pred_seq[:,:-1]
    dt = tgt_seq[:,1:]  - tgt_seq[:,:-1]
    return F.l1_loss(dp, dt)

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {n: p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
    def update(self, model):
        for n,p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1-self.decay)
    def copy_to(self, model):
        for n,p in model.named_parameters():
            if p.requires_grad:
                p.data.copy_(self.shadow[n])

class ShardMMDataset(Dataset):
    def __init__(self, shards_dir):
        with open(os.path.join(shards_dir, "manifest.json"), "r", encoding="utf-8") as f:
            mani = json.load(f)
        self.img_size  = tuple(mani["img_size"])
        self.clip_len  = mani["clip_len"]
        self.dtype     = mani.get("dtype", "uint8")

        self.shards = []
        for name in sorted(os.listdir(shards_dir)):
            p = os.path.join(shards_dir, name)
            if os.path.isdir(p) and os.path.exists(os.path.join(p, "prevs.npy")):
                self.shards.append(p)
        assert self.shards, "No shard folders found"

        self.index = []
        self.maps = []
        for si, sdir in enumerate(self.shards):
            prevs = np.load(os.path.join(sdir, "prevs.npy"), mmap_mode="r")
            nexts = np.load(os.path.join(sdir, "nexts.npy"), mmap_mode="r")
            joys  = np.load(os.path.join(sdir, "joys.npy"),  mmap_mode="r")
            assert prevs.shape[0] == nexts.shape[0] == joys.shape[0]
            N = prevs.shape[0]
            self.maps.append((prevs, nexts, joys))
            self.index.extend([(si, ci) for ci in range(N)])

    def __len__(self): return len(self.index)

    def __getitem__(self, i):
        si, ci = self.index[i]
        prevs, nexts, joys = self.maps[si]
        prevs = torch.from_numpy(prevs[ci])      # [T,3,H,W]  uint8|fp16|fp32
        nexts = torch.from_numpy(nexts[ci])
        joys  = torch.from_numpy(joys[ci]).float() # [T,2]
        if prevs.dtype == torch.uint8:
            prevs = prevs.float().div_(255.0)
            nexts = nexts.float().div_(255.0)
        else:
            prevs = prevs.float(); nexts = nexts.float()
        return prevs, joys, nexts

def train():
    ds = ShardMMDataset(SHARDS_DIR)
    dl = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=PREFETCH if (PREFETCH and NUM_WORKERS>0) else None,
        persistent_workers=(NUM_WORKERS>0),
    )

    H, W = ds.img_size[1], ds.img_size[0]
    model = FramePredictor(img_ch=3, base=96, joy_dim=2).to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9,0.999), weight_decay=1e-4)
    scaler= torch.amp.GradScaler('cuda', enabled=(DEVICE=='cuda'))
    ema   = EMA(model)

    global_step = 0
    for epoch in range(1, EPOCHS+1):
        model.train()
        t0 = time.time()
        losses = []
        for prevs, joys, nexts in dl:
            B,T,C,H,W = prevs.shape
            prevs = prevs.to(DEVICE, non_blocking=True)
            nexts = nexts.to(DEVICE, non_blocking=True)
            joys  = joys.to(DEVICE, non_blocking=True)

            p_sched = min(0.5, 0.05 * max(0, epoch-2))  # scheduled sampling

            preds = []
            h = None
            cur = prevs[:,0]
            with torch.amp.autocast('cuda', enabled=(DEVICE=='cuda')):
                for t in range(T):
                    pred, h = model(cur, joys[:,t], h)
                    preds.append(pred)
                    use_pred = (torch.rand(B, device=DEVICE) < p_sched).float().view(B,1,1,1)
                    cur = use_pred * pred.detach() + (1 - use_pred) * nexts[:, t]

                pred_seq = torch.stack(preds, dim=1)  # [B,T,3,H,W]
                l1 = F.l1_loss(pred_seq, nexts)
                s  = 1.0 - ssim(pred_seq.reshape(-1,3,H,W), nexts.reshape(-1,3,H,W))
                tl = temporal_l1(pred_seq, nexts)
                loss = 0.7*l1 + 0.27*s + 0.03*tl

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            ema.update(model)
            losses.append(loss.item())
            global_step += 1

        ema.copy_to(model)
        ckpt = os.path.join(OUT_DIR, f"fp_mm_epoch{epoch:03d}.pt")
        torch.save({"model": model.state_dict(), "epoch": epoch, "img_size": (W,H)}, ckpt)
        print(f"Epoch {epoch}/{EPOCHS} | loss {np.mean(losses):.4f} | {time.time()-t0:.1f}s | saved {ckpt}")

if __name__ == "__main__":
    train()
