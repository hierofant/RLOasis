# model_unified.py
# Единая модель для трейна и инференса.
# Работает в [0,1]. Предсказывает Δ (дельту) к прошлому кадру и добавляет её: out = clamp(prev + α * tanh(Δ)).
# Есть U-Net скипы и FiLM-кондиционирование по джойстику.
# Рекомендовано подавать джойстик как [yaw, acc, dyaw, dacc] (joy_dim=4).

import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------- Utils --------------------

def film(x, g, b):
    # x: [B,C,H,W], g/b: [B,C,1,1]
    return x * (1 + g) + b

def ssim(x, y, C1=0.01**2, C2=0.03**2):
    """
    x, y in [0,1], shape [B,3,H,W] or [B,1,H,W]
    Возвращает средний SSIM по батчу.
    """
    if x.size(1) == 1:
        x = x.repeat(1, 3, 1, 1)
        y = y.repeat(1, 3, 1, 1)
    mu_x = F.avg_pool2d(x, 3, 1, 1)
    mu_y = F.avg_pool2d(y, 3, 1, 1)
    sigma_x  = F.avg_pool2d(x*x, 3, 1, 1) - mu_x*mu_x
    sigma_y  = F.avg_pool2d(y*y, 3, 1, 1) - mu_y*mu_y
    sigma_xy = F.avg_pool2d(x*y, 3, 1, 1) - mu_x*mu_y
    num = (2*mu_x*mu_y + C1) * (2*sigma_xy + C2)
    den = (mu_x*mu_x + mu_y*mu_y + C1) * (sigma_x + sigma_y + C2)
    return (num / (den + 1e-12)).mean()

# -------------------- Blocks --------------------

class ConvGRUCell(nn.Module):
    def __init__(self, ch, ks=3):
        super().__init__()
        p = ks // 2
        self.zr = nn.Conv2d(ch*2, ch*2, ks, 1, p)
        self.hn = nn.Conv2d(ch*2, ch,   ks, 1, p)
    def forward(self, x, h=None):
        if h is None:
            h = torch.zeros_like(x)
        zr = torch.sigmoid(self.zr(torch.cat([x, h], dim=1)))
        z, r = torch.chunk(zr, 2, dim=1)
        n = torch.tanh(self.hn(torch.cat([x, r*h], dim=1)))
        return (1 - z) * h + z * n

class JoyCond(nn.Module):
    def __init__(self, ch, joy_dim=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(joy_dim, 128), nn.ReLU(True),
            nn.Linear(128, ch*2)
        )
        self.ch = ch
    def forward(self, joy):
        # joy: [B,joy_dim]
        g, b = torch.chunk(torch.tanh(self.fc(joy)), 2, dim=1)  # [-1,1]
        return g[..., None, None], b[..., None, None]

# -------------------- Model --------------------

class FramePredictor(nn.Module):
    """
    Residual UNet + ConvGRU + FiLM.
    Вход:  prev  [B,3,H,W] в [0,1]; joy [B,joy_dim] обычно [yaw, acc, dyaw, dacc]
    Выход: next  [B,3,H,W] в [0,1]
    Параметры:
      base: ширина каналов (96 ок для 256x144/384x216)
      alpha: множитель для дельты (шаг резидуала)
    """
    def __init__(self, img_ch=3, base=96, joy_dim=4, alpha=0.25):
        super().__init__()
        self.alpha = alpha
        # Encoder
        self.e1 = nn.Sequential(                     # H,W -> H/2,W/2
            nn.Conv2d(img_ch, base, 4, 2, 1), nn.ReLU(inplace=True)
        )
        self.e2 = nn.Sequential(                     # -> H/4,W/4
            nn.Conv2d(base, base*2, 4, 2, 1), nn.ReLU(inplace=True)
        )
        self.e3 = nn.Sequential(                     # -> H/8,W/8
            nn.Conv2d(base*2, base*4, 4, 2, 1), nn.ReLU(inplace=True)
        )
        self.lat_ch = base*4

        # Temporal core + conditioning
        self.gru  = ConvGRUCell(self.lat_ch)
        self.cond = JoyCond(self.lat_ch, joy_dim)

        # Decoder with skips
        self.d3 = nn.Sequential(                     # H/8 -> H/4
            nn.ConvTranspose2d(self.lat_ch, base*2, 4, 2, 1), nn.ReLU(inplace=True)
        )
        self.d2 = nn.Sequential(                     # concat e2 -> H/2
            nn.ConvTranspose2d(base*2*2, base, 4, 2, 1), nn.ReLU(inplace=True)
        )
        self.d1 = nn.ConvTranspose2d(base*2, 3, 4, 2, 1)  # concat e1 -> H

    def forward(self, prev, joy, h=None):
        # prev: [B,3,H,W] in [0,1]
        e1 = self.e1(prev)            # [B,base,H/2,W/2]
        e2 = self.e2(e1)              # [B,2b,H/4,W/4]
        z  = self.e3(e2)              # [B,4b,H/8,W/8]

        g, b = self.cond(joy)         # [B,4b,1,1] x2
        z = film(z, g, b)
        h = self.gru(z, h)            # [B,4b,H/8,W/8]

        x = self.d3(film(h, g, b))    # [B,2b,H/4,W/4]
        x = torch.cat([x, e2], dim=1) # [B,4b,H/4,W/4]
        x = self.d2(x)                # [B,b,H/2,W/2]
        x = torch.cat([x, e1], dim=1) # [B,2b,H/2,W/2]
        delta = torch.tanh(self.d1(x))  # [-1,1] -> Δ
        out = (prev + self.alpha * delta).clamp(0.0, 1.0)
        return out, h
