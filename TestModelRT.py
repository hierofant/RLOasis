
import cv2, numpy as np, torch, argparse
from model_unified import FramePredictor

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--size", default="256x144")
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()
    W,H = map(int, args.size.split("x"))
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.ckpt, map_location=dev)
    model = FramePredictor(img_ch=3, base=64, joy_dim=2, in_size=(H,W)).to(dev)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    # start from gray frame
    cur = torch.full((1,3,H,W), 0.5, device=dev)
    h = None
    yaw = 0.0
    acc = 0.0

    win = "RT predictor (arrows: left/right for yaw, up/down for acc, space to reset)"
    cv2.namedWindow(win)

    dt = 1.0/args.fps
    while True:
        # keyboard input
        k = cv2.waitKey(int(1000*dt)) & 0xFF
        if k == 27:  # ESC
            break
        elif k == ord(' '):
            yaw = 0.0; acc = 0.0
        elif k == 81 or k == ord('a'):  # left
            yaw -= 0.05
        elif k == 83 or k == ord('d'):  # right
            yaw += 0.05
        elif k == 82 or k == ord('w'):  # up
            acc += 0.05
        elif k == 84 or k == ord('s'):  # down
            acc -= 0.05
        yaw = float(np.clip(yaw, -1, 1)); acc = float(np.clip(acc, -1, 1))

        joy = torch.tensor([[yaw, acc]], device=dev).float()
        with torch.no_grad():
            nxt, h = model(cur, joy, h)
        img = (nxt[0].clamp(0,1)*255).permute(1,2,0).cpu().numpy().astype(np.uint8)
        cv2.imshow(win, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cur = nxt

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
