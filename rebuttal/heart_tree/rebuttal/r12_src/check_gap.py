"""Is the train-val gap a real gap, or EMA lag? Score raw and EMA weights on the same fixed train and val cells."""
import sys, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import countsdiff2k as M
ck = torch.load(sys.argv[2], map_location="cpu", weights_only=False); c = ck["conf"]; d = Path(sys.argv[1])
dev = M.device(); order = c["labels"]
Xt, yt = M.load_split(d, "train", order, c["nmax"]); Xv, yv = M.load_split(d, "val", order, c["nmax"])
for name in ("model", "ema"):
    m = M.CountsDiffAttnCond(len(c["iscale"]), np.array(c["iscale"]), len(order), c["nmax"]); m.load_state_dict(ck[name]); m.to(dev).eval()
    print(name, "step", ck["step"], "train", round(M.val_loss(m, Xt, yt, dev, 256), 2), "val", round(M.val_loss(m, Xv, yv, dev, 256), 2), flush=True)
