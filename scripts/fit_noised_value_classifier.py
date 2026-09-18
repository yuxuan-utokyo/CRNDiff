# -*- coding: utf-8 -*-
"""Fit the noised-state classifier that scores candidates for value-guided resampling.

The training data is noised ON THE FLY and never materialised, since doing so would be tens of
gigabytes. Each batch draws a grid step at random and noises the training cells to that step
with the FROZEN forward kernel; the labels are the eleven cell types. The kernel is
n_t = Binomial(x0, e^-t) + Poisson(V (1 - e^-t)), clipped, word for word as in the frozen
trainer, and its constants come from the frozen module rather than being restated here.

    python scripts/fit_noised_value_classifier.py --help
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
try:
    import torch
except ModuleNotFoundError as _exc:                                      # noqa: E402
    raise SystemExit(
        "this entry point needs PyTorch, which the offline reproduction does not "
        "install. See README section 7; in short: pip install torch") from _exc
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from crndiff import config as C                                          # noqa: E402
from crndiff.io import sha256_file                                       # noqa: E402

# Data-dependent imports are deferred until after CLI parsing (--help needs no atlas).

FIT_SEED = 20260817
NOISE_SEED = 20260818
EMB_DIM = 64
HIDDEN = 512
EPOCHS = 10
BATCH = 1024
LR = 1e-3


class NoisedClassifier(nn.Module):
    def __init__(self, dim: int, n_steps: int, n_labels: int,
                 emb_dim: int = EMB_DIM, hidden: int = HIDDEN):
        super().__init__()
        self.dim, self.n_steps, self.n_labels = dim, n_steps, n_labels
        self.step_emb = nn.Embedding(n_steps, emb_dim)
        self.net = nn.Sequential(
            nn.Linear(dim + emb_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_labels))

    def forward(self, n_t: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        x = torch.cat([torch.log1p(n_t), self.step_emb(k)], dim=1)
        return self.net(x)

    @torch.no_grad()
    def probability(self, n_t: torch.Tensor, k: int) -> torch.Tensor:
        kk = torch.full((n_t.shape[0],), int(k), dtype=torch.long, device=n_t.device)
        return F.softmax(self.forward(n_t.to(torch.float32), kk), dim=1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--out-dir", default=str(C.MODELS / "noised_clf"))
    a = ap.parse_args()
    for name in ("meta.json", "train.npy", "val.npy"):
        C.require(C.LEGACY_DATA / "all" / name, "atlas input; see README section 6")

    C.add_base_code_to_path()
    import config_hvg2k as CFG
    import model_hvg2k as M

    outd = Path(a.out_dir)
    ck, js = outd / "noised_clf.pt", outd / "log.json"
    if ck.exists() or js.exists():
        raise SystemExit(f"refusing to overwrite {ck}")
    outd.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    d, meta = M.load_data(("train", "val"))
    tr, va = d["train"], d["val"]
    der = M.derive(tr)
    T = M.horizon()
    gf, nfe = M.build_grid(a.K, T)
    tgrid = np.asarray(gf, dtype=np.float64)
    # there are as many grid points as the time grid has entries, the last being the clean
    # state. The state after a given step lies on the next grid point, and the last step lands
    # exactly on the clean state, so the embedding table must cover EVERY grid point rather
    n_steps = len(tgrid)
    ytr_raw = np.load(C.LEGACY_DATA / "all" / "train_celltype.npy", allow_pickle=True).astype(str)
    yva_raw = np.load(C.LEGACY_DATA / "all" / "val_celltype.npy", allow_pickle=True).astype(str)
    vocab = sorted(set(ytr_raw.tolist()))
    lut = {v: i for i, v in enumerate(vocab)}
    ytr = torch.tensor([lut[v] for v in ytr_raw], dtype=torch.long)
    yva = torch.tensor([lut[v] for v in yva_raw], dtype=torch.long)
    dev = M.DEV
    print(f"[data] train {tr.shape} val {va.shape} | K={a.K} n_steps={n_steps} "
          f"| labels {len(vocab)}", flush=True)

    cap = CFG.NMAX if CFG.CLIP_X0_TO_NMAX else None
    Xtr = torch.tensor(np.clip(tr, 0, cap) if cap else tr, dtype=torch.float32)
    Xva = torch.tensor(np.clip(va, 0, cap) if cap else va, dtype=torch.float32)
    Vt = torch.tensor(np.asarray(der["V"], np.float64), dtype=torch.float32, device=dev)

    torch.manual_seed(FIT_SEED)
    np.random.seed(FIT_SEED)
    g_fit = torch.Generator(device="cpu").manual_seed(FIT_SEED)       # batch sampling
    g_noise = torch.Generator(device="cpu").manual_seed(NOISE_SEED)   # the noising stream, kept separate from the fitting stream

    model = NoisedClassifier(tr.shape[1], n_steps, len(vocab)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    def corrupt(x0: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        t = torch.as_tensor(tgrid[k.cpu().numpy()], dtype=torch.float32, device=dev)
        r = torch.exp(-t).unsqueeze(1)
        return (torch.binomial(x0, r.expand_as(x0))
                + torch.poisson(Vt.unsqueeze(0) * (1 - r))).clamp(0, CFG.NMAX)

    n = Xtr.shape[0]
    steps_per_epoch = n // a.batch
    curve = []
    for ep in range(a.epochs):
        model.train()
        perm = torch.randperm(n, generator=g_fit)
        tot, cor, ls = 0, 0, 0.0
        for s in range(steps_per_epoch):
            idx = perm[s * a.batch:(s + 1) * a.batch]
            x0 = Xtr[idx].to(dev)
            y = ytr[idx].to(dev)
            k = torch.randint(0, n_steps, (len(idx),), generator=g_noise).to(dev)
            with torch.no_grad():
                n_t = corrupt(x0, k)
            logits = model(n_t, k)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            ls += float(loss) * len(idx)
            cor += int((logits.argmax(1) == y).sum())
            tot += len(idx)
        curve.append({"epoch": ep + 1, "train_loss": ls / tot, "train_acc": cor / tot,
                      "walltime_s": time.time() - t0})
        print(f"  epoch {ep+1}/{a.epochs} loss={ls/tot:.4f} acc={cor/tot:.4f} "
              f"[{(time.time()-t0)/60:.1f} min]", flush=True)

    # validation accuracy per grid step, on a fixed noising stream separate from training
    model.eval()
    g_eval = torch.Generator(device="cpu").manual_seed(NOISE_SEED + 1)
    torch.manual_seed(NOISE_SEED + 1)
    per_k = {}
    nv = min(20000, Xva.shape[0])
    vi = torch.randperm(Xva.shape[0], generator=g_eval)[:nv]
    prior = np.bincount(ytr.numpy(), minlength=len(vocab)) / len(ytr)
    with torch.no_grad():
        for k in range(n_steps):
            cor = 0
            for lo in range(0, nv, 2048):
                idx = vi[lo:lo + 2048]
                x0 = Xva[idx].to(dev)
                kk = torch.full((len(idx),), k, dtype=torch.long, device=dev)
                n_t = corrupt(x0, kk)
                cor += int((model(n_t, kk).argmax(1) == yva[idx].to(dev)).sum())
            per_k[str(k)] = {"t": float(tgrid[k]), "val_acc": cor / nv}
            print(f"  k={k:2d} t={tgrid[k]:8.4f} val_acc={cor/nv:.4f}", flush=True)

    torch.save(model.state_dict(), ck)
    blob = {"ticket": 200, "block": "M1-7a", "what": "noised-state classifier for VGR-noised",
            "architecture": f"[log1p(n_t) ({tr.shape[1]}), emb(k) ({EMB_DIM})] -> {HIDDEN} "
                            f"-> {HIDDEN} -> {len(vocab)}",
            "emb_dim_note": "the embedding width was not pre-specified; it was fixed before the run and not swept",
            "optimizer": "Adam", "lr": LR, "epochs": a.epochs, "batch": a.batch,
            "fit_seed": FIT_SEED, "noise_stream_seed": NOISE_SEED,
            "forward_kernel": "Binom(x0, exp(-t)) + Pois(V*(1-exp(-t))), clipped to NMAX -- "
                              "verbatim model_hvg2k.train_ours_hvg2k::noise",
            "V_source": "model_hvg2k.derive(train)['V'] (frozen code via add_base_code_to_path)",
            "dim": int(tr.shape[1]), "hidden": HIDDEN, "emb_dim": EMB_DIM,
            "K": a.K, "n_steps": n_steps, "tgrid": tgrid.tolist(),
            "label_vocabulary": vocab, "label_prior_train": prior.tolist(),
            "n_train": int(n), "n_val_eval": int(nv),
            "train_curve": curve, "val_accuracy_per_k": per_k,
            "n_params": int(sum(p.numel() for p in model.parameters())),
            "ckpt": str(ck), "ckpt_sha256": None,
            "walltime_s": time.time() - t0,
            "environment": {"python": sys.version, "numpy": np.__version__,
                            "torch": torch.__version__, "cuda": torch.version.cuda,
                            "platform": platform.platform()},
            "written_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    blob["ckpt_sha256"] = sha256_file(ck)
    js.write_text(json.dumps(blob, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"[out] {ck}\n[out] {js}  ({(time.time()-t0)/60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
