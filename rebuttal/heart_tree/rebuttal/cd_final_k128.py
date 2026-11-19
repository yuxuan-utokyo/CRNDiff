# -*- coding: utf-8 -*-
"""CountsDiff HVG2K K128 final arrays. Copy of cd_final_seed.py (3b2f1bc9) with
the picks set to K128 and the K128 sample folders: `python -m rebuttal.cd_final_k128 <train seed>`.
  s20260625: Endothelial = out/R12/hvg2k_samples/Endothelial_K128_g1p0 (2,500 @ 20261061, batch 256, selection phase) + out/R12/hvg2k_k128_s20260625/
             Endothelial_K128_g1p0_topup (7,557 @ 20261062, batch 128, 4090 11-10); Myeloid/Neuronal = hvg2k_samples/<t>_K128_g1p0 (2,500, batch 256).
  s20260626 (4090) / s20260627 (the GPU server): out/R12/hvg2k_k128_s<seed>/ (all batch 128).
Endothelial = 2,500 + top-up 7,557 concatenated in that order (10,057); same checkpoint, K, guidance asserted. Staged as
out/runs/rebuttal_CD_K128_s<seed>/<arm>/<name>.{npy,json} for the frozen scorer; copied to _data_exchange/runs/countsdiff_hvg2k_k128_s<seed>/.
"""
import hashlib, json, shutil, sys
from pathlib import Path
import numpy as np
TS = sys.argv[1]
R = Path(__file__).resolve().parents[1]; S = R / "rebuttal/out/R12"; runs = R / f"out/runs/rebuttal_CD_K128_s{TS}"
X = Path(r"work\exchange\runs\countsdiff_hvg2k_k128_s" + TS)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
base = S / ("hvg2k_samples" if TS == "20260625" else f"hvg2k_k128_s{TS}"); top = S / f"hvg2k_k128_s{TS}"
X.mkdir(parents=True, exist_ok=True); assert not runs.exists(), f"{runs} exists; refusing to overwrite"
for t in ("Endothelial", "Myeloid", "Neuronal"):
    arm = f"{t}_K128_g1p0"; a = base / f"{arm}.npy"; ma = json.loads(a.with_suffix(".json").read_text(encoding="utf-8"))
    assert ma["K"] == 128 and float(ma["guidance"]) == 1.0 and ma["type"] == t and str(TS) in ma["ckpt"], (a, ma)
    parts = [{"file": str(a), "sha256": sha(a), "seed": ma["seed"], "n": ma["n"]}]
    if t == "Endothelial":
        b = top / f"{arm}_topup.npy"; mb = json.loads(b.with_suffix(".json").read_text(encoding="utf-8"))
        same = lambda p: Path(str(p).replace("\\", "/")).name
        assert (mb["K"], mb["guidance"], same(mb["ckpt"]), mb["type"]) == (ma["K"], ma["guidance"], same(ma["ckpt"]), ma["type"]), "top-up settings differ"
        x = np.concatenate([np.load(a), np.load(b)]); parts.append({"file": str(b), "sha256": sha(b), "seed": mb["seed"], "n": mb["n"]})
        assert x.shape[0] == 10057, x.shape
        seeds = f"{ma['seed']}+{mb['seed']}"
    else:
        x = np.load(a); seeds = str(ma["seed"]); assert x.shape[0] == 2500, x.shape
    full = f"{arm}_n{x.shape[0]}"; name = f"CountsDiff_{full}_train{TS}_samp{seeds.replace('+', '_')}"
    d = runs / full; d.mkdir(parents=True, exist_ok=True); npy = d / f"{name}.npy"; np.save(npy, x)
    m = {k: ma[k] for k in ("ckpt", "ckpt_step", "type", "K", "nfe", "guidance", "eta")}
    m.update({"request": t, "n_delivered": int(x.shape[0]), "arm": full, "seed": seeds, "parts_in_order": parts,
              "generator": "CountsDiff (r12 v2, authors' recipe)", "selected_by": "the author 18:35 (Q): K128 g1.0 for every target",
              "npy_sha256": sha(npy)})
    (d / f"{name}.json").write_text(json.dumps(m, indent=1), encoding="utf-8")
    tx = X / full / f"{name}.npy"; tx.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(npy, str(tx) + ".part"); Path(str(tx) + ".part").replace(tx); shutil.copyfile(d / f"{name}.json", tx.with_suffix(".json"))
    assert sha(tx) == m["npy_sha256"]
    with open(X / "SHA256SUMS", "a", encoding="utf-8", newline="\n") as f: f.write(f"{m['npy_sha256']}  {full}/{name}.npy\n")
    print(f"{full}: n={x.shape[0]} seeds {seeds} sha {m['npy_sha256'][:16]}", flush=True)
