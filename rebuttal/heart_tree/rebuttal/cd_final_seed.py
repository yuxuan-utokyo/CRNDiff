# -*- coding: utf-8 -*-
"""(11-09: per-train-seed copy of cd_final.py; `python -m rebuttal.cd_final_seed <train seed>`, samples in out/R12/hvg2k_samples_s<seed>/,
staged as out/runs/rebuttal_CD_final_s<seed>/, exchange runs/countsdiff_hvg2k_final_s<seed>/; s20260626 sampled at batch 128 on the 4090.)
CountsDiff HVG2K final arrays: the Mac's picks Endothelial K256, Myeloid K32, Neuronal K256, all guidance 1.0.
Endothelial = the selection-phase 2,500 (seed 20261061) + the top-up 7,557 (seed 20261062, same checkpoint, K, guidance), concatenated in
that order to 10,057; Myeloid and Neuronal = the selection-phase 2,500 as they are. Staged as out/runs/rebuttal_CD_final/<arm>/<name>.{npy,json}
for the frozen scorer and copied to _data_exchange/runs/countsdiff_hvg2k_final/<arm>/<file>.npy with SHA256SUMS (2 columns, LF, no BOM).
    python -m rebuttal.cd_final
"""
import hashlib, json, shutil, sys
TS = sys.argv[1]
from pathlib import Path
import numpy as np
R = Path(__file__).resolve().parents[1]; S = R / "rebuttal/out/R12"; runs = R / f"out/runs/rebuttal_CD_final_s{TS}"
X = Path(r"work\exchange\runs\countsdiff_hvg2k_final_s" + TS)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
X.mkdir(parents=True, exist_ok=True)
picks = {"Endothelial": "Endothelial_K256_g1p0", "Myeloid": "Myeloid_K32_g1p0", "Neuronal": "Neuronal_K256_g1p0"}
for t, arm in picks.items():
    a = S / f"hvg2k_samples_s{TS}" / f"{arm}.npy"; ma = json.loads(a.with_suffix(".json").read_text(encoding="utf-8"))
    parts = [{"file": str(a), "sha256": sha(a), "seed": ma["seed"], "n": ma["n"]}]
    if t == "Endothelial":
        b = S / f"hvg2k_samples_s{TS}" / f"{arm}_topup.npy"; mb = json.loads(b.with_suffix(".json").read_text(encoding="utf-8"))
        assert (mb["K"], mb["guidance"], mb["ckpt"], mb["type"]) == (ma["K"], ma["guidance"], ma["ckpt"], ma["type"]), "top-up settings differ"
        x = np.concatenate([np.load(a), np.load(b)]); parts.append({"file": str(b), "sha256": sha(b), "seed": mb["seed"], "n": mb["n"]})
        assert x.shape[0] == 10057, x.shape
        seeds = f"{ma['seed']}+{mb['seed']}"
    else:
        x = np.load(a); seeds = str(ma["seed"])
    full = f"{arm}_n{x.shape[0]}"; name = f"CountsDiff_{full}_train{TS}_samp{seeds.replace('+', '_')}"
    d = runs / full; d.mkdir(parents=True, exist_ok=True); npy = d / f"{name}.npy"; np.save(npy, x)
    m = {k: ma[k] for k in ("ckpt", "ckpt_step", "type", "K", "nfe", "guidance", "eta")}
    m.update({"request": t, "n_delivered": int(x.shape[0]), "arm": full, "seed": seeds, "parts_in_order": parts,
              "generator": "CountsDiff (r12 v2, authors' recipe)", "selected_by": "Mac, val distance rule", "npy_sha256": sha(npy)})
    (d / f"{name}.json").write_text(json.dumps(m, indent=1), encoding="utf-8")
    tx = X / full / f"{name}.npy"; tx.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(npy, str(tx) + ".part"); Path(str(tx) + ".part").replace(tx); shutil.copyfile(d / f"{name}.json", tx.with_suffix(".json"))
    assert sha(tx) == m["npy_sha256"]
    with open(X / "SHA256SUMS", "a", encoding="utf-8", newline="\n") as f: f.write(f"{m['npy_sha256']}  {full}/{name}.npy\n")
    print(f"{full}: n={x.shape[0]} seeds {seeds} sha {m['npy_sha256'][:16]}", flush=True)
