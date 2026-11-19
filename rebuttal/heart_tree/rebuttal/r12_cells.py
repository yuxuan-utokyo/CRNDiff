# -*- coding: utf-8 -*-
"""R12 cells for the Mac. Read-only, one-off. For every Table 1 run of OURS, MDLM, CFGen, scVI, scANVI
(w2_runs.TABLE1, paths from w2_runs.run_list) and the 3 CountsDiff picks, on all 3 targets: exactly the cells w2_runs.load_cells hands
the R1/R2/R3 scripts (same subsample seed rule, same order), stored as int16 with np.savez_compressed under key "x" in
_data_exchange/r12_cells2500/<target>/<arm>_<train>_<samp>.npz; a file over 10 MB after compression is NOT copied
(listed in r12_cells_oversize.tsv next to the log, size reported); NAMES.tsv = file, source path, n, subsample rule; SHA256SUMS appended.
NB: load_cells keeps N_OUT[target] cells, i.e. 10,057 for Endothelial and 2,500 for Myeloid/Neuronal.
    python -m rebuttal.r12_cells
"""
import hashlib, io, json, sys
from pathlib import Path
import numpy as np
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from rebuttal import w2_runs as W
X = Path(r"work\exchange\r12_cells2500")
X.mkdir(parents=True, exist_ok=True)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
CD = ROOT / "out/runs/rebuttal_CD_final"
CDP = {"Endothelial": "Endothelial_K256_g1p0_n10057", "Myeloid": "Myeloid_K32_g1p0_n2500", "Neuronal": "Neuronal_K256_g1p0_n2500"}
done = set((X / "NAMES.tsv").read_text(encoding="utf-8").split("\n")) if (X / "NAMES.tsv").exists() else set()
for t in W.TARGETS:
    runs = [r for r in W.run_list(t) if W.in_table1(r["arm"], r["train"], r["samp"])]
    for f in sorted((CD / CDP[t]).glob("*.npy")):
        m = json.loads(f.with_suffix(".json").read_text(encoding="utf-8"))
        runs.append(dict(arm="CountsDiff", train=20260625, samp=str(m["seed"]).replace("+", "_"), path=f))
    assert len(runs) == 16, (t, len(runs))
    for r in runs:
        x = W.load_cells(r, t); n0 = int(np.load(r["path"], mmap_mode="r").shape[0])
        xi = x.astype(np.int16); assert np.array_equal(xi.astype(np.float64), x), "not integer / int16 overflow"
        name = f"{t}/{r['arm']}_{r['train']}_{r['samp']}.npz"; p = X / name
        if any(l.startswith(name + "\t") for l in done): continue
        buf = io.BytesIO(); np.savez_compressed(buf, x=xi); b = buf.getvalue()
        if len(b) > 10_000_000:
            with open(ROOT / "rebuttal/logs/r12_cells_oversize.tsv", "a", encoding="utf-8") as fh: fh.write(f"{name}\t{len(b)}\n")
            print(f"OVERSIZE not copied: {name} {len(b)/1e6:.2f} MB", flush=True); continue
        p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(b)
        rule = (f"SeedSequence([{W.SUBSAMPLE_SEED}, samp, train]) sorted choice {xi.shape[0]} of {n0}" if n0 > xi.shape[0] else f"all {n0} rows, as stored")
        with open(X / "SHA256SUMS", "a", encoding="utf-8", newline="\n") as fh: fh.write(f"{sha(p)}  {name}\n")
        with open(X / "NAMES.tsv", "a", encoding="utf-8", newline="\n") as fh:
            fh.write(f"{name}\t{Path(r['path']).relative_to(ROOT).as_posix()}\tn={xi.shape[0]}{' (Endothelial: N_OUT 10,057, not 2,500)' if t == 'Endothelial' else ''}\t{rule}\n")
        print(f"{name}: n={xi.shape[0]} of {n0}, {p.stat().st_size/1e6:.2f} MB", flush=True)
