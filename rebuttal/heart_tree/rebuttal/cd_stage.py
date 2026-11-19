# -*- coding: utf-8 -*-
"""Stage CountsDiff HVG2K sample arrays (rebuttal/out/R12/hvg2k_samples/<arm>.npy + .json from countsdiff2k.py sample)
as runs out/runs/rebuttal_CD/<arm>/<name>.{npy,json} for the frozen scorer, and copy them to _data_exchange/runs/countsdiff_hvg2k/<arm>/
seed<seed>.npy with SHA256SUMS (two columns, LF, no BOM) and NAMES.tsv. Arrays are copied byte for byte; existing targets are skipped.
    python -m rebuttal.cd_stage
"""
import hashlib, json, shutil
from pathlib import Path
R = Path(__file__).resolve().parents[1]; src = R / "rebuttal/out/R12/hvg2k_samples"; runs = R / "out/runs/rebuttal_CD"
X = Path(r"work\exchange\runs\countsdiff_hvg2k")
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
X.mkdir(parents=True, exist_ok=True); n = 0
for npy in sorted(src.glob("*.npy")):
    if not npy.with_suffix(".json").exists(): continue
    m = json.loads(npy.with_suffix(".json").read_text(encoding="utf-8")); arm = npy.stem
    name = f"CountsDiff_{arm}_train20260625_samp{m['seed']}"; d = runs / arm; d.mkdir(parents=True, exist_ok=True)
    if not (d / f"{name}.json").exists():
        shutil.copyfile(npy, d / f"{name}.npy")
        m.update({"request": m["type"], "n_delivered": m["n"], "arm": arm, "generator": "CountsDiff (r12 v2, authors' recipe)", "npy_sha256": sha(npy)})
        (d / f"{name}.json").write_text(json.dumps(m, indent=1), encoding="utf-8")
    t = X / arm / f"seed{m['seed']}.npy"
    if not t.exists():
        t.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(npy, str(t) + ".part"); Path(str(t) + ".part").replace(t)
        shutil.copyfile(npy.with_suffix(".json"), t.with_suffix(".json"))
        with open(X / "SHA256SUMS", "a", encoding="utf-8", newline="\n") as f: f.write(f"{sha(t)}  {arm}/seed{m['seed']}.npy\n")
        with open(X / "NAMES.tsv", "a", encoding="utf-8", newline="\n") as f: f.write(f"{arm}/seed{m['seed']}.npy\t{name}\n")
        n += 1
print("staged/copied", n)
