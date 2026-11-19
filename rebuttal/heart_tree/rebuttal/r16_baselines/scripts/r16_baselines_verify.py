"""Byte check of r16_baselines_trainlib.py against the DG seed-1 outputs of r13_baselines_trainlib.py.
Every counts file the reference run wrote for scVI 20261050 / scANVI 20261060 must exist in the new run with the same sha256, and
no extra counts files. generate_meta.json is compared after dropping wall-time-like keys (it is checked field by field otherwise).
    python r16_baselines_verify.py NEW_OUT REF_OUT OUT.json
"""
import hashlib, json, sys
from pathlib import Path

new, ref, out = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
res, ok = {"new": str(new), "ref": str(ref), "runs": {}}, True


def strip(d):
    if isinstance(d, dict):
        return {k: strip(v) for k, v in d.items() if "walltime" not in k and "time" != k}
    return [strip(v) for v in d] if isinstance(d, list) else d


RUNS = sys.argv[4].split(",") if len(sys.argv) > 4 else ["scvi_s20261050", "scanvi_s20261060"]   # optional 4th arg: comma list
for run in RUNS:
    rf = sorted((ref / run / "out" / "counts").glob("*.npy")); nf = sorted((new / run / "out" / "counts").glob("*.npy"))
    files = {f.name: {"ref": sha(f), "new": sha(new / run / "out" / "counts" / f.name) if (new / run / "out" / "counts" / f.name).exists() else None} for f in rf}
    same_set = [f.name for f in rf] == [f.name for f in nf]
    all_eq = all(v["ref"] == v["new"] for v in files.values())
    ma, mb = json.loads((ref / run / "out" / "generate_meta.json").read_text()), json.loads((new / run / "out" / "generate_meta.json").read_text())
    meta_eq = strip(ma) == strip(mb)
    res["runs"][run] = {"n_files": len(rf), "same_file_set": same_set, "all_counts_byte_identical": all_eq, "generate_meta_equal_ex_time": meta_eq, "files": files}
    ok &= same_set and all_eq and meta_eq and len(rf) > 0
    print(run, len(rf), "files; same set", same_set, "; all identical", all_eq, "; meta equal", meta_eq, flush=True)
res["ALL_OK"] = bool(ok); res["script_sha256"] = sha(__file__)
json.dump(res, open(out, "w"), indent=1); print("ALL_OK", ok)
sys.exit(0 if ok else 1)
