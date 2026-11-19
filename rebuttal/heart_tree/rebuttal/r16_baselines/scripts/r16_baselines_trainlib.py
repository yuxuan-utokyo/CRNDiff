#!/usr/bin/env python
"""R16: variant of r13_baselines_trainlib.py (ae93a341..., left unchanged) for the HLCA scVI/scANVI baselines.

Same as the original: regenerate scVI and scANVI samples from already-trained models (made by r13_baselines.py), with library sizes
drawn from TRAIN cells of the target; the generate scripts are imported unchanged and only their library-pool function is replaced,
in this process. Added options; with none of them given, the behaviour is the original's (checked byte for byte on DG seed 1):
  --methods      which methods to run (default: scvi scanvi, the original's order)
  --n-seeds k    only the first k seeds of each method in SPEC (default 3 = all, as the original)
  --sample-seed  set the generate scripts' SAMPLE_SEED (default: not set, i.e. scVI 20261051 / scANVI 20261061 as on DG); the output
                 file names carry the seed used
  --no-symlink   copy the model folder from --from instead of os.symlink (Windows without Developer Mode)
  --torch-global-seed  call torch.manual_seed(S) right before each method/seed's generation (default: not called, as the original).
                 Why: the generate scripts seed z with their own Generator, but px.sample() draws from torch's GLOBAL generator, which
                 this torch (2.14.1) seeds randomly per process and nothing in the scripts sets; so without this option two runs of
                 the same command give different counts (found 11-10 while checking this variant; the original does the same).
The symlink now points at the absolute --from path (the original linked the path as given, which breaks for a relative --from).
  python r16_baselines_trainlib.py --data DATA --from RUNS --out OUT --types A B C [--n-gen N] [--methods ...] [--n-seeds k]
                                   [--sample-seed S] [--no-symlink] [--torch-global-seed S]
"""
from __future__ import annotations
import argparse, json, os, shutil, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from r13_baselines import SPEC, load, sha

ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("--data", type=Path, required=True); ap.add_argument("--from", dest="src", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True); ap.add_argument("--types", nargs="+", required=True); ap.add_argument("--n-gen", type=int, default=20000)
ap.add_argument("--methods", nargs="+", default=list(SPEC), choices=list(SPEC)); ap.add_argument("--n-seeds", type=int, default=3)
ap.add_argument("--sample-seed", type=int, default=None); ap.add_argument("--no-symlink", action="store_true")
ap.add_argument("--torch-global-seed", type=int, default=None); a = ap.parse_args()
gene_hash = json.loads((a.data / "meta.json").read_text(encoding="utf-8"))["gene_hash"]


def train_pool(ctype):   # unchanged from r13_baselines_trainlib.py
    X = np.load(a.data / "train.npy", mmap_mode="r"); y = np.load(a.data / "train_celltype.npy", allow_pickle=True).astype(str)
    return np.asarray(X[np.sort(np.flatnonzero(y == ctype))], dtype=np.float64).sum(1)


for method in a.methods:
    sp = SPEC[method]
    for seed in sp["seeds"][: a.n_seeds]:
        root, src = a.out / f"{method}_s{seed}", a.src / f"{method}_s{seed}"
        if (root / "out" / "generate_meta.json").exists():
            print(f"[{method} s{seed}] exists, skipped", flush=True); continue
        assert (src / "logs").is_dir() and (src / "out" / "model").is_dir(), f"no trained model in {src}"
        (root / "logs").mkdir(parents=True, exist_ok=True); (root / "out").mkdir(parents=True, exist_ok=True)
        for f in (src / "logs").iterdir():
            shutil.copy2(f, root / "logs" / f.name)
        if not (root / "out" / "model").exists():
            if a.no_symlink:
                shutil.copytree(src / "out" / "model", root / "out" / "model")
            else:
                os.symlink((src / "out" / "model").resolve(), root / "out" / "model")
        before = sha(sp["gen"]); t0 = time.time()
        gm = load(sp["gen"], f"_r16_gen_{method}_{seed}")
        gm.HERE, gm.DATA, gm.TYPES, gm.N_GEN = root, a.data, a.types, a.n_gen
        if hasattr(gm, "GENE_HASH"):
            gm.GENE_HASH = gene_hash
        if a.sample_seed is not None:
            assert hasattr(gm, "SAMPLE_SEED"), method
            gm.SAMPLE_SEED = a.sample_seed
        name = "real_library_pool" if hasattr(gm, "real_library_pool") else "lib_pool"
        assert hasattr(gm, name), method
        setattr(gm, name, train_pool)
        if a.torch_global_seed is not None:
            import torch
            torch.manual_seed(a.torch_global_seed)
        gm.main()
        assert sha(sp["gen"]) == before
        with (a.out / "runs.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({"method": method, "seed": seed, "library": "train cells of the target", "replaced": name,
                                "sample_seed": int(gm.SAMPLE_SEED), "sample_seed_overridden": a.sample_seed is not None,
                                "model": "copied" if a.no_symlink else "symlink", "n_gen": a.n_gen, "types": a.types,
                                "torch_global_seed": a.torch_global_seed,
                                "generate_walltime_s": time.time() - t0, "script_sha256": before,
                                "wrapper_sha256": sha(Path(__file__))}) + "\n")
        print(f"[done] {method} s{seed} {time.time() - t0:.0f}s", flush=True)
