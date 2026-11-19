# -*- coding: utf-8 -*-
"""A′ sampling (ticket 200 · AMEND-2 section 2.1: unconditional sampler + conditional ckpt + given label).

**the frozen sampler is unchanged to the byte**: this file `import sample_hvg2k_noa4`(the frozen copy itself),
by its `main()` copies the same preparation step by step (the same `DedupBank`, the same chunk seeds
`samp*1000 + chunk_index`, the same `_NO_A4` / `_LA_TAU=0` state),
then call **its own** `sample_chunk(net, bank, dmax_t, b, seed, nmax, batched=True)`.
`net` is `hvg.model_conditional.FixedLabel` thin shell that only `.post(n_t, t)` converted to
`.post(n_t, t, y=const)` — the sampler sees exactly the same interface as the frozen network.

why the frozen `main()`: its own `build_net` + `load_state_dict`,
no parameter allows plugging in the conditional network, and changing it means touching a frozen copy (red line).

gate **G-A2**:`--label-index -1`(label channel not used) + seed 20260931 + n=2048,
**bit-identical** to the first 2048 rows of our unconditional pool.
> n 2048 instead of 2000:`cell_chunk=256`, 2000 would leave block 8 with only 208 rows,
> the RNG consumption of that block differs from a full block, so **the prefix could never be bit-identical anyway**. 2048 = 8 full blocks,
> is the only testable version of the claim that the first 2000 rows are bit-identical; stricter than the original, not looser.

    python -m tools.t200_sample_conditional --ckpt <cond.pt> --label Endothelial \
        --n 10057 --seed 20260931 --out out/runs/aprime/...
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hvg import config as C                                          # noqa: E402
from hvg.io import sha256_file                                       # noqa: E402
from hvg import model_conditional as MC                              # noqa: E402

C.add_base_code_to_path()
import config_hvg2k as CFG                                           # noqa: E402
import model_hvg2k as M                                              # noqa: E402
import sample_hvg2k_noa4 as S                                        # noqa: E402
from bank_dedup import DedupBank                                     # noqa: E402
import importlib                                                     # noqa: E402
VO = importlib.import_module("panel61_shared.vendor_ours")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="conditional ckpt (A′ training output)")
    ap.add_argument("--label", default=None, help="type name; if not given, use the unlabelled path (G-A2)")
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--K", type=int, default=32)
    ap.add_argument("--cell-chunk", type=int, default=256)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", default="aprime")
    ap.add_argument("--experiment", default="aprime")
    a = ap.parse_args()

    out = Path(a.out)
    if out.exists() or out.with_suffix(".json").exists():
        raise SystemExit(f"refusing to overwrite {out}")
    out.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    d, meta = M.load_data(("train",))
    tr = d["train"]
    der = M.derive(tr)
    dim, nmax = tr.shape[1], CFG.NMAX
    T = M.horizon()
    logfact = VO._logfact_arr(nmax)
    vocab = MC.label_vocabulary()
    net = MC.build_conditional(dim, der["iscale"], len(vocab)).to(MC.DEV)
    sd = torch.load(a.ckpt, map_location=MC.DEV, weights_only=False)
    net.load_state_dict(sd)
    net.eval()
    li = None if a.label in (None, "", "NONE") else vocab.index(a.label)
    wrapped = MC.FixedLabel(net, li)
    dmax_t = torch.as_tensor(der["dmax"], dtype=torch.long, device=MC.DEV)

    # module state of the frozen sampler, set item by item as in main() (--no-a4, --tau 0.0, no --last-only)
    S._NO_A4 = True
    S._LA_TAU = 0.0
    S._LA_PI_TAU = None
    S._LA_LAST_ONLY = False

    gf, nfe = M.build_grid(a.K, T)
    bank = DedupBank(der["V"], gf, nmax, lambda tau, v, nm: VO.kmat(tau, v, nm, logfact),
                     lambda v, nm: VO.pois_prior(v, nm, logfact), MC.DEV)
    g = bank.verify(lambda tau, v, nm: VO.kmat(tau, v, nm, logfact), der["V"])
    if not g["bit_exact"]:
        raise SystemExit(f"dedup bank gate not bit-exact: {g}")

    t1 = time.time()
    parts = []
    for ci, lo in enumerate(range(0, a.n, a.cell_chunk)):
        b = min(a.cell_chunk, a.n - lo)
        parts.append(S.sample_chunk(wrapped, bank, dmax_t, b, int(a.seed) * 1000 + ci,
                                    nmax, batched=True))
        if ci % 20 == 0:
            print(f"  chunk {ci} ({lo + b}/{a.n})  {time.time()-t1:.0f}s", flush=True)
    gen = np.concatenate(parts, 0).astype(np.int16)
    wall = time.time() - t1
    np.save(out, gen)

    blob = {"ticket": 200, "block": "A-prime", "experiment": a.experiment, "arm": a.arm,
            "stem": out.stem, "request": a.label, "target": a.label,
            "label": a.label, "label_index": li, "label_vocabulary": vocab,
            "sampler": "frozen sample_hvg2k_noa4.sample_chunk (imported, not modified)",
            "sampler_file": str(C.BASE_CODE / "sample_hvg2k_noa4.py"),
            "sampler_file_sha256": sha256_file(C.BASE_CODE / "sample_hvg2k_noa4.py"),
            "config_hvg2k_sha256": sha256_file(C.BASE_CODE / "config_hvg2k.py"),
            "conditioning": "training-time label embedding added to the time embedding",
            "seed": a.seed, "n": int(a.n), "n_out": int(gen.shape[0]),
            "n_delivered": int(gen.shape[0]),
            "K": a.K, "NFE": int(nfe), "NFE_chain": int(nfe), "tau": 0.0,
            "no_a4": True, "batched": True, "cell_chunk": a.cell_chunk,
            "ckpt_path": str(Path(a.ckpt).resolve()), "ckpt_sha256": sha256_file(a.ckpt),
            "logpi_path": None, "logpi_sha256": None,
            "gene_hash": meta["gene_hash"], "argv": sys.argv,
            "walltime_s": wall, "setup_walltime_s": t1 - t0,
            "dtype": str(gen.dtype), "shape": list(gen.shape),
            "npy_sha256": sha256_file(out), "samples_sha256": sha256_file(out),
            "device": str(MC.DEV), "device_class": "A", "machine": "machine1",
            "unique_cell_fraction": 1.0, "unique_lineage_fraction": 1.0,
            "lineage_note": "iid_by_construction (no resampling in the unconditional chain)",
            "n_intermediate_resamples": 0,
            "aux_calls": {"model": None, "n_calls": 0, "device": None},
            "kernel_bank_gate": g,
            "mean": float(gen.mean()), "max": int(gen.max()),
            "zero_fraction": float((gen == 0).mean()),
            "library_size_mean": float(gen.sum(1).mean()),
            "environment": {"python": sys.version, "numpy": np.__version__,
                            "torch": torch.__version__, "cuda": torch.version.cuda,
                            "platform": platform.platform()},
            "written_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    out.with_suffix(".json").write_text(json.dumps(blob, indent=1, ensure_ascii=False),
                                        encoding="utf-8")
    print(f"[sample] {out}  n={gen.shape[0]} {wall/60:.1f}min  sha256={blob['npy_sha256'][:16]}…")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
