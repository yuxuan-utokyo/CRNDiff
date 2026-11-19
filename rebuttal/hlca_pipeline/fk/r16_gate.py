# -*- coding: utf-8 -*-
"""R16 bank-placement gate (vice 11-09 21:0x condition 2), HVG2K at NMAX 512, the deployed generator. Runs the frozen code
unedited, either as is (bank = bank_dedup.DedupBank) or with rebuttal/bank_place.install() (bank = DedupBankV2 host_resident).
Outputs go under $R16_GATE_ROOT (sampler ROOT/LOG redirected, as rebuttal/r13_make_pool.py does) and $HVG_OUT (FK runs).

    python -m rebuttal.r16_gate bank                              # frozen bank_dedup_v2.gate_vs_v1, K 32, host_resident=True
    python -m rebuttal.r16_gate pool [--placed] -- <sample_hvg2k_noa4.py arguments>
    python -m rebuttal.r16_gate fk   [--placed] -- <experiments/run_one.py arguments>
"""
import json, os, sys
from pathlib import Path
sys.dont_write_bytecode = True
HVG = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(HVG))
argv = sys.argv[1:]
mode = argv[0]; placed = "--placed" in argv[1:argv.index("--")] if "--" in argv else "--placed" in argv
rest = argv[argv.index("--") + 1:] if "--" in argv else []
assert mode in ("bank", "pool", "fk"), mode

from hvg import config as C          # noqa: E402
C.add_base_code_to_path()
import sample_hvg2k_noa4 as S        # noqa: E402  (frozen)
GR = Path(os.environ.get("R16_GATE_ROOT", str(Path.home() / "QIU/hlca/r16_gate")))
(GR / "Baseline" / "OURS" / "results").mkdir(parents=True, exist_ok=True)
S.ROOT, S.LOG = GR, GR / "RUN_LOG.md"
if placed:
    from rebuttal import bank_place
    bank_place.install()
print(f"[r16_gate] mode={mode} placed={placed} NMAX={S.CFG.NMAX} bank={S.DedupBank.__module__}.{S.DedupBank.__name__}", flush=True)

if mode == "bank":
    from rebuttal import bank_place
    import torch
    d, _ = S.M.load_data(("train",)); der = S.M.derive(d["train"]); nmax = S.CFG.NMAX
    gf, nfe = S.M.build_grid(32, S.M.horizon()); lf = S.VO._logfact_arr(nmax)
    r = bank_place.gate(der["V"], gf, nmax, lambda tau, v, nm: S.VO.kmat(tau, v, nm, lf), lambda v, nm: S.VO.pois_prior(v, nm, lf), S.DEV)
    print("[r16_gate] gate_vs_v1", json.dumps(r), flush=True)
    sys.exit(0 if r["bit_exact"] else 1)
elif mode == "pool":
    sys.argv = ["sample_hvg2k_noa4.py"] + rest
    S.main()
else:
    from experiments import run_one
    sys.exit(run_one.main(rest))
