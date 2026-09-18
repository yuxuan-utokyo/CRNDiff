# -*- coding: utf-8 -*-
"""Driver for the MDLM baseline: --mode {gate,train,sample}.

    gate    run a short trajectory on untrained weights and check the absorbing-state mechanism
            signature. It takes seconds and must pass before any training starts.
    train   train to a plateau under the same early-stopping rule as our own model and deliver
            the best-validation checkpoint.
    sample  two decoding budgets, three seeds each. BOTH budgets are reported; neither is chosen
            after the fact.

Four guards against a straw-man comparison are recorded field by field in the json: fidelity to
the published objective (masked cross-entropy, the published weighting and unmasking rule),
matched capacity, matched data and split, and matched budget.

    python scripts/run_mdlm_baseline.py --mode gate
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from crndiff import config as C                                          # noqa: E402
from crndiff.io import sha256_file                                       # noqa: E402
# Data-dependent imports are deferred until after CLI parsing (--help needs no atlas).
OUTD = C.MODELS / "mdlm"
TAG = "mdlm_s20260817"
FIT_SEED = 20260817          # the fitting seed, shared with the selection and noised-state classifiers


def ours_param_count(dim: int, iscale) -> int:
    net = M.build_net(dim, iscale)
    n = int(sum(p.numel() for p in net.parameters()))
    del net
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("gate", "train", "sample"), required=True)
    ap.add_argument("--hours", type=float, default=5.0)
    ap.add_argument("--batch", type=int, default=None,
                    help="defaults to the batch of our own checkpoint (same budget); reduce it only for memory")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--steps-cap", type=int, default=60000)
    ap.add_argument("--T", type=int, default=32)
    ap.add_argument("--seed", type=int, default=20260931)
    ap.add_argument("--n", type=int, default=100000)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--gate-n", type=int, default=64)
    ap.add_argument("--gate-steps", type=int, default=32)
    ap.add_argument("--device", choices=("cuda", "cpu"), default=None,
                    help="G-M1 is a mechanism check and is device independent; use cpu while the "
                         "generation queue has the GPU, so that two GPU jobs never run at once.")
    a = ap.parse_args()
    for name in ("meta.json", "train.npy", "val.npy"):
        C.require(C.LEGACY_DATA / "all" / name, "atlas input; see README section 6")

    global MD, CFG, M
    from baselines.mdlm import model_masked_mdlm as MD
    CFG = MD.CFG
    M = MD.M
    if a.device:
        MD.DEV = torch.device(a.device)
        print(f"[device] forced to {MD.DEV}", flush=True)

    OUTD.mkdir(parents=True, exist_ok=True)
    C.GATES.mkdir(parents=True, exist_ok=True)
    (C.OUT / "summary").mkdir(parents=True, exist_ok=True)
    (C.OUT / "uncond_pools").mkdir(parents=True, exist_ok=True)
    d, meta = M.load_data(("train", "val"))
    tr, va = d["train"], d["val"]
    der = M.derive(tr)
    dim = tr.shape[1]
    ck = OUTD / f"{TAG}.pt"

    # ---------------------------------------------------------------- G-M1
    if a.mode == "gate":
        net = MD.MaskedMDLMNet(CFG.NMAX, dim).to(MD.DEV)
        if ck.exists():
            net.load_state_dict(torch.load(ck, map_location=MD.DEV, weights_only=False))
            which = "trained"
        else:
            which = "untrained (gate checks the mechanism, not the fit)"
        net.eval()
        _, traj = MD.sample_mdlm(net, a.gate_n, dim, n_steps=a.gate_steps,
                                 seed=a.seed, chunk=a.gate_n, record_trajectory=True)
        res = MD.gate_gm1(traj, net.mask_id)
        ours_n = ours_param_count(dim, der["iscale"])
        mdlm_n = int(sum(p.numel() for p in net.parameters()))
        blob = {"ticket": 200, "gate": "G-M1", "what": "absorbing-state mechanism signature",
                "weights": which, "n_cells": a.gate_n, "n_steps": a.gate_steps,
                "result": res,
                "capacity_match": {
                    "ours_n_params": ours_n, "mdlm_n_params": mdlm_n,
                    "ratio": mdlm_n / ours_n,
                    "within_10_percent": bool(abs(mdlm_n / ours_n - 1.0) <= 0.10),
                    "note": "backbone is OURS (model_hvg2k AttnX0Posterior), not the DiT of the "
                            "MDLM paper; count_proj is deleted and a token embedding "
                            "(0..NMAX plus MASK) takes its place -- an absorbing-state "
                            "diffusion needs the extra MASK symbol by construction."},
                "vocab_size": CFG.NMAX + 1, "mask_id": CFG.NMAX + 1,
                "written_at": time.strftime("%Y-%m-%d %H:%M:%S")}
        p = C.GATES / "200_GM1_mdlm.json"
        if p.exists():
            p = C.GATES / f"200_GM1_mdlm_{int(time.time())}.json"
        p.write_text(json.dumps(blob, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"[G-M1] passed={res['passed']}  unmask_events={res['n_unmask_events']}  "
              f"violations: changed={res['violations_unmasked_token_changed']} "
              f"remasked={res['violations_remasked']} still_masked={res['final_still_masked']}")
        print(f"[cap ] ours {ours_n} vs mdlm {mdlm_n} -> ratio {mdlm_n/ours_n:.4f} "
              f"(within 10%: {blob['capacity_match']['within_10_percent']})")
        print(f"[out] {p}")
        if not res["passed"]:
            (C.OUT / "summary" / "MDLM_IMPL_BLOCKED.md").write_text(
                f"# MDLM # The MDLM implementation did not pass the absorbing-state gate G-M1\n\n{json.dumps(res, indent=1, ensure_ascii=False)}\n\n"
                "The defining property of an absorbing state does not hold, so the implementation is broken and its numbers may not be used.\n",
                encoding="utf-8")
            return 1
        return 0

    # training
    if a.mode == "train":
        js = OUTD / f"{TAG}.json"
        if ck.exists() or js.exists():
            raise SystemExit(f"refusing to overwrite {ck}")
        ckpt_cfg = json.loads(
            Path(str(C.GENERATOR_CKPT).replace(".pt", ".json")).read_text(encoding="utf-8"))
        batch = a.batch or int(ckpt_cfg["batch"])
        t0 = time.time()
        ema, curve, mt = MD.train_mdlm(
            tr, va, FIT_SEED, max_seconds=a.hours * 3600.0, steps_cap=a.steps_cap,
            batch=batch, lr=a.lr, wd=a.wd, ckpt_path=ck,
            log_fn=lambda s: print(s, flush=True))
        best = Path(str(ck) + ".best")
        if best.exists():
            ema.load_state_dict(torch.load(best, map_location=MD.DEV, weights_only=False))
            print(f"[take] delivering best_val (val={mt['best_val']:.5f} at "
                  f"step {mt.get('best_val_step')})", flush=True)
        torch.save(ema.state_dict(), ck)
        ours_n = ours_param_count(dim, der["iscale"])
        blob = {
            "ticket": 200, "block": "AMEND-7 §3 MDLM", "model": "MDLM (Sahoo et al., 2024)",
            "identity": "our implementation for count data; the released code targets text "
                        "(OpenWebText / LM1B) and provides no scRNA entry point. Backbone and "
                        "capacity are matched to ours.",
            "do_not_write": "we ran MDLM",
            "ported_from": "cc/experiments/EXP08_toy_countsdiff/baselines.py::MaskedMDLM",
            "published_formulas": {
                "forward": "t ~ U(0,1); each gene position independently set to MASK w.p. t",
                "loss": "cross-entropy on masked positions, weight 1/t (MDLM / SUBS)",
                "reverse": "start all-MASK; unmask each masked token w.p. (t-s)/t; "
                           "final step forces a prediction for whatever is left"},
            "anti_strawman": {
                "1_faithful_to_published_formulas": True,
                "2_capacity_and_budget_matched": {
                    "ours_n_params": ours_n, "mdlm_n_params": mt["n_params"],
                    "ratio": mt["n_params"] / ours_n,
                    "within_10_percent": bool(abs(mt["n_params"] / ours_n - 1.0) <= 0.10),
                    "same_data": True, "same_early_stop_rule": True,
                    "backbone_is_ours_not_dit": True,
                    "batch_note": f"batch={batch}, taken from the frozen ckpt's own json "
                                  f"({ckpt_cfg['batch']}) so the budget matches ours; MDLM's "
                                  f"published batch is a text-model batch and is not transferable"},
                "3_no_tuning_for_it": {"lr": a.lr, "weight_decay": a.wd,
                                       "both_T_reported": [32, 256]},
                "4_mechanism_gate": "G-M1, see out/gates/200_GM1_mdlm.json"},
            "vocab_size": CFG.NMAX + 1, "mask_id": CFG.NMAX + 1,
            "truncation_disclosure": "counts are clipped to NMAX = %d, the same support as ours; "
                                     "the fraction of real cells containing any count > NMAX is "
                                     "in out/summary/200_support_truncation.json" % CFG.NMAX,
            "gene_hash": meta["gene_hash"], "argv": sys.argv,
            "curve": curve, **mt,
            "ckpt": str(ck), "ckpt_sha256": None,
            "environment": {"python": sys.version, "numpy": np.__version__,
                            "torch": torch.__version__, "cuda": torch.version.cuda,
                            "platform": platform.platform()},
            "total_walltime_s": time.time() - t0,
            "written_at": time.strftime("%Y-%m-%d %H:%M:%S")}
        blob["ckpt_sha256"] = sha256_file(ck)
        js.write_text(json.dumps(blob, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"[MDLM] steps={mt['steps_done']} stop={mt['stop_reason']} "
              f"best_val={mt['best_val']:.5f} @ step {mt['best_val_step']}  "
              f"params {mt['n_params']} vs ours {ours_n} "
              f"({mt['n_params']/ours_n:.3f}x)")
        print(f"[out] {ck}\n[out] {js}")
        return 0

    # sampling
    out = C.OUT / "uncond_pools" / f"mdlm_uncond_T{a.T}_seed{a.seed}_n{a.n}.npy"
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    net = MD.MaskedMDLMNet(CFG.NMAX, dim).to(MD.DEV)
    net.load_state_dict(torch.load(C.require(ck, "MDLM ckpt"), map_location=MD.DEV,
                                   weights_only=False))
    net.eval()
    t1 = time.time()
    x, traj = MD.sample_mdlm(net, a.n, dim, n_steps=a.T, seed=a.seed, chunk=a.chunk,
                             record_trajectory=True, log_fn=lambda s: print(s, flush=True))
    wall = time.time() - t1
    gm1 = MD.gate_gm1(traj, net.mask_id)
    x16 = x.astype(np.int16)
    assert np.array_equal(x16.astype(np.int64), x), "int16 cast is not lossless"
    np.save(out, x16)
    train_js = json.loads((OUTD / f"{TAG}.json").read_text(encoding="utf-8"))
    y = x16[:20000]
    blob = {"ticket": 200, "block": "AMEND-7 §3 MDLM sampling",
            "generator": "mdlm", "model": "MDLM (Sahoo et al., 2024)",
            "identity": train_js["identity"], "T": a.T, "n_steps": a.T,
            "NFE_per_sample": a.T, "NFE_chain": a.T,
            "seed": a.seed, "n": int(a.n), "chunk": a.chunk,
            "vocab_size": CFG.NMAX + 1, "mask_id": CFG.NMAX + 1,
            "library_size_source": "generated",
            "training_budget": {k: train_js.get(k) for k in
                                ("steps_done", "stop_reason", "best_val", "best_val_step",
                                 "plateau_evidence", "walltime_s", "batch", "lr")},
            "gate_G_M1_on_this_run": gm1,
            "ckpt": str(ck), "ckpt_sha256": train_js["ckpt_sha256"],
            "gene_hash": meta["gene_hash"], "argv": sys.argv,
            "walltime_s": wall, "s_per_cell": wall / max(a.n, 1),
            "dtype": str(x16.dtype), "shape": list(x16.shape),
            "npy_sha256": sha256_file(out),
            "device": str(MD.DEV), "device_class": "A", "machine": "machine1",
            "local_sanity": {"library_size_mean": float(y.sum(1).mean()),
                             "zero_fraction": float((y == 0).mean()),
                             "max": int(y.max()), "mean": float(y.mean()),
                             "fraction_negative": 0.0, "fraction_non_integer": 0.0},
            "environment": {"python": sys.version, "numpy": np.__version__,
                            "torch": torch.__version__, "cuda": torch.version.cuda},
            "written_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    out.with_suffix(".json").write_text(json.dumps(blob, indent=1, ensure_ascii=False),
                                        encoding="utf-8")
    print(f"[mdlm] T={a.T} seed={a.seed} n={a.n} {wall/60:.1f} min "
          f"({wall/a.n*1000:.3f} ms/cell)  G-M1 passed={gm1['passed']}  "
          f"lib={blob['local_sanity']['library_size_mean']:.1f}")
    print(f"[out] {out}")
    return 0 if gm1["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
