# -*- coding: utf-8 -*-
"""Driver for value-guided candidate resampling (SVDD-style, Li et al., 2024).

The arm is named the same way in the paper and in the table headers. It is NOT classifier
guidance, which modifies the CTMC rates, and it is NOT the exact transition reweighting of
Schiff et al. (2024).

    python scripts/sample_value_guided.py --request Endothelial --seed 20260987 \
        --gamma 1 --scorer noised
"""
from __future__ import annotations

import argparse
import importlib
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
from crndiff.sampler import PortableHGBT, make_twist_generator           # noqa: E402
from crndiff.sampler_value_guided import VGRSampler                     # noqa: E402

# Data-dependent imports are deferred until after CLI parsing (--help needs no atlas).


def build_scorer(kind: str, request: str, dev):
    """Return (callable, metadata). The callable maps a kind, a batch and a step to probabilities."""
    if kind == "clean":
        d = C.MODELS / "selection" / request
        clf = PortableHGBT(C.require(d / "selection_classifier_portable.npz", f"S1 {request}"))
        feat = np.load(d / "features.npy")
        log = json.loads((d / "log.json").read_text(encoding="utf-8"))

        def fn(_kind, x, _k):
            z = np.asarray(x, dtype=np.float32)
            lib = z.sum(axis=1, dtype=np.float64)
            sc = (1.0e4 / np.maximum(lib, 1.0)).astype(np.float32)
            xf = np.log1p(z[:, feat] * sc[:, None]).astype(np.float32, copy=False)
            return np.asarray(clf.predict_probability(xf), dtype=np.float64)

        return fn, {"scorer": "clean", "model": "S1 selection classifier (clean state)",
                    "dir": str(d), "val_auroc": log["val_auroc"],
                    "portable_sha256": sha256_file(d / "selection_classifier_portable.npz"),
                    "device": "cpu"}

    if kind == "noised":
        from scripts.fit_noised_value_classifier import NoisedClassifier     # noqa: PLC0415
        d = C.MODELS / "noised_clf"
        log = json.loads(C.require(d / "log.json", "noised classifier log").read_text(
            encoding="utf-8"))
        vocab = log["label_vocabulary"]
        li = vocab.index(request)
        net = NoisedClassifier(int(log["dim"]), int(log["n_steps"]), len(vocab),
                               emb_dim=int(log.get("emb_dim", 64)),
                               hidden=int(log.get("hidden", 512))).to(dev)
        net.load_state_dict(torch.load(str(d / "noised_clf.pt"), map_location=dev,
                                       weights_only=False))
        net.eval()

        @torch.no_grad()
        def fn(_kind, x, k):
            t = x if torch.is_tensor(x) else torch.as_tensor(np.asarray(x), device=dev)
            p = net.probability(t.to(torch.float32), int(k))
            return p[:, li].detach().cpu().numpy().astype(np.float64)

        return fn, {"scorer": "noised", "model": "noised-state MLP classifier",
                    "dir": str(d), "ckpt_sha256": log["ckpt_sha256"],
                    "val_accuracy_per_k": log["val_accuracy_per_k"],
                    "label_index": li, "device": str(dev)}

    raise SystemExit(f"unknown scorer {kind}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--request", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--gamma", type=float, required=True)
    ap.add_argument("--scorer", choices=("noised", "clean"), required=True)
    ap.add_argument("--n-particles", type=int, default=None)
    ap.add_argument("--n-out", type=int, default=None)
    ap.add_argument("--J", type=int, default=C.DEFAULT_J)
    ap.add_argument("--K", type=int, default=32)
    ap.add_argument("--cell-chunk", type=int, default=256)
    ap.add_argument("--experiment", default="vgr")
    ap.add_argument("--arm", default=None)
    ap.add_argument("--endpoint-topn", action="store_true",
                    help="compute-matched compute-matched variant: one chain per unit of budget, delivering by forced top-N on the same scorer N")
    a = ap.parse_args()
    for name in ("meta.json", "train.npy", "val.npy"):
        C.require(C.LEGACY_DATA / "all" / name, "atlas input; see README section 6")

    C.add_base_code_to_path()
    import config_hvg2k as CFG
    import model_hvg2k as M
    import sample_hvg2k_noa4 as S
    from bank_dedup import DedupBank
    VO = importlib.import_module("panel61_shared.vendor_ours")

    D = json.loads(C.require(C.OUT / "summary" / "200_derived_constants.json",
                             "derived constants").read_text(encoding="utf-8"))
    N = int(D["types"][a.request]["N"])
    Mp = a.n_particles or (int(D["types"][a.request]["B1"]) if a.endpoint_topn else N)
    n_out = a.n_out or N
    gtag = f"g{a.gamma}".replace(".", "p")
    arm = a.arm or (f"{a.request}_vgr_{a.scorer}_{gtag}"
                    + ("_cm" if a.endpoint_topn else ""))
    stem = f"{arm}_M{Mp}_N{n_out}_J{a.J}_K{a.K}_seed{a.seed}"
    outd = C.RUNS / a.experiment / arm
    outd.mkdir(parents=True, exist_ok=True)
    npy, jsn = outd / f"{stem}.npy", outd / f"{stem}.json"
    if npy.exists() or jsn.exists():
        raise SystemExit(f"refusing to overwrite {npy}")

    t0 = time.time()
    d, meta = M.load_data(("train",))
    tr = d["train"]
    der = M.derive(tr)
    dim, nmax = tr.shape[1], CFG.NMAX
    T = M.horizon()
    logfact = VO._logfact_arr(nmax)
    net = M.build_net(dim, der["iscale"]).to(M.DEV)
    net.load_state_dict(torch.load(str(C.GENERATOR_CKPT), map_location=M.DEV,
                                   weights_only=False))
    net.eval()
    dmax_t = torch.as_tensor(der["dmax"], dtype=torch.long, device=M.DEV)
    gf, nfe = M.build_grid(a.K, T)
    bank = DedupBank(der["V"], gf, nmax, lambda tau, v, nm: VO.kmat(tau, v, nm, logfact),
                     lambda v, nm: VO.pois_prior(v, nm, logfact), M.DEV)
    gate = bank.verify(lambda tau, v, nm: VO.kmat(tau, v, nm, logfact), der["V"])
    if not gate["bit_exact"]:
        raise SystemExit(f"dedup bank gate not bit-exact: {gate}")

    scorer, smeta = build_scorer(a.scorer, a.request, M.DEV)
    torch_gen = torch.Generator(device=M.DEV).manual_seed(int(a.seed) + 1)
    twist_gen = make_twist_generator(M.DEV, a.seed)
    vgr = VGRSampler(S, net, bank, dmax_t, scorer, J=a.J, gamma=a.gamma,
                     cell_chunk=a.cell_chunk, torch_gen=torch_gen, twist_gen=twist_gen,
                     no_a4=True)
    t1 = time.time()
    x, diag = vgr.run(Mp, a.seed, a.scorer, verbose=True)
    wall = time.time() - t1

    endpoint = None
    if a.endpoint_topn:
        s = scorer(a.scorer, x if a.scorer == "clean" else
                   torch.as_tensor(x, device=M.DEV), len(bank.tgrid) - 1)
        order = np.argsort(-np.asarray(s, dtype=np.float64), kind="stable")[:n_out]
        idx = np.sort(order)
        endpoint = {"variant": "compute_matched_endpoint_top_N",
                    "n_candidates": int(Mp), "selected": int(len(idx)),
                    "score_min": float(np.asarray(s)[order].min())}
        x = x[idx]
    elif x.shape[0] > n_out:
        x = x[:n_out]
    np.save(npy, x.astype(np.int16))

    n_steps = len(bank.tgrid) - 1
    blob = {"ticket": 200, "block": "M1-7b", "experiment": a.experiment, "arm": arm,
            "stem": stem, "request": a.request, "target": a.request, "seed": a.seed,
            "method": "value-guided candidate resampling (SVDD-style, Li et al. 2024)",
            "method_note": "does NOT modify the CTMC rates (Nisonoff 2024) and is NOT the "
                           "exact transition reweighting of Schiff 2024; each particle picks "
                           "one of J candidates by a multinomial on classifier value.",
            "sampler": "vgr", "scorer": a.scorer, "scorer_meta": smeta,
            "gamma": a.gamma, "J": a.J, "K": a.K, "tau": 0.0, "tilt_on": False,
            "gamma_note": ("J=16 at this candidate count the largest exponent is close to an argmax over the candidates,"
                           if a.gamma >= 4 else None),
            "n_particles": int(Mp), "n_out": int(x.shape[0]),
            "n_delivered": int(x.shape[0]),
            "endpoint_selection": endpoint,
            "cell_chunk": a.cell_chunk, "no_a4": True,
            "NFE": int(n_steps), "NFE_chain": int(nfe),
            "generator_forward_passes_per_particle": int(n_steps),
            "NFE_total": int(Mp * n_steps),
            "NFE_fk": int(D["types"][a.request]["NFE_fk"]),
            "aux_calls": {"model": smeta["scorer"], "n_calls": int(diag["aux_calls"]),
                          "device": smeta["device"]},
            "bridge_draws_per_step_per_particle": diag["bridge_draws_per_step_per_particle"],
            "diagnostics": {"per_step_mean_max_weight": diag["per_step_mean_max_weight"],
                            "per_step_mean_ess_J": diag["per_step_mean_ess_J"]},
            "n_intermediate_resamples": 0,
            "unique_lineage_fraction": 1.0,
            "unique_cell_fraction": float(
                len(np.unique(np.ascontiguousarray(x.astype(np.int16)).view(
                    np.dtype((np.void, 2 * x.shape[1]))))) / max(x.shape[0], 1)),
            "lineage_note": "no cross-particle weights and no resampling: lineages are "
                            "independent by construction",
            "ckpt_path": str(C.GENERATOR_CKPT), "ckpt_sha256": C.GENERATOR_CKPT_SHA256,
            "logpi_path": None, "logpi_sha256": None,
            "gene_hash": meta["gene_hash"], "argv": sys.argv,
            "walltime_s": wall, "setup_walltime_s": t1 - t0,
            "dtype": str(x.dtype), "shape": list(x.shape),
            "npy_sha256": None, "device": str(M.DEV), "device_class": "A",
            "machine": "machine1", "kernel_bank_gate": gate,
            "mean": float(x.mean()), "max": int(x.max()),
            "zero_fraction": float((x == 0).mean()),
            "library_size_mean": float(x.sum(1).mean()),
            "environment": {"python": sys.version, "numpy": np.__version__,
                            "torch": torch.__version__, "cuda": torch.version.cuda,
                            "platform": platform.platform()},
            "written_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    blob["npy_sha256"] = sha256_file(npy)
    blob["samples_sha256"] = blob["npy_sha256"]
    jsn.write_text(json.dumps(blob, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"[vgr] {stem}  n={x.shape[0]}  {wall/60:.1f} min  sha={blob['npy_sha256'][:16]}…")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
