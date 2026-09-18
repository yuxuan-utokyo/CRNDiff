# -*- coding: utf-8 -*-
"""A thin wrapper that adds multiple evaluation seeds and several top-K values.

No statistics are computed here. `block_a`, `block_b`, `logfc`, `top_overlap` and
`load_manifest` are all imported from the unmodified archived module, whose sha256 is asserted
`dee8e40d722fe0acbd6c922ad21421bde5e9941101a7375346bccae3654a8007`).
below. This file loops over seeds, varies K, and aggregates into mean and standard deviation.

Two mechanical rewirings, neither of which touches a number:

1. the data constant pointed at the retired tree and is repointed at
      `assets/legacy_root/_shared/data/all`, which is the same tree and the same bytes;
2. the output constant pointed at the ARCHIVED directory, where the module's own `main()`
      would overwrite an archived artefact, so it is repointed into this repository's output
      tree. The archived file itself is untouched: only module constants are overwritten after
      import, and its sha256 is rechecked afterwards.

# ### How top-K gets the log fold changes without copying `block_b`

`block_b` does not return the fold-change vectors, only the correlation and the top-50 overlap.
But it calls `top_overlap` through a MODULE GLOBAL, so that global is replaced by an interceptor
that records the arguments, calls the original function and returns its value unchanged:

    * the return value is bitwise unchanged, since it is the original function;
    * no random number is consumed;
    * `block_b` is not edited and no statistics are copied.

With the vectors in hand, each K is computed by that same original `top_overlap`.
The mapping from call order to (type, arm) is not guessed: the interceptor records the order and
it is checked afterwards against what `block_b` returned; a mismatch marks the K sweep unusable.

    python -m analysis.e33_multiseed --mode body      #     python -m evaluation.downstream_multiseed --mode body      # the unmodified module
    python -m analysis.e33_multiseed --mode wrapper   #     python -m evaluation.downstream_multiseed --mode wrapper   # the wrapper, one seed
    python -m analysis.e33_multiseed --mode full      #     python -m evaluation.downstream_multiseed --mode full      # five seeds and three K values
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from crndiff import config as C                                      # noqa: E402

SRC = (C.ARCHIVE
       / "universality_fk" / "code" / "e33_downstream.py")
SRC_SHA = "dee8e40d722fe0acbd6c922ad21421bde5e9941101a7375346bccae3654a8007"
ARCHIVED = SRC.parent.parent / "out" / "e33" / "E33_downstream.json"
PREREG = SRC.parent.parent / "out" / "e33" / "E33_prereg.json"
WORK = C.OUT / "e33_184"
SUMMARY = C.OUT / "summary"

SEEDS = [20260833, 20260834, 20260835, 20260836, 20260837]   # registered in advance; not to be added to or removed from
KS = [50, 100, 200]                                          # registered in advance; not to be chosen after seeing the numbers


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_body(out_dir: Path):
    """Import the unmodified module and overwrite only the two path constants that are stale or would write into the archive."""
    got = sha256_file(SRC)
    if got != SRC_SHA:
        raise SystemExit(f"e33_downstream.py changed: {got} != {SRC_SHA} -- STOP")
    spec = importlib.util.spec_from_file_location("_e33_body", SRC)
    m = importlib.util.module_from_spec(spec)
    sys.modules["_e33_body"] = m
    spec.loader.exec_module(m)
    old_data, old_out = m.DATA, m.OUT
    m.DATA = C.LEGACY_DATA / "all"
    out_dir.mkdir(parents=True, exist_ok=True)
    m.OUT = out_dir
    if not (out_dir / "E33_prereg.json").exists():
        shutil.copy2(PREREG, out_dir / "E33_prereg.json")
    return m, {"source": str(SRC), "sha256": got,
               "DATA_was": str(old_data), "DATA_now": str(m.DATA),
               "DATA_reason": "the Experiments/X tree was retired into assets/legacy_root",
               "OUT_was": str(old_out), "OUT_now": str(m.OUT),
               "OUT_reason": "the body's main() writes E33_downstream.json into OUT, which "
                             "is the ARCHIVE directory; redirected so the archive is never "
                             "overwritten",
               "file_edited": False}


class TopOverlapRecorder:
    """Record the arguments of `top_overlap`; the return value comes from the original function."""

    def __init__(self, original):
        self.original = original
        self.calls = []

    def __call__(self, u, v, k=None):
        out = self.original(u, v) if k is None else self.original(u, v, k)
        self.calls.append({"u": np.asarray(u), "v": np.asarray(v), "returned": out})
        return out


def measured_pairs(block_b_res, m):
    """The (type, arm) order `block_b` actually measures: types outermost, arms innermost, skipping missing ones."""
    out = []
    for t in m.TYPES:
        for arm in m.ARMS:
            e = block_b_res["types"].get(t, {}).get("arms", {}).get(arm, {})
            if "top50_overlap_fraction" in e:
                out.append((t, arm))
    return out


def run_block_b_with_k(m, by, seed):
    """Run `block_b` once, capture the fold-change vectors, then recompute the overlap for each K."""
    rec = TopOverlapRecorder(m.top_overlap)
    original = m.top_overlap
    m.top_overlap = rec
    try:
        res = m.block_b(by, np.random.default_rng(seed))
    finally:
        m.top_overlap = original
    pairs = measured_pairs(res, m)
    ok = len(pairs) == len(rec.calls)
    if ok:
        for (t, arm), call in zip(pairs, rec.calls):
            if res["types"][t]["arms"][arm]["top50_overlap_fraction"] != call["returned"]:
                ok = False
                break
    topk, fc_stats = {}, {}
    if ok:
        for (t, arm), call in zip(pairs, rec.calls):
            topk.setdefault(t, {})[arm] = {
                f"top{k}_overlap_fraction": float(original(call["u"], call["v"], k))
                for k in KS}
        # the magnitude distribution of the real fold changes, recorded once per type
        for (t, _arm), call in zip(pairs, rec.calls):
            if t in fc_stats:
                continue
            a = np.abs(np.asarray(call["u"], dtype=np.float64))
            fc_stats[t] = {
                "n_genes": int(a.size),
                "abs_logfc_quantiles": {q: float(np.quantile(a, float(q)))
                                        for q in ("0.1", "0.25", "0.5", "0.75", "0.9")},
                "frac_abs_logfc_below_0.1": float((a < 0.1).mean()),
                "frac_abs_logfc_below_0.25": float((a < 0.25).mean()),
                "frac_abs_logfc_below_0.5": float((a < 0.5).mean())}
    return res, topk, fc_stats, {
        "n_top_overlap_calls": len(rec.calls), "n_measured_pairs": len(pairs),
        "mapping_verified": ok,
        "how": "the recorder's returned value is compared with the top50_overlap_fraction "
               "block_b itself wrote, pair by pair; a mismatch disables the K sweep and "
               "leaves B.1 untouched"}


def mean_sd(v):
    a = np.asarray(v, dtype=np.float64)
    return {"mean": float(a.mean()),
            "sd": float(a.std(ddof=1)) if a.size > 1 else None,
            "n": int(a.size), "per_seed": [float(x) for x in a]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("body", "wrapper", "full"), default="full")
    a = ap.parse_args(argv)
    WORK.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- body --
    if a.mode == "body":
        m, prov = load_body(WORK / "body")
        t0 = time.time()
        m.main()                       # the module's own entry point, unmodified
        prov["walltime_s"] = time.time() - t0
        prov["product"] = str(WORK / "body" / "E33_downstream.json")
        prov["sha256_after_run"] = sha256_file(SRC)
        (WORK / "body_provenance.json").write_text(
            json.dumps(prov, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"[body] -> {prov['product']}")
        print(f"[body] e33_downstream.py sha256 after the run: {prov['sha256_after_run']}")
        return 0

    seeds = [SEEDS[0]] if a.mode == "wrapper" else SEEDS
    m, prov = load_body(WORK / "wrapper")
    by = m.load_manifest()

    per_seed_a, per_seed_b, per_seed_topk, fc_stats, map_checks = {}, {}, {}, {}, []
    for s in seeds:
        print(f"\n=== seed {s} : block_a ===", flush=True)
        per_seed_a[s] = m.block_a(by, np.random.default_rng(s))
        print(f"=== seed {s} : block_b ===", flush=True)
        rb, tk, fcs, chk = run_block_b_with_k(m, by, s)
        per_seed_b[s], per_seed_topk[s] = rb, tk
        map_checks.append({"seed": s, **chk})
        if not fc_stats:
            fc_stats = fcs

    blob = {"produced_by": "analysis/e33_multiseed.py", "mode": a.mode,
            "body": prov, "seeds": seeds, "K_registered": KS,
            "wrapper_sha256": sha256_file(Path(__file__)),
            "no_statistics_here": "block_a / block_b / logfc / top_overlap / load_manifest "
                                  "are imported from the untouched body; this file only "
                                  "loops seeds and K and aggregates",
            "classifier_random_state_note":
                "block_a's LogisticRegression uses random_state=SEED, the body's module "
                "constant, NOT the loop seed. It is a frozen hyperparameter and lbfgs is "
                "deterministic, so it does not affect the numbers; left untouched.",
            "top_overlap_interception": map_checks,
            "per_seed_block_a": {str(k): v for k, v in per_seed_a.items()},
            "per_seed_block_b": {str(k): v for k, v in per_seed_b.items()},
            "per_seed_topk": {str(k): v for k, v in per_seed_topk.items()},
            "real_logfc_stats": fc_stats,
            "written_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    p = WORK / f"raw_{a.mode}.json"
    p.write_text(json.dumps(blob, ensure_ascii=False, indent=1, default=float) + "\n",
                 encoding="utf-8")
    print(f"\n[out] {p}")
    print(f"[out] sha256 {sha256_file(p)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
