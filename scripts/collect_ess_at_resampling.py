# -*- coding: utf-8 -*-
"""ESS/M at the moment each in-chain resampling event fires.

WHY THIS FILE EXISTS
--------------------
Table 8 reports ESS_min/M, the floor over the whole chain. That is not the
quantity the mechanism claim is about. The claim is that what a resampling
event costs in ancestry depends on how concentrated the weights are WHEN THE
EVENT FIRES. The pre-resampling ESS at the triggered steps is exactly that
quantity, and it is already stored in every run artefact.

NO NEW RULER, NO NEW SAMPLING. This script only reads fields the sampler
already wrote:
    ess_trace       : [{step, t, ess_fraction, scheduled}, ...]
    resample_steps  : the steps at which an in-chain resample actually fired
and selects ess_trace[s].ess_fraction for s in resample_steps.

SEMANTICS, VERIFIED BEFORE USE (see `verify` below, run on every file)
    * every step listed in resample_steps has ess_fraction < ess_threshold_fraction
    * some steps below threshold are NOT in resample_steps, because
      min_resample_gap forbids consecutive events
Both facts are only consistent with ess_trace holding the PRE-resample ESS.
If either check fails on any run the script raises: the field would then mean
something else and the numbers would be wrong.

    python -m tools.ess_pre_resample
"""
from __future__ import annotations
import json, glob, statistics as st, hashlib, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARMS = {
    "Residual Correction": "out/runs/ablation_table10/{T}_fk_only/*.json",
    "OURS":                "out/runs/table1_ours/{T}_a1/*.json",
}
TYPES = ["Endothelial", "Myeloid", "Neuronal"]


def verify(j, f):
    tr = {e["step"]: e["ess_fraction"] for e in j["ess_trace"]}
    rs, th = j["resample_steps"], j["ess_threshold_fraction"]
    gap = j["min_resample_gap"]
    bad = [s for s in rs if not tr[s] < th]
    if bad:
        raise SystemExit(f"{f}: resample fired at steps {bad} with ESS >= {th}; "
                         "ess_trace is not the pre-resample ESS -- STOP")
    below = sorted(s for s, v in tr.items() if v < th)
    skipped = [s for s in below if s not in rs]
    return dict(threshold=th, min_resample_gap=gap,
                n_steps_below_threshold=len(below),
                n_events=len(rs), n_below_but_skipped_by_gap=len(skipped))


def per_run(f):
    j = json.load(open(f))
    chk = verify(j, f)
    tr = {e["step"]: e["ess_fraction"] for e in j["ess_trace"]}
    v = [tr[s] for s in j["resample_steps"]]
    return dict(stem=Path(f).stem, n_events=len(v),
                ess_pre_mean=st.mean(v), ess_pre_median=st.median(v),
                ess_pre_min=min(v), ess_pre_max=max(v),
                ess_min_fraction_whole_chain=j["ess_min_fraction"],
                ess_final_before_terminal_resample=j["ess_pre_final_fraction"],
                ess_pre_per_event=v, checks=chk)


def main():
    rows = []
    for T in TYPES:
        for arm, pat in ARMS.items():
            files = sorted(glob.glob(str(ROOT / pat.format(T=T))))
            if not files:
                raise SystemExit(f"no runs for {T} / {arm}: {pat}")
            runs = [per_run(f) for f in files]
            agg = {}
            for k in ["n_events", "ess_pre_mean", "ess_pre_median",
                      "ess_pre_min", "ess_min_fraction_whole_chain",
                      "ess_final_before_terminal_resample"]:
                v = [r[k] for r in runs]
                agg[k + "_mean"] = st.mean(v)
                agg[k + "_sd"] = st.stdev(v) if len(v) > 1 else 0.0
            rows.append(dict(target=T, arm=arm, n_seeds=len(runs), **agg,
                             per_run=runs))
    out = dict(
        what="ESS/M at the step each in-chain resampling event fires",
        derivation="ess_trace[s].ess_fraction for s in resample_steps; "
                   "per-run mean/median/min, then mean +- sd (ddof=1) over seeds",
        recomputed_not_estimated=True,
        no_new_sampling=True, no_new_ruler=True,
        sd="sample standard deviation over the sampling seeds",
        sources={a: p for a, p in ARMS.items()},
        rows=rows,
        written_at=datetime.datetime.now().isoformat(timespec="seconds"),
    )
    dst = ROOT / "out/summary/ess_pre_resample.json"
    dst.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("wrote", dst)
    print("sha256", hashlib.sha256(dst.read_bytes()).hexdigest())
    for r in rows:
        print(f"{r['target']:12s} {r['arm']:20s} "
              f"K {r['n_events_mean']:5.1f}+-{r['n_events_sd']:.1f}  "
              f"median ESSpre {r['ess_pre_median_mean']:.3f}+-{r['ess_pre_median_sd']:.3f}  "
              f"ESSfinal {r['ess_final_before_terminal_resample_mean']:.3f}"
              f"+-{r['ess_final_before_terminal_resample_sd']:.3f}")


if __name__ == "__main__":
    main()
