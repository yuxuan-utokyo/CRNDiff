# -*- coding: utf-8 -*-
"""Sum the wall-clock time, deduplicating by entry and taking the MAXIMUM per entry.

This rule was learned the hard way. The queue log is append-only, so after a restart it holds
both starts, and the second one records an already-finished entry as a zero-second skip. Taking
the last record would erase the real machine time. So:

* the STATUS of an entry is its last attempt, which is the final outcome;
* the TIME of an entry is the maximum over all its attempts, which is what was really spent;
* failed attempts count towards the time, because they occupied the device just the same.

Runs whose wall clock is an outlier within one arm and configuration are flagged. An outlier
changes no number, but it has a cause worth naming rather than averaging away.

    python scripts/collect_wallclock.py
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from crndiff import config as C                                          # noqa: E402
from crndiff.io import load_runs                                         # noqa: E402

OUTLIER_RATIO = 1.5      # flagged if walltime > 1.5x the arm's median; a flag, not a filter


def main() -> int:
    stages, total = {}, 0.0
    for p in sorted((C.OUT / "queue").glob("*.jsonl")):
        attempts: dict[int, list[dict]] = {}
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            attempts.setdefault(int(r["i"]), []).append(r)
        rows, secs = [], 0.0
        for i in sorted(attempts):
            tries = attempts[i]
            last = tries[-1]
            burned = max(float(t.get("walltime_s") or 0.0) for t in tries)
            secs += burned
            rows.append({"i": i, "stem": last["stem"], "final_rc": last["rc"],
                         "n_attempts": len(tries), "gpu_seconds_max_over_attempts": burned,
                         "final_attempt_walltime_s": last.get("walltime_s"),
                         "attempt_rcs": [t["rc"] for t in tries]})
        ok = sum(1 for r in rows if r["final_rc"] == 0)
        stages[p.stem] = {"entries": len(rows), "final_ok": ok,
                          "final_failed": len(rows) - ok,
                          "gpu_seconds": secs, "gpu_minutes": secs / 60.0, "rows": rows}
        total += secs
        print(f"[stage] {p.stem:<20} entries {len(rows):3d}  ok {ok:3d}  "
              f"failed {len(rows) - ok:3d}  {secs / 60:8.1f} min")

    # ---- per-arm dispersion, from the run jsons (the queue jsonl has no arm) ----------
    flags = []
    by_arm: dict[tuple, list[dict]] = {}
    for exp_dir in sorted((C.OUT / "runs").glob("*")):
        if not exp_dir.is_dir():
            continue
        for r in load_runs(exp_dir.name):
            key = (exp_dir.name, r.get("arm"))
            by_arm.setdefault(key, []).append(r)
    arms = {}
    for (exp, arm), rs in sorted(by_arm.items()):
        w = [float(r["walltime_s"]) for r in rs if r.get("walltime_s")]
        if not w:
            continue
        med = statistics.median(w)
        entry = {"experiment": exp, "arm": arm, "n": len(w), "min_s": min(w),
                 "median_s": med, "max_s": max(w),
                 "seeds": {int(r["seed"]): float(r["walltime_s"]) for r in rs
                           if r.get("walltime_s")}}
        for r in rs:
            ws = float(r.get("walltime_s") or 0.0)
            if med > 0 and ws > OUTLIER_RATIO * med:
                flags.append({"experiment": exp, "arm": arm, "seed": r.get("seed"),
                              "walltime_s": ws, "arm_median_s": med,
                              "ratio": ws / med, "written_at": r.get("written_at"),
                              "note": "walltime outlier for an identical-config arm; results "
                                      "are seed+code determined, so this is environmental "
                                      "(GPU shared with another process, or OS activity). "
                                      "Flagged, not averaged away."})
        arms[f"{exp}/{arm}"] = entry
        print(f"[arm]   {exp}/{str(arm):<24} n={len(w)}  min {min(w):7.1f}  med {med:7.1f}  "
              f"max {max(w):7.1f} s")
    for f in flags:
        print(f"[flag]  {f['experiment']}/{f['arm']} seed {f['seed']}: {f['walltime_s']:.1f} s "
              f"= {f['ratio']:.2f}x the arm median {f['arm_median_s']:.1f} s ({f['written_at']})")

    blob = {"rule": "dedupe by (manifest, i); status = last attempt, GPU seconds = MAX over "
                    "attempts; failed attempts count toward machine time",
            "total_gpu_seconds": total, "total_gpu_hours": total / 3600.0,
            "stages": stages, "per_arm": arms, "walltime_outliers": flags,
            "outlier_ratio": OUTLIER_RATIO}
    dest = C.OUT / "summary" / "wallclock.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(blob, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"[total] {total / 3600:.2f} GPU hours over {len(stages)} manifests")
    print(f"[out]   {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
