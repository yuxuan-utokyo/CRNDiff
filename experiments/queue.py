# -*- coding: utf-8 -*-
"""Manifest-driven queue: one process per run, never two at once, resumable.
Purpose: run a manifest one entry at a time in a subprocess. It resumes, skipping an entry only
when its provenance matches, reporting a conflict rather than overwriting when it does not, and
refusing to spawn at all when a stem and its knobs disagree, which is a manifest error.

    python -m experiments.queue out/manifests/gates.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from crndiff.config import OUT, RUNS                    # noqa: E402
from crndiff.io import run_stem                         # noqa: E402

CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def argv_for(r: dict) -> list[str]:
    sampler = r.get("sampler", "twist")
    cmd = [sys.executable, "-m", "experiments.run_one",
           "--experiment", r["experiment"], "--arm", r["arm"],
           "--role", r.get("role", "experiment"),
           "--request", r["request"], "--seed", str(r["seed"]),
           "--sampler", sampler,
           "--K", str(r["K"]),
           # ticket 170: for a multi-component row `tau` in the manifest is the ENGINE
           # temperature (1.0, because the components are already tempered), while the CLI
           # wants the COMPONENT temperature. run_one then cross-checks it against the
           # request table's own note and refuses a mismatch.
           "--tau", str(r.get("component_tau", r["tau"])),
           "--tilt-mode", r.get("tilt_mode", "every"),
           "--n-particles", str(r["n_particles"]), "--n-out", str(r["n_out"]),
           "--cell-chunk", str(r["cell_chunk"])]
    if sampler != "tilt_only":
        # tilt_only evaluates no reward and never resamples: alpha / J / reward_* / the
        # trigger are not knobs of it, so they are not passed rather than passed as "None"
        cmd += ["--reward-mode", r["reward_mode"], "--reward-model", r["reward_model"],
                "--alpha", str(r["alpha"]), "--J", str(r["J"]),
                "--trigger", r.get("trigger", "scheduled"),
                "--ess-frac", str(r["ess_frac"]),
                "--resample-interval", str(r["resample_interval"])]
    if not r.get("tilt_on", True):          # the residual_only ablation: no product tilt at all
        cmd += ["--tilt", "off"]
    if r.get("residual_dir"):
        cmd += ["--residual-dir", str(r["residual_dir"])]
    if r.get("logpi"):                      # an explicit tilt table (e.g. one still in the old tree)
        cmd += ["--logpi", str(r["logpi"])]
    if r.get("types"):                      # ticket 170: the multi-component Table 8 arms
        cmd += ["--types", ",".join(r["types"]),
                "--weights", ",".join(str(w) for w in r["weights"])]
        if r.get("stratify"):
            cmd += ["--stratify", ",".join(str(v) for v in r["stratify"])]
    if r.get("reward_target"):
        cmd += ["--reward-target", str(r["reward_target"])]
    if r.get("ticket") is not None:
        cmd += ["--ticket", str(r["ticket"])]
    if not r.get("intermediate_resampling", True):
        cmd.append("--no-intermediate-resampling")
    return cmd


def status_of(r: dict) -> str:
    if run_stem(r) != r.get("stem"):        # a stale hand-edited manifest must not spawn
        return "manifest_error"
    pj = RUNS / r["experiment"] / r["arm"] / f"{r['stem']}.json"
    if not pj.exists():
        return "todo"
    try:
        d = json.loads(pj.read_text(encoding="utf-8"))
    except Exception:
        return "collision"
    same = (d.get("experiment") == r.get("experiment")
            and d.get("role", "experiment") == r.get("role", "experiment")
            and d.get("ticket") == r.get("ticket"))
    return "done" if same else "collision"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifest")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--stop-on-error", action="store_true")
    a = ap.parse_args(argv)
    runs = json.loads(Path(a.manifest).read_text(encoding="utf-8"))
    logd = OUT / "queue"
    logd.mkdir(parents=True, exist_ok=True)
    log = logd / (Path(a.manifest).stem + ".jsonl")
    n_done = n_skip = n_fail = n_coll = n_manifest = 0
    started_all = time.time()
    for i, r in enumerate(runs):
        if a.limit is not None and n_done >= a.limit:
            break
        st = status_of(r)
        if st == "manifest_error":
            print(f"[manifest] {i}: stem does not match the knobs -- refusing to spawn: "
                  f"{r.get('stem')}")
            n_manifest += 1
            continue
        if st == "collision":
            print(f"[collision] {i}: {r['stem']} exists with a different provenance")
            n_coll += 1
            continue
        if st == "done":
            n_skip += 1
            continue
        cmd = argv_for(r)
        if a.dry_run:
            print(" ".join(cmd[1:]))
            n_done += 1
            continue
        t0 = time.time()
        rc = subprocess.run(cmd, cwd=str(ROOT), creationflags=CREATE_NO_WINDOW).returncode
        dt = time.time() - t0
        with log.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"i": i, "stem": r["stem"], "rc": rc, "walltime_s": dt,
                                 "at": time.strftime("%Y-%m-%d %H:%M:%S")}) + "\n")
        if rc == 0:
            n_done += 1
            print(f"[ok {dt:.0f}s] {r['stem']}")
        else:
            n_fail += 1
            print(f"[FAIL rc={rc}] {r['stem']}")
            if a.stop_on_error:
                break
    print(f"[queue] done={n_done} skipped={n_skip} failed={n_fail} collisions={n_coll} "
          f"manifest_errors={n_manifest} in {(time.time() - started_all) / 60:.1f} min -> {log}")
    return 1 if (n_fail or n_coll or n_manifest) else 0


if __name__ == "__main__":
    sys.exit(main())
