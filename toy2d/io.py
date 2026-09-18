# -*- coding: utf-8 -*-
"""Run naming and on-disk layout: out/runs/<experiment>/<arm>/<stem>.{json,npz}.
Purpose: how a run is named and written. Every knob that can change a number goes into the file
name; an existing file is refused rather than overwritten; a whole experiment can be read back.
Ported from the 10-dimensional toy package, with the 2D request and sampler arm added.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .config import DEFAULT_REWARD_FLOOR, RUNS


def fmt(v) -> str:
    return str(v).replace(".", "p").replace("-", "m")


def model_tag(model_stem: str) -> str:
    """Deterministic short checkpoint identifier used inside run stems (first two tokens plus
    the trailing training-seed token); the FULL stem is pinned in the manifest, forwarded via
    --model and recorded in every run json. Keeps the longest path under Windows MAX_PATH."""
    parts = str(model_stem).split("_")
    return "_".join(parts[:2] + parts[-1:]) if len(parts) > 3 else str(model_stem)


def run_stem(a: dict) -> str:
    """Every knob that can move a number is in the stem, so two configurations can never share
    a file and a run refuses to overwrite an existing one."""
    sampler = a.get("sampler", "fk")
    parts = [a["experiment"], a["request"]]
    if sampler == "tilt_only":
        parts.append("tiltonly")
        parts.append("tilt" if a["tilt_on"] else "notilt")
        parts += [f"M{a['n_particles']}", f"N{a['n_out']}"]
    else:
        parts.append("tilt" if a["tilt_on"] else "notilt")
        parts += [f"J{a['J']}", f"a{fmt(a['alpha'])}",
                  f"th{fmt(a['ess_frac'])}", f"int{a['resample_interval']}",
                  f"M{a['n_particles']}", f"N{a['n_out']}"]
        if a.get("sigma_logit", 0.0) > 0:
            parts.append(f"sig{fmt(a['sigma_logit'])}")
        if not a.get("intermediate_resampling", True):
            parts.append("noresample")
        if a.get("reward_family", "proposal_residual") != "proposal_residual":
            parts.append(a["reward_family"].replace("proposal_", ""))   # short token, MAX_PATH headroom
        if a.get("reward_floor", DEFAULT_REWARD_FLOOR) != DEFAULT_REWARD_FLOOR:
            parts.append(f"rf{fmt(a['reward_floor'])}")
        parts.append(a["reward_mode"])
    parts.append(model_tag(a["model"]))
    parts.append(f"seed{a['seed']}")
    return "_".join(parts)


def paths_for(stem: str, experiment: str, arm: str) -> tuple[Path, Path]:
    d = RUNS / experiment / arm
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{stem}.json", d / f"{stem}.npz"


def write_run(stem: str, experiment: str, arm: str, x: np.ndarray, diag: dict,
              extra_arrays: dict | None = None) -> Path:
    pj, pn = paths_for(stem, experiment, arm)
    if pj.exists() or pn.exists():
        raise SystemExit(f"refusing to overwrite an existing run: {pj}")
    arrays = {"X": x.astype(np.uint8)}
    if extra_arrays:
        arrays.update(extra_arrays)
    np.savez_compressed(pn, **arrays)
    diag = dict(diag)
    diag["written_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with pj.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(diag, indent=1) + "\n")
    return pj


def load_runs(experiment: str) -> list[dict]:
    d = RUNS / experiment
    out = []
    for p in sorted(d.rglob("*.json")) if d.exists() else []:
        r = json.loads(p.read_text(encoding="utf-8"))
        r["_path"] = str(p)
        out.append(r)
    return out
