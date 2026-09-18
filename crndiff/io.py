# -*- coding: utf-8 -*-
"""Run naming and on-disk layout for HVG runs.
Purpose: how a run is named and written. Every knob that can change a number goes into the file
name; an existing file is refused rather than overwritten; a whole experiment can be read back.
Isomorphic to the toy package's io. The stem follows the frozen `fk_hvg_v2.main` with three
changes: `sc{intermediate_scale}` is gone, because the single-parameter form removed it; `J{J}`
is added (the number of twist candidate endpoints, present only on the twist arm); and
`reward_mode` is added (odds is the weight the paper defines, probability is the port-fidelity

    out/runs/<experiment>/<arm>/<stem>.{json,npy}
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from .config import RUNS

__all__ = ["fmt", "sha256_file", "sha256_array", "run_stem", "paths_for", "write_run",
           "load_runs", "celltypist_identity", "RECORDED_INPUTS"]


def fmt(v) -> str:
    return str(v).replace(".", "p").replace("-", "m")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_array(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def run_stem(a: dict) -> str:
    """Every knob that can move a number is in the stem, so two configurations can never share
    a file and a run refuses to overwrite an existing one."""
    sampler = a.get("sampler", "twist")
    if sampler == "tilt_only":
        # this arm evaluates no reward and never resamples, so alpha / J / theta / the trigger
        # are not knobs of it at all; putting them in the stem would invent distinctions
        parts = [a["request"], f"hvg{sampler}",
                 "tilt" if a.get("tilt_on", True) else "notilt",
                 f"K{a['K']}", f"M{a['n_particles']}", f"N{a['n_out']}",
                 f"tau{fmt(a['tau'])}", f"cc{a['cell_chunk']}", f"seed{a['seed']}"]
        return "_".join(parts)
    mode = ("endpoint_gate" if not a.get("intermediate_resampling", True)
            else "fk")
    parts = [a["request"], a["reward_model"], f"hvg{sampler}", mode,
             # ticket 170: the two Table 8 arms differ ONLY in the resampler, so that has to be
             # in the stem. The components and their weights are already pinned by `request`
             # (MIX0500_Endothelial_Myeloid), and run_one refuses a --weights that disagrees
             # with that table's own provenance.w_A, so no second pair can reach this name.
             *(("quota",) if a.get("stratified") else ()),
             # the components are tempered at THEIR OWN tau inside build_mixture_tilt and the
             # engine then runs at tau=1.0, so the stem's `tau` token is the engine's. The
             # component temperature is a knob that moves numbers, so it gets its own token.
             *((f"ctau{fmt(a['component_tau'])}",) if a.get("component_tau") is not None
               else ()),
             # tilt_on is a different knob from tilt_mode (WHETHER vs WHEN); it moves numbers,
             # so it is in the stem -- but only as a token when it is off, so every stem written
             # before this knob existed still reproduces exactly
             *(() if a.get("tilt_on", True) else ("notilt",)),
             f"tilt{a.get('tilt_mode', 'every')}", f"trig{a.get('trigger', 'scheduled')}",
             f"K{a['K']}", f"M{a['n_particles']}", f"N{a['n_out']}",
             f"alpha{fmt(a['alpha'])}"]
    if sampler == "twist":
        parts.append(f"J{a['J']}")
    parts += [f"int{a['resample_interval']}", f"th{fmt(a['ess_frac'])}",
              f"tau{fmt(a['tau'])}", f"cc{a['cell_chunk']}", a["reward_mode"],
              f"seed{a['seed']}"]
    return "_".join(parts)


def paths_for(stem: str, experiment: str, arm: str) -> tuple[Path, Path]:
    d = RUNS / experiment / arm
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{stem}.json", d / f"{stem}.npy"


# ---- the two recording debts ruling 164 section 4 closes -------------------------------------
# 1. the frozen `sample_log_*.json` never recorded WHICH logpi/tilt table a run used, so an
#    archived arm cannot be reproduced from its own json (it cost gate H0 one attempt).
# 2. the archived CellTypist jsons recorded only the model FILE NAME -- no version, no sha256 --
#    so a model upgrade would silently move the purity ruler.
# Every run json we write from now on carries all seven fields. Missing ones are recorded as
# None with a reason, never omitted.
RECORDED_INPUTS = ("logpi_path", "logpi_sha256", "tilt_table_path", "tilt_table_sha256",
                   "celltypist_model_sha256", "celltypist_version", "celltypist_model_date")


def celltypist_identity() -> dict:
    """The three fields that pin the purity ruler (gate G5). Never raises: a run must still be
    writable on a box without celltypist, but then the fields say so explicitly."""
    out = {"celltypist_model_sha256": None, "celltypist_version": None,
           "celltypist_model_date": None}
    try:
        import os
        import celltypist
        out["celltypist_version"] = str(celltypist.__version__)
        p = (Path(os.path.expanduser("~")) / ".celltypist" / "data" / "models"
             / "Healthy_Adult_Heart.pkl")
        if p.exists():
            out["celltypist_model_sha256"] = sha256_file(p)
            try:
                m = celltypist.models.Model.load(str(p))
                d = getattr(m, "description", {}) or {}
                out["celltypist_model_date"] = str(d.get("date"))
                out["celltypist_model_version"] = str(d.get("version"))
            except Exception as e:                       # loadable-but-unreadable description
                out["celltypist_model_date"] = f"unreadable: {e!r}"
        else:
            out["celltypist_model_sha256"] = f"model not found at {p}"
    except Exception as e:
        out["celltypist_version"] = f"celltypist unavailable: {e!r}"
    return out


def write_run(stem: str, experiment: str, arm: str, x: np.ndarray, diag: dict) -> Path:
    pj, pn = paths_for(stem, experiment, arm)
    if pj.exists() or pn.exists():
        raise SystemExit(f"refusing to overwrite an existing run: {pj}")
    missing = [k for k in RECORDED_INPUTS if k not in diag]
    if missing:
        raise SystemExit(f"run json is missing the recorded-input fields {missing} "
                         "(ruling 164 section 4); every run must carry all seven")
    cells = np.asarray(x, dtype=np.int16)
    np.save(pn, cells)
    diag = dict(diag)
    diag["samples_sha256"] = sha256_array(cells)
    diag["samples_file"] = pn.name
    diag["written_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with pj.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(diag, ensure_ascii=False, indent=1) + "\n")
    return pj


def load_runs(experiment: str) -> list[dict]:
    d = RUNS / experiment
    out = []
    for p in sorted(d.rglob("*.json")) if d.exists() else []:
        r = json.loads(p.read_text(encoding="utf-8"))
        r["_path"] = str(p)
        out.append(r)
    return out
