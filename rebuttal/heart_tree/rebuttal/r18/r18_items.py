# -*- coding: utf-8 -*-
"""R18: same-machine runtime re-measurement on HVG2K (one wrapper per item, one item per process).

    cd Experiments\\HVG
    python -u rebuttal\\r18\\r18_items.py --item gen_train --run-dir rebuttal\\out\\R18\\run_<stamp>

Rules this file keeps (reviewer, R18):
  * No source file is edited. Every cap / path override is applied HERE, in this process, after import
    (module-constant override, monkeypatched scheduler `.step`, Lightning callback, or a copy of code
    into the R18 shadow dir). Frozen trees are only ever opened for reading.
  * Every output goes under <run-dir>/items/<item>/ (inside rebuttal/out/R18/). `_inside()` asserts it.
  * Short timed runs, extrapolated: training = warm-up 20 steps (untimed) + 200 timed steps, the
    scheduler horizon is left at the REAL total so per-step cost (incl. the 1-in-200 evaluation and its
    checkpoint write) is the deployed one; scVI/scANVI = 1 warm-up epoch + 2 timed epochs.
    Sampling = a small warm-up call, then 512 Neuronal cells timed.
  * CPU threads pinned to R18_THREADS = 4 (env set before numpy/torch import + torch.set_num_threads(4)), recorded.
  * Each item prints ONE line `R18_RESULT {json}` at the end and writes <run-dir>/results/<item>.json.
    A failure is recorded as status "failed" (the report then shows "not re-measured").
"""
from __future__ import annotations

import os
import sys

R18_THREADS = 4                             # vice 11-10: one fixed count for every item (tonight's cap 8), set before import
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = str(R18_THREADS)
os.environ.setdefault("PYTHONUTF8", "1")
sys.dont_write_bytecode = True              # no __pycache__ may appear in any frozen code dir

import argparse                             # noqa: E402
import importlib.util                       # noqa: E402
import json                                 # noqa: E402
import platform                             # noqa: E402
import shutil                               # noqa: E402
import time                                 # noqa: E402
import traceback                            # noqa: E402
from pathlib import Path                    # noqa: E402

HERE = Path(__file__).resolve().parent                       # HVG/rebuttal/r18
HVG = HERE.parents[1]                                        # Experiments/HVG
CRN = HVG.parents[1]                                         # CRNDIFF
CGE = CRN / "Conditional generation experiments" / "HVG2K"
DATA = HVG / "assets" / "legacy_root" / "_shared" / "data" / "all"
BASE_CODE = HVG / "assets" / "legacy_root" / "Baseline" / "OURS" / "code"
R18_ROOT = HVG / "rebuttal" / "out" / "R18"

WARM_STEPS, TIMED_STEPS = 20, 200
WARM_EPOCHS, TIMED_EPOCHS = 1, 2
N_CELLS = 512
TARGET = "Neuronal"

# Real totals used for the extrapolation (value, source). Re-read and cross-checked by r18_report.py.
REAL = {
    "gen_train": (20000, "assets/legacy_root/Baseline/OURS/models/pilot_run214_s20260626.json::steps_requested"),
    "cond_train": (40985, "assets/models/conditional/aprime_v2_cond_s20260817.json::steps_done"),
    "cd_train": (20000, "rebuttal/out/R12/hvg2k/s20260625.json::finished_steps"),
    "scvi": (22, "out/baseline_214/scvi_s20261050/logs/train_scvi.json::actual_epochs"),
    "scanvi": (22, "out/baseline_214/scanvi_s20261060/logs/train_scanvi.json::actual_epochs"),
    "cfgen_train": ({"encoder": 24112, "fm": 29187},
                    "CGE/baseline_cfgen/out/cfgen_train_full.json::{encoder,fm}.global_step"),
    "mdlm_train": ({"s20260934": 4400, "s20260935": 5800},
                   "out/mdlm_cond_216/mdlm_cond_s2026093{4,5}/out/summary/207_m2_mdlm_cond_training.json::steps"),
}


# ----------------------------------------------------------------------------------------- helpers
class StopTiming(Exception):
    """Raised from inside the training loop once the timed window is complete."""


def _inside(p: Path, root: Path) -> Path:
    p = Path(p).resolve()
    if not str(p).lower().startswith(str(Path(root).resolve()).lower()):
        raise SystemExit(f"[R18] refusing: output path {p} is not under {root}")
    return p


def load_module(path: Path, name: str):
    """Import a file by path (read-only; bytecode writing is off)."""
    spec = importlib.util.spec_from_file_location(name, str(path))
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def _torch():
    import torch
    torch.set_num_threads(R18_THREADS)
    return torch


def _sync():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


class StepTimer:
    """tick() is called once at the END of every optimiser step. Window = end of step WARM .. end of step
    WARM+TIMED, i.e. exactly TIMED steps (with whatever evaluation / checkpointing falls inside them)."""

    def __init__(self, warm=WARM_STEPS, timed=TIMED_STEPS, raise_at_end=True, name=""):
        self.warm, self.timed, self.raise_at_end, self.name = warm, timed, raise_at_end, name
        self.perf, self.wall = [], []
        self.done = False

    def tick(self):
        _sync()
        self.perf.append(time.perf_counter()); self.wall.append(time.time())
        n = len(self.perf)
        if n == self.warm + self.timed:
            self.done = True
            print(f"[R18] {self.name} timing window complete: {self.summary()['seconds']:.2f} s "
                  f"for {self.timed} steps", flush=True)
            if self.raise_at_end:
                raise StopTiming(self.name)

    def summary(self) -> dict:
        if not self.done:
            return {"done": False, "ticks": len(self.perf)}
        a, b = self.warm - 1, self.warm + self.timed - 1
        d = [self.perf[i + 1] - self.perf[i] for i in range(a, b)]
        ds = sorted(d)
        return {"done": True, "seconds": self.perf[b] - self.perf[a], "timed_steps": self.timed,
                "s_per_step": (self.perf[b] - self.perf[a]) / self.timed,
                "median_step_s": ds[len(ds) // 2], "p90_step_s": ds[int(0.9 * (len(ds) - 1))],
                "max_step_s": ds[-1], "window_wall": [self.wall[a], self.wall[b]]}


_TIMERS: dict = {}
_TIMED_CLS: dict = {}


def _timed_step(self, *a, **k):
    r = _TIMED_CLS[type(self)].step(self, *a, **k)        # the original class's step
    _TIMERS[id(self)].tick()
    return r


def wrap_sched_step(sched, timer: StepTimer):
    """v2 (pass 2, vice 07:5x): swap the instance's class for a subclass whose step() calls the original step and ticks the timer.
    v1 set an instance attribute `step` (a closure); LambdaLR.state_dict() copies __dict__, so CountsDiff's checkpoint torch.save failed
    with "Can't pickle local object". Now __dict__ is unchanged; the source and the original class are untouched."""
    base = type(sched)
    cls = next((c for c, b in _TIMED_CLS.items() if b is base), None)
    if cls is None:
        cls = type("R18Timed" + base.__name__, (base,), {"step": _timed_step})
        _TIMED_CLS[cls] = base
    _TIMERS[id(sched)] = timer
    sched.__class__ = cls
    return sched


def env_record() -> dict:
    rec = {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                          "NUMEXPR_NUM_THREADS", "PYTORCH_CUDA_ALLOC_CONF", "HVG_OUT",
                                          "CUDA_VISIBLE_DEVICES")}
    rec["python"] = sys.version.split()[0]
    rec["platform"] = platform.platform()
    try:
        import torch
        rec["torch_threads"] = torch.get_num_threads()
        rec["torch"] = torch.__version__
        if torch.cuda.is_available():
            rec["gpu"] = torch.cuda.get_device_name(0)
    except Exception as e:                                      # CPU-only item without torch
        rec["torch_threads"] = f"n/a ({e!r})"
    return rec


def peak_gb():
    try:
        import torch
        if torch.cuda.is_available():
            return float(torch.cuda.max_memory_allocated() / 2 ** 30)
    except Exception:
        pass
    return None


def reset_peak():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def with_argv(argv, fn):
    old = sys.argv
    sys.argv = list(argv)
    try:
        return fn()
    finally:
        sys.argv = old


def train_result(item, method, timer: StepTimer, batch, real_total, real_src, extra=None):
    s = timer.summary()
    if not s.get("done"):
        raise RuntimeError(f"timing window not reached ({s.get('ticks')} steps ticked)")
    r = {"item": item, "method": method, "phase": "train", "status": "ok", "unit": "step",
         "timed_steps": s["timed_steps"], "seconds": s["seconds"], "s_per_step": s["s_per_step"],
         "median_step_s": s["median_step_s"], "p90_step_s": s["p90_step_s"], "max_step_s": s["max_step_s"],
         "warmup": f"{timer.warm} steps (untimed)", "batch": batch, "window_wall": s["window_wall"],
         "real_total": real_total, "real_total_source": real_src, "extrapolated": True}
    if isinstance(real_total, (int, float)):
        r["extrapolated_total_s"] = s["s_per_step"] * real_total
    if extra:
        r.update(extra)
    return r


def sample_result(item, method, seconds, n_cells, window_wall, warmup, batch, extra=None):
    r = {"item": item, "method": method, "phase": "sample", "status": "ok", "unit": "cells",
         "n_cells": n_cells, "seconds": seconds, "s_per_1k": seconds / n_cells * 1000.0,
         "extrapolated_2500_s": seconds / n_cells * 2500.0, "warmup": warmup, "batch": batch,
         "window_wall": window_wall, "extrapolated": True}
    if extra:
        r.update(extra)
    return r


# -------------------------------------------------------------------------------- 1 generator train
def item_gen_train(d: Path):
    """CRNDiff generator training: frozen train_pilot.main(), ROOT/LOG redirected after import."""
    torch = _torch()
    sys.path.insert(0, str(BASE_CODE))
    TP = load_module(BASE_CODE / "train_pilot.py", "r18_train_pilot")
    root = _inside(d / "shadow_root", d)
    for sub in ("models", "loss"):
        (root / "Baseline" / "OURS" / sub).mkdir(parents=True, exist_ok=True)
    TP.ROOT, TP.LOG = root, root / "RUN_LOG.md"          # ckpt_path / loss / json / RUN_LOG -> shadow
    timer = StepTimer(name="gen_train")
    BB = TP.M.BB
    orig = BB._cosine_sched
    BB._cosine_sched = lambda opt, steps, warmup: wrap_sched_step(orig(opt, steps, warmup), timer)
    argv = ["train_pilot.py", "--steps", "20000", "--batch", "384", "--name", "r18_gen", "--seed", "20260626"]
    reset_peak()
    try:
        rc = with_argv(argv, TP.main)
        raise RuntimeError(f"train_pilot.main returned rc={rc} before the timing window")
    except StopTiming:
        pass
    n, src = REAL["gen_train"]
    return train_result("gen_train", "CRNDiff generator (pilot_run214 recipe)", timer, 384, n, src,
                        {"peak_mem_gb": peak_gb(), "argv": argv,
                         "patch": "train_pilot.ROOT/LOG -> shadow; backbone._cosine_sched wrapped (steps=20000 kept)",
                         "notes": "bf16 + grad ckpt from config_hvg2k; seed 20260626 = pilot_run214_s20260626"})


# ------------------------------------------------------------------------------------- 2 tilted FK
def _replace_opt(argv, flag, val):
    a = list(argv)
    i = a.index(flag)
    a[i + 1] = str(val)
    return a


def item_fk_sample(d: Path):
    """Tilted FK, Neuronal_a1 deployed settings, via experiments.run_one with HVG_OUT -> R18."""
    # v2 (pass 2, vice 07:5x): pass 1 hit Windows MAX_PATH (260) under <run>\items\fk_sample\hvg_out\runs\...; use a short root
    # outside every frozen root instead: work\r18o\<run dir name>\fk (README there points back to the run dir).
    out = Path("work/r18o") / d.parent.parent.name / "fk"; out.mkdir(parents=True, exist_ok=True)
    (out.parent / "README.txt").write_text(f"R18 fk_sample HVG_OUT (short path for MAX_PATH); run dir {d.parent.parent}\n", encoding="utf-8")
    os.environ["HVG_OUT"] = str(out)                  # must precede `import hvg.config`
    torch = _torch()
    src = sorted((HVG / "out" / "runs" / "table1_ours" / "Neuronal_a1").glob("*seed20260987.json"))[0]
    rerun = json.loads(src.read_text(encoding="utf-8"))["_rerun_argv"]
    base = rerun[2:] if rerun[:2] == ["-m", "experiments.run_one"] else rerun
    base = _replace_opt(base, "--experiment", "rebuttal_R18")
    sys.path.insert(0, str(HVG))
    from experiments import run_one as RO                                    # noqa: E402
    from hvg import config as C                                              # noqa: E402
    assert Path(C.OUT).resolve() == out.resolve(), (C.OUT, out)
    warm = _replace_opt(_replace_opt(_replace_opt(base, "--arm", "Neuronal_a1_R18warm"),
                                     "--n-particles", 128), "--n-out", 64)
    timed = _replace_opt(_replace_opt(_replace_opt(base, "--arm", "Neuronal_a1_R18"),
                                      "--n-particles", 2 * N_CELLS), "--n-out", N_CELLS)
    rc = RO.main(warm)
    if rc != 0:
        raise RuntimeError(f"warm-up run_one rc={rc}")
    w0 = time.time(); _sync(); t0 = time.perf_counter()
    rc = RO.main(timed)
    _sync(); dt = time.perf_counter() - t0; w1 = time.time()
    if rc != 0:
        raise RuntimeError(f"timed run_one rc={rc}")
    js = sorted((out / "runs" / "rebuttal_R18" / "Neuronal_a1_R18").glob("*.json"))
    if len(js) != 1:
        raise RuntimeError(f"expected one run json, found {js}")
    j = json.loads(js[0].read_text(encoding="utf-8"))
    wt = float(j["walltime_s"])            # bank build + FK sampling: same definition as Neuronal_a1 walltime_s
    return sample_result("fk_sample", "CRNDiff tilted FK (Neuronal_a1 settings)", wt, N_CELLS, [w0, w1],
                         "run_one M=128/N=64 (same process, untimed)", "cell_chunk 256",
                         {"process_seconds_incl_load": dt, "M_particles": 2 * N_CELLS,
                          "M_over_N": 2, "peak_mem_gb": j.get("peak_alloc_gb"),
                          "resample_steps": j.get("resample_steps"), "argv": timed,
                          "rerun_argv_source": str(src.relative_to(HVG)),
                          "patch": "HVG_OUT env -> R18; --experiment rebuttal_R18; M,N = 1024,512 (M/N=2 as deployed)",
                          "notes": "discriminator scoring (portable HGBT) and resampling included; "
                                   "s/1k includes the per-run kernel-bank build, as the reference walltime_s does"})


# ----------------------------------------------------------------------------------- 3 uncond sample
def item_uncond_sample(d: Path):
    torch = _torch()
    sys.path.insert(0, str(BASE_CODE))
    S = load_module(BASE_CODE / "sample_hvg2k_noa4.py", "sample_hvg2k_noa4")
    root = _inside(d / "shadow_root", d)
    (root / "Baseline" / "OURS" / "results").mkdir(parents=True, exist_ok=True)
    S.ROOT, S.LOG = root, root / "RUN_LOG.md"        # exactly what rebuttal/r10_make_proposal.py:29-30 does
    ck = HVG / "assets" / "models" / "generator" / "pilot_run1c_s20260625.pt"
    sw, st = 20261801, 20261802                     # first seed = warm-up, second = timed
    argv = ["sample_hvg2k_noa4.py", "--ckpt", str(ck), "--ks", "32", "--samps", str(sw), str(st),
            "--n-gen", str(N_CELLS), "--cell-chunk", "256", "--batched", "--no-a4", "--tau", "0.0", "--tag", "_R18"]
    reset_peak()
    w0 = time.time()
    rc = with_argv(argv, S.main)
    w1 = time.time()
    if rc != 0:
        raise RuntimeError(f"sample_hvg2k_noa4.main rc={rc}")
    lg = sorted((root / "Baseline" / "OURS" / "results").glob("sample_log_*.json"))[0]
    cells = json.loads(lg.read_text(encoding="utf-8"))["cells"]
    c = cells[f"K32|samp{st}"]
    return sample_result("uncond_sample", "CRNDiff unconditional (tau 0, K32)", float(c["walltime_s"]), N_CELLS,
                         [w0, w1], f"samp {sw}: {cells[f'K32|samp{sw}']['walltime_s']:.1f} s (untimed)", "cell_chunk 256",
                         {"peak_mem_gb": peak_gb(), "argv": argv,
                          "patch": "sample_hvg2k_noa4.ROOT/LOG -> shadow (module constants)",
                          "notes": "walltime_s of the timed seed excludes the kernel-bank build, same definition as "
                                   "the reference 200_GU_machine1 walltime_s"})


# ----------------------------------------------------------------------------- 4 conditional train
def item_cond_train(d: Path):
    os.environ["HVG_OUT"] = str(_inside(d / "hvg_out", d))
    torch = _torch()
    sys.path.insert(0, str(HVG))
    import tools.t200_train_aprime as TA                                     # noqa: E402
    timer = StepTimer(name="cond_train")
    BB = TA.M.BB
    orig = BB._cosine_sched
    BB._cosine_sched = lambda opt, steps, warmup: wrap_sched_step(orig(opt, steps, warmup), timer)
    probe = {}
    orig_probe = TA.probe_s_per_step

    def probe_rec(*a, **k):
        probe["s_per_step"] = orig_probe(*a, **k)
        return probe["s_per_step"]
    TA.probe_s_per_step = probe_rec
    outd = _inside(d / "models", d)
    argv = ["t200_train_aprime.py", "--out-dir", str(outd), "--tag", "r18_aprime_v2", "--batch-from-ckpt-json",
            "--take", "best", "--divergence-guard-ratio", "1.5", "--divergence-guard-steps", "2000",
            "--hours", "5", "--seed", "20260817"]
    reset_peak()
    try:
        rc = with_argv(argv, TA.main)
        raise RuntimeError(f"t200_train_aprime.main returned rc={rc} before the timing window")
    except StopTiming:
        pass
    n, src = REAL["cond_train"]
    return train_result("cond_train", "Conditional fine-tune (A'-v2)", timer, 384, n, src,
                        {"peak_mem_gb": peak_gb(), "argv": argv, "probe_s_per_step": probe.get("s_per_step"),
                         "patch": "--out-dir -> R18; backbone._cosine_sched wrapped (steps_cap from the probe, as deployed)",
                         "notes": "G-A1 gate and the 55-step probe run as in the deployed entry point (untimed)"})


def item_cond_sample(d: Path):
    os.environ["HVG_OUT"] = str(_inside(d / "hvg_out", d))
    torch = _torch()
    sys.path.insert(0, str(HVG))
    import tools.t200_sample_conditional as TS                               # noqa: E402
    ck = HVG / "assets" / "models" / "conditional" / "aprime_v2_cond_s20260817.pt"
    ow, ot = _inside(d / "warm.npy", d), _inside(d / "timed.npy", d)
    a = ["t200_sample_conditional.py", "--ckpt", str(ck), "--label", TARGET, "--seed", "20260931", "--K", "32",
         "--cell-chunk", "256"]
    rc = with_argv(a + ["--n", "256", "--out", str(ow)], TS.main)
    if rc != 0:
        raise RuntimeError(f"warm-up rc={rc}")
    reset_peak()
    w0 = time.time()
    rc = with_argv(a + ["--n", str(N_CELLS), "--out", str(ot)], TS.main)
    w1 = time.time()
    if rc != 0:
        raise RuntimeError(f"timed rc={rc}")
    j = json.loads(ot.with_suffix(".json").read_text(encoding="utf-8"))
    return sample_result("cond_sample", "Conditional fine-tune (A'-v2)", float(j["walltime_s"]), N_CELLS, [w0, w1],
                         "n=256 call (untimed)", "cell_chunk 256",
                         {"peak_mem_gb": peak_gb(), "setup_walltime_s": j.get("setup_walltime_s"),
                          "patch": "--out -> R18 (the script's own argument)"})


# --------------------------------------------------------------------------------------- 5 CountsDiff
def _cd_module():
    return load_module(HVG / "rebuttal" / "r12_src" / "countsdiff2k.py", "countsdiff2k")


def _cd_memcap(torch):
    gb = float(os.environ.get("R18_CD_MEMCAP_GB", "8"))     # rebuttal/run_cd.py ran the reference with --mem-gb 8
    if gb > 0 and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(gb * 1024 ** 3 / torch.cuda.get_device_properties(0).total_memory, 0)
    return gb


def item_cd_train(d: Path):
    torch = _torch()
    cap = _cd_memcap(torch)
    CD = _cd_module()
    timer = StepTimer(name="cd_train")
    orig = CD.warmup_sched
    CD.warmup_sched = lambda opt, warmup: wrap_sched_step(orig(opt, warmup), timer)
    outd = _inside(d / "ckpt", d)
    argv = ["countsdiff2k.py", "train", "--data", str(DATA), "--out", str(outd), "--seed", "20260625",
            "--nmax", "512", "--steps", "20000", "--batch", "384", "--device", "cuda"]
    reset_peak()
    try:
        rc = with_argv(argv, CD.main)
        raise RuntimeError(f"countsdiff2k train returned rc={rc} before the timing window")
    except StopTiming:
        pass
    n, src = REAL["cd_train"]
    return train_result("cd_train", "CountsDiff (conditional)", timer, 384, n, src,
                        {"peak_mem_gb": peak_gb(), "argv": argv, "gpu_mem_cap_gb": cap,
                         "patch": "countsdiff2k.warmup_sched wrapped; --out -> R18; per-process memory cap as run_cd.py",
                         "notes": "eval every 200 + last.pt write included at the deployed 1-in-200 rate"})


def item_cd_sample(d: Path):
    torch = _torch()
    cap = _cd_memcap(torch)
    CD = _cd_module()
    ck = HVG / "rebuttal" / "out" / "R12" / "hvg2k" / "s20260625.best.pt"
    base = ["countsdiff2k.py", "sample", "--ckpt", str(ck), "--type", TARGET, "--guidance", "1.0",
            "--seed", "20261061", "--batch", "256", "--device", "cuda"]
    rc = with_argv(base + ["--n", "64", "--steps-k", "16", "--out-npy", str(_inside(d / "warm.npy", d))], CD.main)
    if rc != 0:
        raise RuntimeError(f"warm-up rc={rc}")
    res = {}
    for K in (256, 128):
        f = _inside(d / f"Neuronal_K{K}.npy", d)
        reset_peak()
        w0 = time.time()
        rc = with_argv(base + ["--n", str(N_CELLS), "--steps-k", str(K), "--out-npy", str(f)], CD.main)
        w1 = time.time()
        if rc != 0:
            raise RuntimeError(f"K{K} rc={rc}")
        j = json.loads(f.with_suffix(".json").read_text(encoding="utf-8"))
        res[K] = (float(j["walltime_s"]), [w0, w1], peak_gb())
    wt, win, pk = res[256]
    return sample_result("cd_sample", "CountsDiff (K256, g1.0 = Neuronal pick)", wt, N_CELLS, win,
                         "n=64, K=16 call (untimed)", "256",
                         {"peak_mem_gb": pk, "gpu_mem_cap_gb": cap,
                          "K128": {"seconds": res[128][0], "s_per_1k": res[128][0] / N_CELLS * 1000,
                                   "extrapolated_2500_s": res[128][0] / N_CELLS * 2500, "window_wall": res[128][1]},
                          "patch": "--out-npy -> R18 (the script's own argument); ckpt read-only",
                          "notes": "cd_final.py picks K256 g1.0 for Neuronal; cd_final_k128.py variant timed as K128"})


# --------------------------------------------------------------------------------- 6/7 scVI, scANVI
def _epoch_timer_cls():
    from lightning.pytorch import Callback          # scvi-tools 1.3.3 is built on `lightning`

    class EpochTimer(Callback):
        def __init__(self):
            self.ends_perf, self.ends_wall, self.start_perf = [], [], None

        def on_train_epoch_start(self, trainer, pl_module):
            if self.start_perf is None:
                _sync(); self.start_perf = time.perf_counter()

        def on_train_epoch_end(self, trainer, pl_module):
            _sync(); self.ends_perf.append(time.perf_counter()); self.ends_wall.append(time.time())
    return EpochTimer


def _vae_item(d: Path, item: str, cls_name: str, train_py: Path, gen_py: Path, gen_types):
    torch = _torch()
    import scvi                                                              # noqa: E402
    scvi.settings.logging_dir = str(_inside(d / "scvi_log", d))
    os.chdir(d)                                                  # any stray relative write lands in R18
    timer = _epoch_timer_cls()()
    cls = getattr(scvi.model, cls_name)
    orig_train = cls.train

    def train(self, *a, **k):                                    # instance call -> class attr patch
        k["max_epochs"] = WARM_EPOCHS + TIMED_EPOCHS
        cbs = list(k.pop("callbacks", None) or [])
        k["callbacks"] = cbs + [timer]
        return orig_train(self, *a, **k)
    cls.train = train
    root = _inside(d / f"{item}_root", d)
    (root / "logs").mkdir(parents=True, exist_ok=True)
    (root / "out").mkdir(parents=True, exist_ok=True)
    TM = load_module(train_py, f"r18_train_{item}")
    was = {"HERE": str(TM.HERE), "DATA": str(TM.DATA), "SEED": TM.SEED}
    TM.HERE, TM.DATA = root, DATA                                # same overrides as tools/t214_baselines.py
    reset_peak()
    TM.main()
    e = timer.ends_perf
    if len(e) < WARM_EPOCHS + TIMED_EPOCHS:
        raise RuntimeError(f"only {len(e)} epochs completed")
    a, b = WARM_EPOCHS - 1, WARM_EPOCHS + TIMED_EPOCHS - 1
    sec = e[b] - e[a]
    n, src = REAL[item]
    r = {"item": item, "method": cls_name, "phase": "train", "status": "ok", "unit": "epoch",
         "timed_steps": TIMED_EPOCHS, "seconds": sec, "s_per_step": sec / TIMED_EPOCHS,
         "first_epoch_s": e[0] - timer.start_perf, "warmup": f"{WARM_EPOCHS} epoch (untimed)",
         "batch": 128, "window_wall": [timer.ends_wall[a], timer.ends_wall[b]], "real_total": n,
         "real_total_source": src, "extrapolated_total_s": sec / TIMED_EPOCHS * n, "extrapolated": True,
         "peak_mem_gb": peak_gb(), "overridden": {"was": was, "now": {"HERE": str(root), "DATA": str(DATA)}},
         "patch": f"{cls_name}.train wrapped: max_epochs={WARM_EPOCHS + TIMED_EPOCHS} + epoch-timer callback; "
                  "HERE/DATA module constants -> R18 / assets data; scvi.settings.logging_dir -> R18",
         "notes": "library default schedule resolves to 22 epochs for 361,205 cells; batch 128 = scvi default"}
    # ---- generation: warm-up call (n=64), then the timed call (n=512 per arm) ----
    GM = load_module(gen_py, f"r18_gen_{item}")
    GM.HERE, GM.DATA, GM.TYPES = root, DATA, list(gen_types)
    GM.N_GEN = 64
    GM.main()
    meta = root / "out" / "generate_meta.json"
    if meta.exists():                                            # scVI refuses to overwrite: rename, never delete
        meta.rename(root / "out" / "generate_meta_warmup.json")
    GM.N_GEN = N_CELLS
    w0 = time.time(); _sync(); t0 = time.perf_counter()
    GM.main()
    _sync(); dt = time.perf_counter() - t0; w1 = time.time()
    ncells = len(gen_types) * 2 * N_CELLS                        # conditional + unconditional per type
    r["sample"] = {"seconds_main": dt, "cells_generated": ncells, "s_per_1k": dt / ncells * 1000.0,
                   "extrapolated_2500_s": dt / ncells * 2500.0, "types": list(gen_types),
                   "window_wall": [w0, w1], "accelerator": getattr(GM, "ACCELERATOR", "auto (GPU)"),
                   "notes": "upper bound: whole generate main() incl. model load and library-pool read, "
                            "divided by all cells it generated (cond+uncond)"}
    return r


def item_scvi(d: Path):
    c = CGE / "baseline_scvi" / "code"
    return _vae_item(d, "scvi", "SCVI", c / "train_scvi.py", c / "generate_scvi.py", [TARGET])


def item_scanvi(d: Path):
    c = CGE / "baseline_scanvi" / "code"
    # generate_scanvi's label-separation guard reads TYPES[0] and TYPES[1], so it needs two types.
    return _vae_item(d, "scanvi", "SCANVI", c / "train_scanvi.py", c / "generate_scanvi.py", [TARGET, "Endothelial"])


# ------------------------------------------------------------------------------------------- 8 CFGen
def _wandb_env(d: Path):
    w = _inside(d / "wandb", d)
    w.mkdir(parents=True, exist_ok=True)
    os.environ.update({"WANDB_MODE": "offline", "WANDB_SILENT": "true", "WANDB_DIR": str(w),
                       "WANDB_CACHE_DIR": str(w / "cache"), "WANDB_CONFIG_DIR": str(w / "config"),
                       "WANDB_DATA_DIR": str(w / "data"), "WANDB_ARTIFACT_DIR": str(w / "artifacts")})


def item_cfgen_train(d: Path):
    _wandb_env(d)
    torch = _torch()
    os.chdir(d)
    TC = load_module(CGE / "code" / "overnight" / "cfgen" / "train_cfgen.py", "r18_train_cfgen")
    sys.path.insert(0, str(TC.UP))
    import pytorch_lightning as pl                                           # noqa: E402
    import cfgen.paths as CP                                                 # noqa: E402
    import cfgen.estimator.encoder_estimator as EE                           # noqa: E402
    import cfgen.estimator.cfgen_estimator as FE                             # noqa: E402
    shadow = _inside(d / "cfgen_shadow", d)
    (shadow / "out").mkdir(parents=True, exist_ok=True)
    was = {"CFGEN": str(TC.CFGEN), "TRAINING_FOLDER": str(EE.TRAINING_FOLDER)}
    CP.TRAINING_FOLDER = EE.TRAINING_FOLDER = FE.TRAINING_FOLDER = shadow / "experiments"
    TC.CFGEN = shadow                                    # cfgen_sampling_state_full.pt / cfgen_train_full.json -> R18
    timers = []

    class LightningStepTimer(pl.Callback):
        def __init__(self, name):
            self.t = StepTimer(name=name, raise_at_end=False)

        def on_train_batch_end(self, trainer, *a, **k):
            if not self.t.done:
                self.t.tick()
                if self.t.done:
                    trainer.should_stop = True
            elif len(self.t.perf) < 10 ** 9:
                self.t.perf.append(time.perf_counter())
                if len(self.t.perf) > WARM_STEPS + TIMED_STEPS + 200:      # should_stop not honoured: abort
                    raise RuntimeError("Lightning did not stop after should_stop; aborting")

    def budget_callback(minutes):                       # replaces train_cfgen.budget_callback (module attr)
        cb = LightningStepTimer(("encoder", "fm")[min(len(timers), 1)])
        timers.append(cb)
        return cb
    TC.budget_callback = budget_callback
    argv = ["train_cfgen.py", "--tag", "full", "--max-minutes", "600"]
    reset_peak()
    err = None
    try:
        with_argv(argv, TC.main)
    except BaseException as e:                          # SystemExit included: keep what was timed
        err = f"{type(e).__name__}: {e}"
        traceback.print_exc()
    stages = {}
    total = 0.0
    real, src = REAL["cfgen_train"]
    for cb in timers:
        s = cb.t.summary()
        stages[cb.t.name] = s
        if s.get("done"):
            total += s["s_per_step"] * real[cb.t.name]
    ok = len(timers) == 2 and all(cb.t.done for cb in timers)
    first = timers[0].t.summary() if timers else {}
    r = {"item": "cfgen_train", "method": "CFGen (encoder + flow matching)", "phase": "train",
         "status": "ok" if ok else "failed", "unit": "step", "stages": stages,
         "timed_steps": TIMED_STEPS, "seconds": sum(s.get("seconds", 0.0) for s in stages.values()),
         "s_per_step": None, "warmup": f"{WARM_STEPS} steps per stage (untimed)", "batch": 256,
         "window_wall": [first.get("window_wall", [None])[0],
                         (timers[-1].t.summary().get("window_wall", [None, None])[1] if timers else None)],
         "real_total": real, "real_total_source": src, "extrapolated_total_s": total if ok else None,
         "extrapolated": True, "peak_mem_gb": peak_gb(), "error": err,
         "overridden": {"was": was, "now": {"CFGEN": str(shadow), "TRAINING_FOLDER": str(shadow / "experiments")}},
         "patch": "train_cfgen.CFGEN and cfgen.*.TRAINING_FOLDER -> R18; train_cfgen.budget_callback replaced by a "
                  "step timer that sets trainer.should_stop after 20+200 steps; wandb offline into R18",
         "notes": "per-epoch validation (~1 per 1270 steps) and the 2x h5ad load are not in the window -> "
                  "extrapolation slightly UNDER-estimates; detect_anomaly=True kept (upstream default)"}
    if not ok:
        r["error"] = err or "a stage did not reach the timing window"
    return r


def item_cfgen_sample(d: Path):
    torch = _torch()
    os.chdir(d)
    G = load_module(CGE / "code" / "overnight" / "cfgen" / "generate_cfgen.py", "r18_generate_cfgen")
    sys.path.insert(0, str(G.UP))
    root = _inside(d / "cfgen_root", d)
    (root / "out").mkdir(parents=True, exist_ok=True)
    state = CGE / "baseline_cfgen" / "out" / "cfgen_sampling_state_full.pt"
    shutil.copy2(state, root / "out" / state.name)     # COPY (no hard link into a frozen dir); 7 kB
    G.CFGEN, G.DATA, G.TYPES, G.SAMPLE_SEED = root, DATA, [TARGET], 20260931
    calls = []
    orig_build = G.build_model

    def build_model(state_, device):
        fm, cfg = orig_build(state_, device)
        ob = fm.batched_sample

        def timed(*a, **k):
            _sync(); w0 = time.time(); t0 = time.perf_counter()
            out = ob(*a, **k)
            _sync(); calls.append((time.perf_counter() - t0, k.get("batch_size"), k.get("repetitions"), w0, time.time()))
            return out
        fm.batched_sample = timed
        return fm, cfg
    G.build_model = build_model
    with_argv(["generate_cfgen.py", "--tag", "full", "--n", "64"], G.main)
    calls.clear()
    reset_peak()
    with_argv(["generate_cfgen.py", "--tag", "full", "--n", str(N_CELLS)], G.main)
    sec, bs, reps, w0, w1 = calls[0]                   # first call of the pass = conditional Neuronal
    return sample_result("cfgen_sample", "CFGen", sec, N_CELLS, [w0, w1], "--n 64 pass (untimed)", bs,
                         {"peak_mem_gb": peak_gb(), "uncond_call_s": calls[1][0] if len(calls) > 1 else None,
                          "n_sample_steps": G.N_SAMPLE_STEPS,
                          "patch": "generate_cfgen.CFGEN/DATA/TYPES/SAMPLE_SEED module constants (as tools/t210_cfgen_sample.py); "
                                   "archived sampling state COPIED into R18; build_model wrapped to time batched_sample",
                          "notes": "archived full model (fm/encoder ckpt read-only via the state's absolute paths)"})


# -------------------------------------------------------------------------------------------- 9 MDLM
MDLM_CODE = HVG / "out" / "machine2" / "MDLM_cond" / "code"
MDLM_BASE = HVG / "out" / "machine2" / "MDLM_Blackout" / "ckpt" / "mdlm_best.pt"


def _mdlm_setup(d: Path):
    """Shadow copy of the two MDLM model files as a regular `hvg` package under R18 (the driver's
    stage_root idea, but COPIES instead of hard links, and no data is linked)."""
    for m in [k for k in sys.modules if k == "hvg" or k.startswith("hvg.")]:
        raise SystemExit(f"[R18] `hvg` already imported ({m}); MDLM shadow package would be shadowed")
    sh = _inside(d / "shadow", d)
    (sh / "hvg").mkdir(parents=True, exist_ok=True)
    (sh / "hvg" / "__init__.py").write_bytes(b"")
    for f in ("model_masked_mdlm.py", "model_masked_mdlm_cond.py"):
        shutil.copy2(MDLM_CODE / "hvg" / f, sh / "hvg" / f)
    sys.path.insert(0, str(BASE_CODE))
    sys.path.insert(0, str(sh))
    C = load_module(MDLM_CODE / "tools" / "common_207_cond.py", "common_207_cond")
    C.R = sh
    C.LOGS = C.SUM = C.GATES = sh / "out"
    C.DETAIL_LOG = sh / "out" / "mdlm_cond.log"
    C.TRAIN_NPY, C.META_JSON = DATA / "train.npy", DATA / "meta.json"
    C.LABEL_DIR, C.LABELS_NPY, C.LABEL_DIR_TRAIN_NPY = DATA, DATA / "train_celltype.npy", DATA / "train.npy"
    C.MODEL_UNCOND_PY, C.MODEL_COND_PY = sh / "hvg" / "model_masked_mdlm.py", sh / "hvg" / "model_masked_mdlm_cond.py"
    C.UNCOND_CKPT = MDLM_BASE
    return C, sh


def item_mdlm_train(d: Path):
    """Replica of train_mdlm_cond.main()'s step loop (that main() refuses to run without its pre-training
    gates, so its loop body is reproduced verbatim-in-effect here; the model/loss code is the original)."""
    torch = _torch()
    import math                                                              # noqa: E402
    import numpy as np                                                       # noqa: E402
    C, sh = _mdlm_setup(d)
    from hvg.model_masked_mdlm_cond import build_mdlm_cond, load_uncond_into_cond, mdlm_cond_loss  # noqa: E402
    import config_hvg2k as CFG                                               # noqa: E402
    D = C.load_data()
    dim = D["dim"]
    dev = torch.device("cuda")
    if C.sha256(C.UNCOND_CKPT) != C.UNCOND_CKPT_SHA256:
        raise RuntimeError("MDLM base ckpt sha256 mismatch")
    uck = torch.load(C.UNCOND_CKPT, map_location=dev, weights_only=False)
    usd = uck["model_state_dict"]
    net = build_mdlm_cond(dim, usd["input_scale"].cpu().numpy()).to(dev)
    load_uncond_into_cond(net, usd)
    opt = torch.optim.AdamW(net.parameters(), lr=CFG.LR, weight_decay=CFG.WD)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / CFG.WARMUP))
    g = torch.Generator(device=dev); g.manual_seed(20260934)
    B = CFG.BATCH
    Xtr = torch.from_numpy(D["X"][D["tr_idx"]]); ytr = torch.from_numpy(D["y"][D["tr_idx"]])
    Xva = torch.from_numpy(D["X"][D["va_idx"]]); yva = torch.from_numpy(D["y"][D["va_idx"]])
    ntr = Xtr.shape[0]; n_eval = min(len(Xva), C.VAL_BATCHES * B)

    def eval_val():
        net.eval()
        gv = torch.Generator(device=dev); gv.manual_seed(C.VAL_NOISE_SEED)
        with torch.no_grad():
            return float(np.mean([float(mdlm_cond_loss(net, Xva[i:i + B].to(dev).long(), yva[i:i + B].to(dev), gen=gv))
                                  for i in range(0, n_eval, B)]))
    eval_val()                                                    # the step-0 reference eval (untimed)
    timer = StepTimer(name="mdlm_train", raise_at_end=False)
    best = math.inf
    bestp = _inside(d / "mdlm_cond_best.pt", d)
    reset_peak()
    step = 0
    while not timer.done:
        idx = torch.randint(0, ntr, (B,), generator=g, device=dev)
        ic = idx.cpu()
        x0 = Xtr[ic].to(dev).long(); yb = ytr[ic].to(dev)
        net.train()
        loss = mdlm_cond_loss(net, x0, yb, gen=g)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sched.step(); step += 1
        lf = float(loss)
        if step % C.EVAL_EVERY == 0:
            vl = eval_val()
            if vl < best:
                best = vl
                torch.save({"model_state_dict": {k: v.detach().cpu().clone() for k, v in net.state_dict().items()},
                            "step": step, "val": vl}, bestp)
        timer.tick()
        if not math.isfinite(lf):
            raise RuntimeError(f"non-finite loss at step {step}")
    real, src = REAL["mdlm_train"]
    s = timer.summary()
    r = train_result("mdlm_train", "MDLM (cell-type conditional fine-tune)", timer, B, real, src,
                     {"peak_mem_gb": peak_gb(),
                      "extrapolated_total_s": {k: s["s_per_step"] * v for k, v in real.items()},
                      "patch": "common_207_cond path constants -> assets data / R18 shadow; model files COPIED to R18 "
                               "shadow `hvg` package; training loop reproduced (main() needs its gate jsons)",
                      "notes": "plateau-stopped runs: 4400 / 5800 steps; fp32, batch 128, eval every 200 incl. best-ckpt "
                               "write; the uncond MDLM base pretraining (machine 2) is NOT re-measured"})
    return r


def item_mdlm_sample(d: Path):
    torch = _torch()
    import numpy as np                                                       # noqa: E402
    C, sh = _mdlm_setup(d)
    from hvg.model_masked_mdlm_cond import build_mdlm_cond, mdlm_cond_sample  # noqa: E402
    ckp = HVG / "out" / "mdlm_cond_216" / "mdlm_cond_s20260934" / "out" / "mdlm_cond_ckpt" / "mdlm_cond_best.pt"
    dev = torch.device("cuda")
    ck = torch.load(ckp, map_location=dev, weights_only=False)
    dim = int(ck["model_state_dict"]["input_scale"].numel())
    net = build_mdlm_cond(dim, ck["model_state_dict"]["input_scale"].cpu().numpy()).to(dev)
    net.load_state_dict(ck["model_state_dict"]); net.eval()
    names = [str(s) for s in np.unique(np.load(DATA / "train_celltype.npy", allow_pickle=True).astype(str))]
    lab = names.index(TARGET)
    mdlm_cond_sample(net, n=32, dim=dim, T=8, device=dev, seed=20260931, label=lab, chunk=C.CHUNK)   # warm-up
    reset_peak()
    w0 = time.time(); _sync(); t0 = time.perf_counter()
    arr = mdlm_cond_sample(net, n=N_CELLS, dim=dim, T=C.K, device=dev, seed=20260931, label=lab, chunk=C.CHUNK)
    _sync(); dt = time.perf_counter() - t0; w1 = time.time()
    np.save(_inside(d / "mdlm_Neuronal_n512.npy", d), arr.numpy().astype(np.int16))
    return sample_result("mdlm_sample", "MDLM (K128)", dt, N_CELLS, [w0, w1], "n=32, T=8 call (untimed)",
                         f"chunk {C.CHUNK}", {"peak_mem_gb": peak_gb(), "K": C.K, "ckpt": str(ckp.relative_to(HVG)),
                                             "patch": "same call as sample_mdlm_cond.main() (its gates skipped); output -> R18"})


# ------------------------------------------------------------------------------ 10 new-target cost
def _full_run(item, method, seconds, w0, w1, extra):
    r = {"item": item, "method": method, "phase": "new_target", "status": "ok", "unit": "full run",
         "seconds": seconds, "window_wall": [w0, w1], "extrapolated": False, "warmup": "none (full run)"}
    r.update(extra)
    return r


def item_nt_table(d: Path):
    BT = load_module(HVG / "rebuttal" / "build_tables.py", "r18_build_tables")
    BT.OUT = _inside(d / "tables", d)                 # --check writes nothing, but OUT.mkdir() would hit rebuttal/tables
    argv = ["build_tables.py", "--type", TARGET, "--check"]
    w0 = time.time(); t0 = time.perf_counter()
    with_argv(argv, BT.main)
    dt = time.perf_counter() - t0; w1 = time.time()
    return _full_run("nt_table", "New target: build tilt table (Neuronal)", dt, w0, w1,
                     {"argv": argv, "cpu_only": True, "patch": "build_tables.OUT -> R18",
                      "notes": "--check rebuilds logc_Neuronal from train cells + the 20,000-cell pilot pool and asserts "
                               "equality with assets/tables/tables_kz/logc_Neuronal.npy (|err| < 1e-12)"})


def _proposal_npy(run_dir: Path):
    res = run_dir / "items" / "nt_proposal" / "root" / "Baseline" / "OURS" / "results"
    hits = sorted(res.glob("linspace_*_R18_noa4.npy")) if res.exists() else []
    if hits:
        return hits[0], "R18 nt_proposal output"
    return (HVG / "rebuttal" / "out" / "R10" / "pool_root" / "Baseline" / "OURS" / "results" /
            "linspace_T_O_K32_spilot_run1c_s20260625_samp20260902_batched1_chunk256_CONDFULL_Neuronal_tables_kz_n20000_g0p4_noa4.npy",
            "R10 pool_root (read-only fallback)")


def item_nt_proposal(d: Path):
    torch = _torch()
    root = _inside(d / "root", d)
    argv = [str(HVG / "rebuttal" / "r10_make_proposal.py"), "--logpi",
            str(HVG / "assets" / "tables" / "tables_kz" / "logc_Neuronal.npy"), "--tag", "_R18", "--root", str(root)]
    import runpy                                                             # noqa: E402
    reset_peak()
    w0 = time.time(); t0 = time.perf_counter()
    with_argv(argv, lambda: runpy.run_path(argv[0], run_name="__main__"))
    dt = time.perf_counter() - t0; w1 = time.time()
    lg = sorted((root / "Baseline" / "OURS" / "results").glob("sample_log_*.json"))[0]
    cells = json.loads(lg.read_text(encoding="utf-8"))["cells"]
    c = next(iter(cells.values()))
    return _full_run("nt_proposal", "New target: discriminator proposal (20,000 tilted cells)", dt, w0, w1,
                     {"argv": argv, "n_cells": 20000, "sampler_walltime_s": float(c["walltime_s"]),
                      "peak_mem_gb": peak_gb(),
                      "patch": "--root -> R18 (r10_make_proposal itself redirects sample_hvg2k_noa4.ROOT/LOG)",
                      "notes": "n=20000, tau 0.4, samp 20260902 are hard-coded in r10_make_proposal (deployed recipe)"})


def item_nt_fit(d: Path, run_dir: Path):
    prop, prop_src = _proposal_npy(run_dir)
    FA = load_module(HVG / "tools" / "fit_apply_residual_hvg.py", "r18_fit_apply_residual_hvg")
    out = _inside(d / "residual_Neuronal", d)
    argv = ["fit_apply_residual_hvg.py", "--x-root", str(HVG / "assets" / "legacy_root"), "--proposal", str(prop),
            "--target", TARGET, "--fit-pos", "1800", "--val-pos", "1000", "--n-features", "192", "--max-iter", "180",
            "-o", str(out)]
    w0 = time.time(); t0 = time.perf_counter()
    with_argv(argv, FA.main)
    dt = time.perf_counter() - t0; w1 = time.time()
    log = json.loads((out / "residual_log.json").read_text(encoding="utf-8"))
    return _full_run("nt_fit", "New target: fit discriminator (HGBT, Neuronal)", dt, w0, w1,
                     {"argv": argv, "cpu_only": True, "proposal_source": prop_src,
                      "heldout_roc_auc": log.get("classifier", {}).get("heldout_roc_auc"),
                      "patch": "-o -> R18 (the script's own argument)",
                      "notes": "settings = assets/models/residual/Neuronal/residual_log.json (splits 8000/2000/10000 vs "
                               "1800/1000, 192 features, 180 iterations); includes the 5 resampling arms it writes"})


ITEMS = {"gen_train": item_gen_train, "fk_sample": item_fk_sample, "uncond_sample": item_uncond_sample,
         "cond_train": item_cond_train, "cond_sample": item_cond_sample, "cd_train": item_cd_train,
         "cd_sample": item_cd_sample, "scvi": item_scvi, "scanvi": item_scanvi, "cfgen_train": item_cfgen_train,
         "cfgen_sample": item_cfgen_sample, "mdlm_train": item_mdlm_train, "mdlm_sample": item_mdlm_sample,
         "nt_table": item_nt_table, "nt_proposal": item_nt_proposal, "nt_fit": item_nt_fit}

# Items that cannot be made provably safe are listed here and never run (none at present; see README_R18.md).
SKIPPED: dict[str, str] = {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--item", required=True, choices=sorted(ITEMS))
    ap.add_argument("--run-dir", required=True)
    a = ap.parse_args()
    run_dir = _inside(Path(a.run_dir), R18_ROOT)
    d = _inside(run_dir / "items" / a.item, R18_ROOT)
    d.mkdir(parents=True, exist_ok=True)
    (run_dir / "results").mkdir(parents=True, exist_ok=True)
    t0, w0 = time.perf_counter(), time.time()
    if a.item in SKIPPED:
        rec = {"item": a.item, "status": "skipped", "notes": SKIPPED[a.item]}
    else:
        try:
            fn = ITEMS[a.item]
            rec = fn(d, run_dir) if a.item == "nt_fit" else fn(d)
        except BaseException as e:                       # noqa: BLE001 -- record, never hide
            traceback.print_exc()
            rec = {"item": a.item, "status": "failed", "error": f"{type(e).__name__}: {e}",
                   "traceback_tail": traceback.format_exc()[-2000:]}
    rec["threads"] = env_record()
    rec["process_wall_s"] = time.perf_counter() - t0
    rec["process_window_wall"] = [w0, time.time()]
    rec["written_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    rec.setdefault("peak_mem_gb", peak_gb())
    p = run_dir / "results" / f"{a.item}.json"
    p.write_text(json.dumps(rec, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    print("R18_RESULT " + json.dumps(rec, ensure_ascii=False, default=str), flush=True)
    return 0 if rec["status"] in ("ok", "skipped") else 2


if __name__ == "__main__":
    raise SystemExit(main())
