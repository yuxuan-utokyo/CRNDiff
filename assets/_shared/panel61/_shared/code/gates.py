# 搬自 cc/experiments/EXP134_more_ckpts/run_exp134.py::save / skip（原子落盘与 [SKIP] 记录）。
# 改动（逐条）：
#   1. save()/skip() 原为闭包在模块级 RES/LOGF 上，此处改为显式传入 path / logger，语义未变
#      （仍是写 .tmp 再 os.replace）。
#   2. 新增 shape_gate_gen()（任务书 §9 的形状闸门：shape==(N,61)、整数 dtype、min>=0），
#      EXP134 的 shape_gate 是给时间网格用的，不是给生成结果用的，故为新增而非搬运。
#   3. 新增 note()（[SCALE-CUT]/[CAP]/[EARLY-SUSPECT]/[P0-DEFER] 等标记的统一登记）。
"""形状闸门、原子落盘、[SKIP]/[SCALE-CUT] 记录。"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np


def save_atomic(obj, path):
    """写 .tmp 再 os.replace —— EXP134::save 逐字同义。"""
    path = str(path)
    t = path + ".tmp"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(t, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, default=float)
    os.replace(t, path)


def load_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return {} if default is None else default
    return json.load(open(path, encoding="utf-8-sig"))


def skip(res, cell, why, log=print, path=None):
    """EXP134::skip —— 记录后继续，绝不整体停止。"""
    log(f"[SKIP] {cell} {why}")
    res.setdefault("skipped", []).append({"cell": str(cell), "reason": str(why)[:300]})
    if path:
        save_atomic(res, path)
    return False


def note(res, tag, msg, log=print, path=None):
    """登记 [SCALE-CUT] / [CAP] / [EARLY-SUSPECT] / [P0-DEFER] / [FALLBACK-IMPORT] /
    [DEGRADED-EARLYSTOP] / [MIGRATION-FAIL] 一类标记。"""
    line = f"[{tag}] {msg}"
    log(line)
    res.setdefault("notes", []).append({"tag": tag, "msg": str(msg)})
    if path:
        save_atomic(res, path)
    return line


def shape_gate_gen(gen, n_expect, dim_expect=61):
    """任务书 §9 形状闸门：shape == (n_expect, dim)、整数 dtype、min >= 0。
    返回 (ok, 问题描述)。不通过由调用方记 [SKIP]。"""
    prob = []
    a = np.asarray(gen)
    if a.shape != (n_expect, dim_expect):
        prob.append(f"shape={a.shape} != ({n_expect}, {dim_expect})")
    if not np.issubdtype(a.dtype, np.integer):
        prob.append(f"dtype={a.dtype} 非整数")
    if a.size and int(a.min()) < 0:
        prob.append(f"min={int(a.min())} < 0")
    return (len(prob) == 0), "; ".join(prob)
