# -*- coding: utf-8 -*-
"""Audit every purity-bearing cell of the paper against the artefacts, one row per cell.

Two mechanisms are used, because the tables are not all on the same footing:

* the main conditional-fidelity table changed convention, so its cells cannot be looked up by
  the printed value; they are read directly from the grid and rescoring artefacts;
* every other table kept its convention, so its cells CAN be looked up by the printed value,
  which makes the provenance self-checking: if no run group matches, the row is written as null
  with a reason, and nothing is ever estimated or substituted from another seed.

Arms that do not appear in the paper are excluded by name rather than dropped silently.

    CRNDIFF_PAPER_TEX=/path/to/paper.tex python scripts/audit_paper_numbers.py
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from crndiff import config as C                                          # noqa: E402

S = C.OUT / "summary"
# The submitted LaTeX source is not shipped with the repository. Point this at your copy:
#   CRNDIFF_PAPER_TEX=/path/to/paper.tex python scripts/audit_paper_numbers.py
TEX = Path(os.environ.get("CRNDIFF_PAPER_TEX", ROOT / "docs" / "paper.tex"))
OUT = S / "215_paper_numbers_v8.json"
TYPES = ["Endothelial", "Myeloid", "Neuronal"]
SHORT = {"Endo.": "Endothelial", "Mye.": "Myeloid", "Neu.": "Neuronal"}

# generator label to the five arms of the paper; a label not listed here never enters
METHOD_OF = {
    "ours": "Ours", "OURS": "Ours", "OURS_tiltonly": "Ours", None: "Ours",
    "scvi": "scVI", "scanvi": "scANVI",
    "cfgen": "CFGen", "cfgen_cond": "CFGen", "cfgen_cond_archived": "CFGen",
    "mdlm": "MDLM", "mdlm_cond": "MDLM",
}
NOT_IN_PAPER = {"dcm", "blackoutT32", "blackoutT128", "mdlmT32", "mdlmT128"}

# the real-cell row's distance columns are real against real and do not depend on the purity
REAL_DIST = {"Endothelial": {"W1": 0.011, "mmd2": 0.0001, "pcc": 1.000},
             "Myeloid": {"W1": 0.037, "mmd2": 0.0003, "pcc": 0.999},
             "Neuronal": {"W1": 0.031, "mmd2": 0.0019, "pcc": 0.993}}


# parsing the paper
def paper_tables():
    tex = TEX.read_text(encoding="utf-8", errors="replace")
    out = {}
    for m in re.finditer(r"\\begin\{table\*?\}(.*?)\\end\{table\*?\}", tex, re.S):
        blk = m.group(0)
        if "urity" not in blk:
            continue
        lab = re.search(r"\\label\{([^}]*)\}", blk)
        if lab:
            out[lab.group(1)] = blk
    return out


NUM = r"\$?\\?m?a?t?h?b?f?\{?\s*([0-9]*\.[0-9]+)\s*\}?\$?"


def cells_of(blk, mode):
    """Extract (row label, request, printed purity) triples from one table block."""
    out, cur = [], None
    for ln in blk.splitlines():
        s = ln.strip()
        g = re.search(r"\\textbf\{(Endothelial|Myeloid|Neuronal)\}", s)
        if g and "multicolumn" in s:
            cur = g.group(1)
            continue
        if not s or s.startswith(("\\toprule", "\\midrule", "\\bottomrule",
                                  "\\cmidrule", "\\begin", "\\end", "\\label",
                                  "\\caption", "%", "\\renewcommand", "\\newcommand")):
            continue
        cols = [c.strip() for c in s.rstrip("\\ ").split("&")]
        if len(cols) < 2:
            continue
        lab = re.sub(r"\\(textbf|emph|mathbf|rev)\{", "", cols[0])
        lab = lab.replace("}", "").replace("$", "").replace("\\", "").strip()
        if not lab:
            continue
        if mode == "wide":                       # three columns, one per request
            vals = [re.search(NUM, c) for c in cols[1:4]]
            if all(vals):
                for t, v in zip(TYPES, vals):
                    out.append((lab, t, float(v.group(1))))
        elif mode == "ladder":                   # C & scVI & CFGen & scANVI & MDLM & Ours
            vals = [re.search(NUM, c) for c in cols[1:6]]
            if cur and all(vals):
                for arm, v in zip(["scVI", "CFGen", "scANVI", "MDLM", "Ours"], vals):
                    out.append((f"{lab}|{arm}", cur, float(v.group(1))))
        else:                                    # a single purity column with group headings
            v = re.search(NUM, cols[1])
            if cur and v:
                out.append((lab, cur, float(v.group(1))))
    return out


# candidate groups
def key_of(stem):
    """Strip the seed part of a stem to get the key of a group."""
    k = re.sub(r"_seed\d+", "", str(stem))
    k = re.sub(r"_samp\d+", "", k)
    k = re.sub(r"train\d+", "train*", k)
    return k


def build_groups(rescore):
    g = defaultdict(list)
    for r in rescore["rows"]:
        if r["generator"] in NOT_IN_PAPER:
            continue
        g[(r["file"], r["request"], key_of(r["stem"]))].append(r)
    return g


def row_rule(lab, rowlab):
    """A predicate deciding whether a (file, stem key) belongs to a given row of the paper.

        This is the primary criterion for provenance; matching the value is only a later check.
    """
    r = rowlab.lower()

    if lab == "tab:hvg_ablation":
        # note the naming: the tilt-only stem and the FK-steering stem differ from the arm field
        # (only the arm field is called fk_only)
        if "marginal tilt" in r:
            return lambda f, s: "hvgtilt_only" in s
        if "fk steering" in r:
            return lambda f, s: "fk_notilt" in s
        if "tilted fk" in r:                       # the reference delivery
            return lambda f, s: ("residual_hvgtwist_fk_tiltevery" in s
                                 and "selftest" in f)
        return None

    if lab == "tab:hvg_tau":
        m = re.match(r"^([0-9.]+)$", rowlab.strip())
        if not m:
            return None
        tag = "_tau" + m.group(1).replace(".", "p") + "_"
        return lambda f, s, tag=tag: tag in s and f.startswith("201_tau")

    if lab == "tab:hvg_sampler_axis":
        m = re.search(r"best-of-n,\s*(\d+)", r)
        if m:
            bx = "_forced_b" + m.group(1) + "$"
            return lambda f, s, bx=bx: (re.search(bx, s) is not None
                                        and "bestofn" in s
                                        and "FINAL" in f)
        m = re.search(r"value-guided,?\s*gamma\s*=?\s*(\d+)", r) or \
            re.search(r"value-guided.*?(\d+)\s*$", r)
        if m:
            g = "_vgr_noised_g" + m.group(1) + "p0_"
            return lambda f, s, g=g: g in s
        if "fine-tuned" in r:
            return lambda f, s: "aprime_v2" in s
        if "frozen generator" in r:
            return lambda f, s: ("residual_hvgtwist_fk_tiltevery" in s
                                 and "selftest" in f)
        return None

    if lab == "tab:hvg_bestofn_ladder":
        # the paper writes it with a thousands separator, so only the digits are extracted
        m = re.match(r"^(.*)\|(\w+)$", rowlab)
        if not m:
            return None
        digits = re.sub(r"\D", "", m.group(1))
        if not digits:
            return None
        cc = "_C" + digits + "$"
        arm = m.group(2)
        tok = {"scVI": "_scvi_", "scANVI": "_scanvi_", "CFGen": "_cfgen_",
               "MDLM": "_mdlm_", "Ours": None}[arm]
        def pred(f, s, cc=cc, tok=tok):
            if "bestofn" not in s or re.search(cc, s) is None:
                return False
            if tok is None:                        # our own arm carries no generator token
                return not any(x in s for x in ("_scvi_", "_scanvi_", "_cfgen_",
                                                "_mdlm_", "_dcm_", "blackout"))
            return tok in s
        return pred

    return None


def stats(rs, field):
    v = np.array([x[field] for x in rs], dtype=float)
    return (float(v.mean()), float(v.std(ddof=1)) if v.size > 1 else 0.0, int(v.size))


def _main() -> int:
    rescore = json.loads((S / "215_purity_v8_rescore.json").read_text(encoding="utf-8"))
    ceil = {t: rescore["ceilings"][t]["ceiling_v8_derived"] for t in TYPES}
    ceil7 = {t: rescore["ceilings"][t]["ceiling_v7_published"] for t in TYPES}
    groups = build_groups(rescore)
    blocks = paper_tables()
    print(f"tables in the paper that carry purity: {sorted(blocks)}\n")

    rows, nulls = [], []

    # ------------------------------------------------------------ Table 2
    grid = json.loads((S / "215_table2_grid_3x3_v8.json").read_text(encoding="utf-8"))
    tab2_src = {"ours": grid["ours"], **grid["baselines"]}

    # stem fallback: if a cell's scored file is no longer in the summary directory, the stem
    # already resolved for that cell in the previous version is reused. No value changes; only
    prev_stems: dict = {}
    if OUT.exists():
        for r_ in json.loads(OUT.read_text(encoding="utf-8")).get("rows", []):
            prev_stems[(r_["table"], r_["row"], r_["request"], r_["metric"])] = \
                r_.get("stems") or []

    def prev_stem(arm_, t_, metric_, i_):
        lst = prev_stems.get(("tab:hvg_single_type_main", arm_, t_, metric_)) or []
        if i_ < len(lst) and not str(lst[i_]).startswith("<unresolved"):
            return str(lst[i_])
        return None

    # our row excludes one training seed and uses the other three
    # use the full grid when it exists and all its gates pass (it carries the real stems)
    og_p = S / "217_ours_grid_3x3.json"
    ours217 = json.loads(og_p.read_text(encoding="utf-8")) if og_p.exists() else None
    if ours217:
        sc = ours217["self_check"]
        if (sc["n_cells"] != 27 or sc["v7_recompute_max_abs_delta"] != 0.0
                or sc["unchanged_seeds_vs_old_v8_grid_max_abs_delta"] != 0.0):
            print(f"[!] 217_ours_grid_3x3.json grid gates did not all pass; our row keeps the previous grid")
            ours217 = None
    ours_best = None
    if ours217:
        # by the supervisor's instruction, our row shows the single best training seed, while the
        # other four arms stay at three training seeds by three sampling seeds
        # (the pre-registered full grid is kept alongside the row)
        # 'best' is the highest mean absolute v8 purity over the nine cells, ties going to the
        # smaller seed. This rule is written here before any ranking is looked at.
        # the pre-registered full-grid result is not lost: each of our rows carries it
        seeds_, samps_ = ours217["train_seeds"], ours217["sample_seeds"]

        def _score(field):
            return {tr: float(np.mean([ours217["cells"][f"{t}|{tr}|{sa}"][field]
                                       for t in TYPES for sa in samps_])) for tr in seeds_}
        sc_abs, sc_rel = _score("purity_abs"), _score("purity_rel")
        best = sorted(sc_abs.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        best_rel = sorted(sc_rel.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        if best_rel != best:
            print(f"[!] absolute and relative purity disagree ({best} vs {best_rel}); absolute decides")
        ours_best = {"presentation": "best single training seed (supervisor's instruction); "
                                     "the other four arms stay at 3 training x 3 sampling seeds",
                     "criterion": "highest mean v8 absolute purity over the 9 cells "
                                  "(3 requests x 3 sampling seeds); ties -> smaller seed",
                     "selected_train_seed": best,
                     "scores_mean_purity_abs_v8": {str(k): v for k, v in sc_abs.items()},
                     "scores_mean_purity_rel_v8": {str(k): v for k, v in sc_rel.items()},
                     "selected_by_rel_criterion": best_rel,
                     "candidate_train_seeds": seeds_,
                     "exclude_train_seeds": ours217["exclude_train_seeds"],
                     "row_index_in_grid": seeds_.index(best)}
        tab2_src["ours"] = {"train_seeds": [best], "sample_seeds": samps_,
                            "cells": {k: v for k, v in ours217["cells"].items()
                                      if v["train_seed"] == best}}
        print(f"[217] OURS row = the best training seed {best} (candidates {seeds_}), "
              f"mean purity {({k: round(v, 4) for k, v in sc_abs.items()})}; "
              f"excluding {ours217['exclude_train_seeds']}")

    # real stem lookup: the deliveries of the other training seeds at the first sampling seed
    # live in their own scored files under the long stem name, not the short one. A synthesised
    # stem would not match the provenance, so it is looked up rather than constructed.
    def real_stem(meth, t, tr, sa, samp0):
        for fn in (f"214g_{meth}_s{tr}_samp{sa}_scored.json",
                   *((f"214_{meth}_s{tr}_scored.json",) if sa == samp0 else ())):
            p = S / fn
            if not p.exists():
                continue
            cand = [r for r in json.loads(p.read_text(encoding="utf-8")).get("rows", [])
                    if (r.get("request") or r.get("arm")) == t and r.get("stem")]
            if cand:
                cand.sort(key=lambda r: not str(r["stem"]).startswith(
                    ("ours_", "scvi_", "scanvi_")))
                return str(cand[0]["stem"])
        return None
    for meth_key, arm in [("ours", "Ours"), ("scvi", "scVI"), ("scanvi", "scANVI")]:
        node = tab2_src[meth_key]
        for t in TYPES:
            vals, rels, stems = [], [], []
            for tr in node["train_seeds"]:
                for sa in node["sample_seeds"]:
                    c = node["cells"].get(f"{t}|{tr}|{sa}")
                    if c:
                        vals.append(c["purity_abs"])
                        # relativise cell by cell and then average; averaging first and dividing afterwards differs
                        # in the last unit of precision
                        rels.append(c["purity_rel"])
                        st = (c.get("stem")
                              or real_stem(meth_key, t, tr, sa, node["sample_seeds"][0])
                              or prev_stem(arm, t, "purity_abs", len(stems)))
                        stems.append(st or f"<unresolved {meth_key} {t} {tr}/{sa}>")
            v, vr = np.array(vals), np.array(rels)
            rows.append({
                "table": "tab:hvg_single_type_main", "row": arm, "request": t,
                "metric": "purity_abs", "mean": float(v.mean()),
                "sd": float(v.std(ddof=1)), "n": int(v.size),
                "n_train": len(node["train_seeds"]), "n_samp": len(node["sample_seeds"]),
                "ruler": "v8", "ceiling_v8": ceil[t], "stems": stems,
                **({"provenance_pending": True,
                    "provenance_note": "the provenance of this baseline row is unresolved; the "
                                       "value given here is the direct conditional generation"}
                   if arm == "scVI" else {}),
                **({"exclude_train_seeds": ours217["exclude_train_seeds"],
                    "train_seeds": node["train_seeds"],
                    "new_seed_weights_note": ours217["new_seed_weights"],
                    "ours_presentation": ours_best}
                   if (arm == "Ours" and ours217) else {})})
            rows.append({**rows[-1], "metric": "purity_rel",
                         "mean": float(vr.mean()), "sd": float(vr.std(ddof=1))})
            if arm == "Ours" and ours217:
                for rr, fld in ((rows[-2], "purity_abs_v8"), (rows[-1], "purity_rel_v8")):
                    rr["all_training_seeds_3x3"] = {
                        **ours217["by"][t][fld]["C_3x3"],
                        "train_seeds": ours217["train_seeds"],
                        "note": "pre-registered 3 training x 3 sampling result, not the row shown"}
    # MDLM also has a full grid once its extra training seeds exist; use the grid when present
    mgrid_p = S / "216_mdlm_grid_3x3.json"
    mgrid = (json.loads(mgrid_p.read_text(encoding="utf-8"))
             if mgrid_p.exists() else None)
    if mgrid and mgrid["self_check"]["n_cells"] != 27:
        print(f"[!] 216_mdlm_grid_3x3.json cells only "
              f"{mgrid['self_check']['n_cells']}/27 MDLM stays on a single training seed")
        mgrid = None

    # CFGen likewise has a full grid of three independent trainings by three sampling seeds
    cgrid_p = S / "219_cfgen_grid_3x3.json"
    cgrid = (json.loads(cgrid_p.read_text(encoding="utf-8"))
             if cgrid_p.exists() else None)
    if cgrid:
        sc_ = cgrid["self_check"]
        if (sc_["n_cells"] != 27 or sc_["v7_recompute_max_abs_delta"] != 0.0
                or sc_["run1_vs_existing_max_abs_delta"] != 0.0):
            print(f"[!] 219_cfgen_grid_3x3.json CFGen stays on a single training seed")
            cgrid = None
    if cgrid:
        for t in TYPES:
            cs = sorted((c for c in cgrid["cells"].values() if c["request"] == t),
                        key=lambda c: (c["cfgen_run"], c["sample_seed"]))
            stems = [str(c["stem"]) for c in cs]
            common = {"table": "tab:hvg_single_type_main", "row": "CFGen", "request": t,
                      "n_train": len(cgrid["train_runs"]),
                      "n_samp": len(cgrid["sample_seeds"]),
                      "train_runs": cgrid["train_runs"], "stems": stems}
            for metric, field in [("purity_abs", "purity_abs_v8"),
                                  ("purity_rel", "purity_rel_v8")]:
                e = cgrid["by"][t][field]["C_3x3"]
                rows.append({**common, "metric": metric, "mean": e["mean"], "sd": e["sd"],
                             "n": e["n"], "ruler": "v8", "ceiling_v8": ceil[t]})
            for metric in ("W1", "mmd2", "pcc"):
                e = cgrid["by"][t][metric]["C_3x3"]
                rows.append({**common, "metric": metric, "mean": e["mean"], "sd": e["sd"],
                             "n": e["n"], "ruler": "n/a (ruler-independent)"})
        print(f"[219] CFGen row now reads {cgrid_p.name}: three trainings x three sampling seeds")

    # one training seed by three sampling seeds, until their grids exist
    for gen, arm, f in [("cfgen_cond", "CFGen", "210_cfgen_cond_scored.json"),
                        ("mdlm_cond", "MDLM", "208_mdlm_cond_scored.json")]:
        if arm == "CFGen" and cgrid:
            continue                       # already written by the grid above
        if arm == "MDLM" and mgrid:
            for t in TYPES:
                cs = [c for k, c in mgrid["cells"].items() if c["request"] == t]
                stems = [str(c["stem"]) for c in cs]
                for metric, field in [("purity_abs", "purity_abs_v8"),
                                      ("purity_rel", "purity_rel_v8")]:
                    e = mgrid["by"][t][field]["C_3x3"]
                    rows.append({
                        "table": "tab:hvg_single_type_main", "row": "MDLM", "request": t,
                        "metric": metric, "mean": e["mean"], "sd": e["sd"], "n": e["n"],
                        "n_train": len(mgrid["train_seeds"]),
                        "n_samp": len(mgrid["sample_seeds"]),
                        "ruler": "v8", "ceiling_v8": ceil[t], "stems": stems,
                        "shared_base_caveat": mgrid["shared_base"]["caveat"]})
                for metric in ("W1", "mmd2", "pcc"):
                    e = mgrid["by"][t][metric]["C_3x3"]
                    rows.append({
                        "table": "tab:hvg_single_type_main", "row": "MDLM", "request": t,
                        "metric": metric, "mean": e["mean"], "sd": e["sd"], "n": e["n"],
                        "n_train": len(mgrid["train_seeds"]),
                        "n_samp": len(mgrid["sample_seeds"]),
                        "ruler": "n/a (ruler-independent)", "stems": stems})
            continue
        for t in TYPES:
            rs = [r for r in rescore["rows"]
                  if r["file"] == f and r["request"] == t and r["generator"] == gen]
            if not rs:
                nulls.append({"table": "tab:hvg_single_type_main", "row": arm,
                              "request": t, "reason": f"no rows in {f}"})
                continue
            m, sd, n = stats(rs, "purity_v8")
            base = {"table": "tab:hvg_single_type_main", "row": arm, "request": t,
                    "metric": "purity_abs", "mean": m, "sd": sd, "n": n,
                    "n_train": 1, "n_samp": n, "ruler": "v8", "ceiling_v8": ceil[t],
                    "stems": [str(x["stem"]) for x in rs]}
            rows.append(base)
            rows.append({**base, "metric": "purity_rel",
                         "mean": m / ceil[t], "sd": sd / ceil[t]})

    # the other tables: looked up by the printed value
    modes = {"tab:hvg_de": "wide", "tab:hvg_tstr": "wide",
             "tab:hvg_bestofn_ladder": "ladder"}
    for lab, blk in blocks.items():
        if lab == "tab:hvg_single_type_main":
            continue
        mode = modes.get(lab, "single")
        for rowlab, t, printed in cells_of(blk, mode):
            if lab in ("tab:hvg_de", "tab:hvg_tstr"):
                # these two tables only repeat the main table, so they point back at it
                arm = {"scVI": "scVI", "CFGen": "CFGen", "scANVI": "scANVI",
                       "MDLM": "MDLM", "Ours": "Ours",
                       "Real cells": "Real cells"}.get(rowlab.replace("emph", "").strip())
                if arm is None:
                    continue
                if arm == "Real cells":
                    rows.append({"table": lab, "row": "Real cells", "request": t,
                                 "metric": "purity_abs", "mean": ceil[t], "sd": None,
                                 "n": 1, "n_train": 0, "n_samp": 0, "ruler": "v8",
                                 "ceiling_v8": ceil[t],
                                 "stems": ["<archived real val cells>"],
                                 "note": "= v8 ceiling"})
                    continue
                src = [x for x in rows
                       if x["table"] == "tab:hvg_single_type_main"
                       and x["row"] == arm and x["request"] == t
                       and x["metric"] == "purity_abs"]
                if not src:
                    nulls.append({"table": lab, "row": arm, "request": t,
                                  "reason": "Table 2 the source row is missing"})
                    continue
                rows.append({**src[0], "table": lab,
                             "note": "repeats Table 2 (tab:hvg_single_type_main)"})
                continue

            # the stem is located by the row's SEMANTICS; the value is only a check.
            # looking up by value alone mismatched once: two unrelated arms agreed to within 3e-4 by
            # coincidence
            need = row_rule(lab, rowlab)
            if need is None:
                nulls.append({"table": lab, "row": rowlab, "request": t,
                              "printed_v7": printed,
                              "reason": "no stem rule is registered for this row"})
                continue
            cand = []
            for k, rs in groups.items():
                if k[1] != t or not need(k[0], k[2]):
                    continue
                m7, _, n = stats(rs, "purity_v7")
                if n in (3, 9):
                    cand.append((k, rs, m7))
            if not cand:
                nulls.append({"table": lab, "row": rowlab, "request": t,
                              "printed_v7": printed,
                              "reason": "no stem group satisfies both the row rule and n in {3,9}"})
                continue
            # gate: the semantically selected group must reproduce the printed value
            ok = [c for c in cand if abs(round(c[2], 3) - printed) <= 5e-4]
            if not ok:
                nulls.append({"table": lab, "row": rowlab, "request": t,
                              "printed_v7": printed,
                              "reason": "the group selected by the semantic rule has a v7 mean that does not "
                                        "reproduce the value printed in the paper: "
                                        + "; ".join(f"{c[0][0]}|{c[0][2]}={c[2]:.4f}"
                                                    for c in cand[:4])})
                continue
            ok.sort(key=lambda c: (0 if "FINAL" in c[0][0] else 1,
                                   0 if "_raw_" not in c[0][0] else 1,
                                   -len(c[0][2])))
            k, rs, m7 = ok[0]
            cand = ok
            gen = rs[0]["generator"]
            m8, sd8, n = stats(rs, "purity_v8")
            rows.append({
                "table": lab, "row": rowlab, "request": t, "metric": "purity_abs",
                "mean": m8, "sd": sd8, "n": n, "n_train": 1, "n_samp": n,
                "ruler": "v8", "ceiling_v8": ceil[t],
                "stems": [str(x["stem"]) for x in rs],
                "method": METHOD_OF.get(gen, "?"),
                "printed_v7": printed, "matched_v7_mean": m7,
                "source_file": k[0],
                "n_candidates": len(cand)})

    # the main table's distance columns
    # they do not depend on the purity ruler, but they follow the same grid convention, so they
    DIST = [("W1", "W1"), ("mmd2", "mmd2_rbf_biased"), ("pcc", "pcc")]
    for meth_key, arm in [("ours", "Ours"), ("scvi", "scVI"), ("scanvi", "scANVI")]:
        node = tab2_src[meth_key]
        for t in TYPES:
            for gk, _ in DIST:
                vals, stems = [], []
                for tr in node["train_seeds"]:
                    for sa in node["sample_seeds"]:
                        c = node["cells"].get(f"{t}|{tr}|{sa}")
                        if c and isinstance(c.get(gk), (int, float)):
                            vals.append(c[gk])
                            st = (c.get("stem")
                                  or real_stem(meth_key, t, tr, sa, node["sample_seeds"][0])
                                  or prev_stem(arm, t, gk, len(stems)))
                            stems.append(st or f"<unresolved {meth_key} {t} {tr}/{sa}>")
                if not vals:
                    nulls.append({"table": "tab:hvg_single_type_main", "row": arm,
                                  "request": t, "metric": gk,
                                  "reason": "3x3 the grid carries no such distance value"})
                    continue
                v = np.array(vals)
                rows.append({
                    "table": "tab:hvg_single_type_main", "row": arm, "request": t,
                    "metric": gk, "mean": float(v.mean()),
                    "sd": float(v.std(ddof=1)) if v.size > 1 else 0.0, "n": int(v.size),
                    "n_train": len(node["train_seeds"]),
                    "n_samp": len(node["sample_seeds"]),
                    "ruler": "n/a (ruler-independent)", "stems": stems,
                    **({"ours_presentation": ours_best,
                        "all_training_seeds_3x3": {
                            **ours217["by"][t][gk]["C_3x3"],
                            "train_seeds": ours217["train_seeds"],
                            "note": "pre-registered 3 training x 3 sampling result, "
                                    "not the row shown"}}
                       if (arm == "Ours" and ours217) else {})})
    for gen, arm, f in [("cfgen_cond", "CFGen", "210_cfgen_cond_scored.json"),
                        ("mdlm_cond", "MDLM", "208_mdlm_cond_scored.json")]:
        if (arm == "MDLM" and mgrid) or (arm == "CFGen" and cgrid):
            continue                       # the distance columns were written with the grid above
        raw = json.loads((S / f).read_text(encoding="utf-8"))
        for t in TYPES:
            rs = [r for r in raw.get("rows", [])
                  if (r.get("request") or r.get("arm")) == t]
            for gk, field in DIST:
                vals = [r[field] for r in rs if isinstance(r.get(field), (int, float))]
                if not vals:
                    nulls.append({"table": "tab:hvg_single_type_main", "row": arm,
                                  "request": t, "metric": gk,
                                  "reason": f"{f} field not present in {field}"})
                    continue
                v = np.array(vals)
                rows.append({
                    "table": "tab:hvg_single_type_main", "row": arm, "request": t,
                    "metric": gk, "mean": float(v.mean()),
                    "sd": float(v.std(ddof=1)) if v.size > 1 else 0.0, "n": int(v.size),
                    "n_train": 1, "n_samp": int(v.size),
                    "ruler": "n/a (ruler-independent)",
                    "stems": [str(r.get("stem")) for r in rs]})

    # gate: the main table must agree cell by cell
    cal = json.loads((S / "215_table2_three_calibres_v8.json").read_text(encoding="utf-8"))
    gate, worst = 0, None
    for meth_key, arm in [("ours", "Ours"), ("scvi", "scVI"), ("scanvi", "scANVI")]:
        for t in TYPES:
            mine = [x for x in rows
                    if x["table"] == "tab:hvg_single_type_main" and x["row"] == arm
                    and x["request"] == t and x["metric"] == "purity_rel"]
            if meth_key == "ours" and ours217:
                # our row is the best training seed, compared against that seed's row in the grid
                mrow = ours217["by"][t]["purity_rel_v8"]["matrix_rows_train_cols_samp"][
                    ours_best["row_index_in_grid"]]
                ref_mean = float(np.array(mrow).mean())
            else:
                e = cal["arms"][meth_key]["by"][t]["purity_rel_v8"]
                if not e.get("complete"):
                    continue
                ref_mean = e["C_3x3"]["mean"]
            d = abs(mine[0]["mean"] - ref_mean)
            gate += 1
            if worst is None or d > worst[0]:
                worst = (d, arm, t)
    print(f"[G] Table 2 {gate} cells vs 215_table2_three_calibres_v8.json: "
          f"max |delta| = {worst[0]:.3e} ({worst[1]}/{worst[2]})")
    if worst[0] != 0.0:
        print("[G] FAIL")
        return 1
    print("[G] PASS (bitwise)")

    blob = {"ticket": "215#2", "ruler": "v8",
            "ruler_file": "assets/rulers/master_purity_v8_lineage.json",
            # the real-cell row: purity is the v8 ceiling, while the distance columns are real against
            # real and do not move with the ruler, so they are recorded here for cross-checking
            "real_cells": {
                t: {"purity_abs": ceil[t], "purity_rel": 1.0,
                    "purity_abs_v7_published": ceil7[t],
                    "distances_unchanged": REAL_DIST[t],
                    "note": "purity moves to the v8 ceiling; W1/MMD^2/PCC are real against real and do not change"}
                for t in TYPES},
            "paper_arms": ["Ours", "scVI", "scANVI", "CFGen", "MDLM"],
            "excluded_arms": sorted(NOT_IN_PAPER),
            "tex": str(TEX), "rows": rows, "nulls": nulls,
            "written_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    if OUT.exists():                      # never overwrite an existing artefact: archive the old version first
        arch = OUT.with_name(f"{OUT.stem}__preexisting_{int(time.time())}.json")
        OUT.rename(arch)
        print(f"[archive] archived previous version -> -> {arch.name}")
    OUT.write_text(json.dumps(blob, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    bytab = defaultdict(int)
    for r in rows:
        bytab[r["table"]] += 1
    print("rows written per table")
    for t, n in sorted(bytab.items()):
        print(f"   {t:<34}{n}")
    print(f"\ntotal rows, of which null:")
    if nulls:
        print("\ncells written as null, and why:")
        for x in nulls:
            print("   ", json.dumps(x, ensure_ascii=False))
    print(f"\n[out] {OUT}")
    return 0


def main() -> int:
    # shared-summary write lock (tickets 217 / 219 may run concurrently)
    from scripts._summary_lock import summary_write_lock
    with summary_write_lock("t215_paper_numbers"):
        return _main()


if __name__ == "__main__":
    raise SystemExit(main())
