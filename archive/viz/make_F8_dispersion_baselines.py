# -*- coding: utf-8 -*-
"""112 / 107: F8 dispersion, our arm against three published baselines.

Zero training, zero sampling. Every array is an already registered product and
every number in the figure is computed here from those arrays and written to
`F8_dispersion.data.json`.

WHAT CHANGED, AND WHY THE GATE CONSTANT MOVED
---------------------------------------------
Ticket 107 asked for the Table 1 cell sets AND for the real median Fano to
reproduce the old figure's 1.44 on Endothelial. Receipt 058 showed those two
cannot both hold: the old figure drew its real cells from a 2500-cell subsample
at seed 20261030, while Table 1 uses the whole held-out set for the type. On
Table 1's cells the real Endothelial median Fano is 1.5671, not 1.44. Ticket 112
adopts option A -- keep the Table 1 cell sets, move the gate constant to 1.57 --
so this script gates on 1.57 / 1.62 / 1.49.

LANDING ON THE SAME CELLS AS TABLE 1 (ticket 107 s2.1)
-------------------------------------------------------
Not just the same files: the same ROWS. `e21_scc_mmd.py` creates ONE generator
per cell type at seed 20261071 and consumes it in a fixed order --

    floor (real train) -> scVI -> CFGen -> scANVI -> OURS (tilt) -> OURS (tilt+FK)

so the draw any one arm receives depends on every draw before it. This figure
drops the train floor and the pure-tilt arm, but it still CONSUMES their draws,
because skipping them would silently hand OURS (tilt+FK) a different subsample
and the figure would no longer sit on Table 1's cells. Real cells are never
subsampled: matched n IS the full held-out count for the type, which is exactly
why the Endothelial constant moved.

THE BOLD RULE (ticket 115 s1)
-----------------------------
The winner is not named in this file. The criterion is
|median Fano - real median Fano|, smallest wins, computed INDEPENDENTLY IN EACH
PANEL from the same numbers that go into F8_dispersion.data.json; `real` is the
reference, never a contender; ties at the displayed precision are all bolded.
Whoever wins gets the whole row bolded. That the answer happens to be OURS in
all three panels today is an output, not an input -- hard-coding it would be
self-declaring first place on the figure.

COLOURS
-------
Read from a source constant rather than typed. `PAPER_ARM_COLOR` lives in
`TOY/_shared/code/style.py`, NOT in `HVG2K/viz/style.py` (which has its own
`ARM_COLOR` with different hexes) -- the four hexes ticket 112 s1.3 names are
that table's. It is loaded under a distinct module name so the two same-named
style modules cannot shadow each other. Note the KEY NAMES there are the
unconditional-family arm names; the ticket fixes which hex each baseline takes,
so the mapping is by hex-as-specified and the inherited key name is recorded
beside it, so nobody later reads "mdlm" as a statement about scVI.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
HVG = HERE.parent
ROOT = HVG.parent
XROOT = Path(__file__).resolve().parents[2] / "assets" / "legacy_root"
DATA = XROOT / "_shared" / "data" / "all"
DIA = HVG / "results" / "diagnostics"
LIVE_FIGS = HVG / "results" / "figures"
FIGS = Path(os.environ["F8_OUT_DIR"]) if os.environ.get("F8_OUT_DIR") else LIVE_FIGS
E21_JSON = DIA / "E21_scc_mmd.json"
PAPER_STYLE = ROOT / "TOY" / "_shared" / "code" / "style.py"
STEM = "F8_dispersion"
STAMP = "20260825_pre115"   # ticket 115 round; the pre112 archive is kept

SEED = 20261071                      # E21's subsample seed
BINS = np.concatenate([[0.0], np.logspace(-3, 1.4, 22)])   # make_dispersion.py:58
EXPORT_DPI = 400

# ticket 112 s1.3: which hex each series takes, and which PAPER_ARM_COLOR key
# that hex is stored under. The key names are inherited and do NOT describe
# these methods.
SERIES = [
    ("OURS (tilt+FK)", "ours_selfcal"),
    ("scVI", "mdlm"),
    ("CFGen", "blackout"),
    ("scANVI", "ddpm"),
]
DISPLAY_DP = 2          # the Fano values are printed to 2 dp
BOLD_CRITERION = ("smallest |median Fano - real median Fano| within the panel; "
                  "real is the reference and is never bolded; ties at the "
                  "displayed precision are all bolded")

# real median Fano the gate must reproduce (ticket 112 s1.2)
GATE = {"Endothelial": 1.57, "Myeloid": 1.62, "Neuronal": 1.49}


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def rel(p: Path) -> str:
    try:
        return p.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(p.resolve())


def select_rows(arr, n: int, rng) -> np.ndarray:
    """Verbatim from code/eval/e21_scc_mmd.py."""
    if arr.shape[0] < n:
        raise SystemExit(f"source has {arr.shape[0]} rows but matched n is {n}")
    if arr.shape[0] == n:
        return np.asarray(arr)
    idx = np.sort(rng.choice(arr.shape[0], n, replace=False))
    return np.asarray(arr[idx])


def band(mean, var):
    """Verbatim from code/eval/make_dispersion.py."""
    out = []
    idx = np.digitize(mean, BINS)
    for b in range(1, len(BINS)):
        m = idx == b
        if m.sum() < 8:
            continue
        out.append(dict(mean_lo=float(BINS[b - 1]), mean_hi=float(BINS[b]),
                        n_genes=int(m.sum()), mean_mid=float(np.median(mean[m])),
                        var_q25=float(np.quantile(var[m], 0.25)),
                        var_med=float(np.median(var[m])),
                        var_q75=float(np.quantile(var[m], 0.75))))
    return out


def dispersion(a):
    a = np.asarray(a, dtype=np.float64)
    mean = a.mean(0)
    var = a.var(0, ddof=1)
    pos = mean > 0
    fano = var[pos] / mean[pos]
    return {"n_genes_mean_gt0": int(pos.sum()),
            "fano_median": float(np.median(fano)),
            "fano_q25": float(np.quantile(fano, 0.25)),
            "fano_q75": float(np.quantile(fano, 0.75)),
            "band": band(mean[pos], var[pos])}, mean[pos], var[pos]


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    sys.path.insert(0, str(HERE))
    import style
    style.apply_rc()
    spec = importlib.util.spec_from_file_location("_paper_style", PAPER_STYLE)
    paper_style = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(paper_style)
    colours = {name: paper_style.PAPER_ARM_COLOR[key] for name, key in SERIES}
    print("[palette] " + "  ".join(f"{n}={c}" for n, c in colours.items()))

    e21 = json.loads(E21_JSON.read_text(encoding="utf-8"))
    if int(e21["seed"]) != SEED:
        raise SystemExit(f"E21 seed drift: {e21['seed']} != {SEED}")
    xv = np.load(DATA / "val.npy", mmap_mode="r")
    yv = np.load(DATA / "val_celltype.npy", allow_pickle=True).astype(str)
    xt = np.load(DATA / "train.npy", mmap_mode="r")
    yt = np.load(DATA / "train_celltype.npy", allow_pickle=True).astype(str)

    types = list(e21["types"])
    data = {
        "figure": STEM,
        "contract": rel(ROOT / "Paper" / "adversarial" / "comms" / "cloud_to_cc"
                        / "112_F8_ruling_option_A.md"),
        "zero_training": True, "zero_sampling": True,
        "domain": "raw counts (Fano is only meaningful before normalisation)",
        "poisson_reference": "Fano = 1",
        "cell_sets": {
            "rule": "identical to Table 1 / E21_scc_mmd.json",
            "real": "the FULL held-out set for the type; matched n IS that count, "
                    "so real cells are never subsampled",
            "generated": "subsampled to matched n with one generator per type at "
                         f"seed {SEED}, consumed in E21's order "
                         "(floor -> scVI -> CFGen -> scANVI -> OURS (tilt) -> "
                         "OURS (tilt+FK)); the floor and pure-tilt draws are "
                         "consumed but not plotted, so the plotted arms receive "
                         "exactly the rows Table 1 scored",
            "e21_source": rel(E21_JSON), "e21_md5": md5(E21_JSON),
            "palette_source": rel(PAPER_STYLE),
        },
        "gate": {"requirement": "ticket 112 s1.2 -- real median Fano must "
                                "reproduce 1.57 / 1.62 / 1.49 at the figure's "
                                "displayed precision",
                 "constants": GATE, "checks": [], "passed": None},
        "palette": {n: {"hex": c, "paper_arm_color_key": k,
                        "note": "key name is inherited from the unconditional "
                                "family table and does not describe this method"}
                    for (n, k), c in zip(SERIES, [colours[n] for n, _ in SERIES])},
        "types": {},
    }

    per_type = {}
    gate_ok = True
    print("[gate] real median Fano on the Table 1 cell sets")
    for ct in types:
        n = int(e21["types"][ct]["n"])
        vi = np.flatnonzero(yv == ct)
        if len(vi) != n:
            raise SystemExit(f"{ct}: expected {n} held-out cells, found {len(vi)}")
        rng = np.random.default_rng(SEED)
        real_raw = np.asarray(xv[vi])
        _floor = select_rows(xt[np.flatnonzero(yt == ct)], n, rng)   # consumed
        rows = e21["types"][ct]["rows"]
        arms, srcs = {}, {}
        for method in ("scVI", "CFGen", "scANVI", "OURS (tilt)", "OURS (tilt+FK)"):
            p = Path(rows[method]["source"])
            if not p.is_file():
                raise SystemExit(f"missing registered source: {p}")
            arr = np.load(p, mmap_mode="r")
            drawn = select_rows(arr, n, rng)
            if method == "OURS (tilt)":
                continue                                            # consumed only
            arms[method] = drawn
            srcs[method] = {"path": rel(p), "md5": md5(p),
                            "n_source": int(arr.shape[0]), "n_used": n}

        real_stat, rmean, rvar = dispersion(real_raw)
        got = round(real_stat["fano_median"], 2)
        same = (got == GATE[ct])
        gate_ok &= same
        data["gate"]["checks"].append(
            {"type": ct, "recomputed": real_stat["fano_median"],
             "rounded": got, "constant": GATE[ct], "matches": bool(same)})
        print(f"    {'OK  ' if same else 'DIFF'}  {ct:<14} "
              f"{real_stat['fano_median']:.6f} -> {got}  vs {GATE[ct]}")

        stats = {"Real val": real_stat}
        for m, a in arms.items():
            stats[m], _, _ = dispersion(a)
        per_type[ct] = {"real": (rmean, rvar), "stats": stats}
        data["types"][ct] = {"n": n, "n_val": int(len(vi)),
                             "sources": srcs,
                             "arms": {k: {kk: vv for kk, vv in v.items()
                                          if kk != "band"} for k, v in stats.items()},
                             "bands": {k: v["band"] for k, v in stats.items()}}

    if not gate_ok:
        print("\n[gate] STOP -- the real cells are not the Table 1 set. Nothing written.")
        raise SystemExit(2)
    data["gate"]["passed"] = True
    print(f"[gate] PASS -- {len(types)}/{len(types)} reproduce\n")

    # ---------------------------------------------------------- archive ----
    old = FIGS / "_old"
    old.mkdir(parents=True, exist_ok=True)
    archived = {}
    # ticket 115: the json was NOT archived in earlier rounds, which left the
    # numbers gate without a before-state and forced it back onto the receipt
    # record. The data/panels json are products too -- archive them.
    for ext in ("pdf", "png", "svg", "panels.json", "data.json"):
        src = FIGS / f"{STEM}.{ext}"
        if src.is_file():
            dst = old / f"{STEM}_{STAMP}.{ext}"
            if dst.exists():
                raise SystemExit(f"refusing to overwrite archive {dst}")
            shutil.copy2(src, dst)
            archived[rel(dst)] = md5(dst)
            print(f"[archive] {dst.name} md5={md5(dst)}")

    # ----------------------------------------------------------- render ----
    bold_rule, box_rows, box_texts = {}, {}, {}
    fig, axes = plt.subplots(1, 3, figsize=(style.WIDE_W, 2.86),
                             sharex=True, sharey=True)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.760, bottom=0.240,
                        wspace=0.09)
    for ax, ct in zip(axes, types):
        st = per_type[ct]["stats"]
        rmean, rvar = per_type[ct]["real"]
        ax.scatter(rmean, rvar, s=1.1, c=style.INK, alpha=0.16, linewidths=0,
                   rasterized=True, zorder=2)
        lo, hi = 1e-3, 30
        ax.plot([lo, hi], [lo, hi], ls=(0, (3, 2)), lw=0.7, color="#9a9892", zorder=3)
        b = st["Real val"]["band"]
        ax.fill_between([x["mean_mid"] for x in b], [x["var_q25"] for x in b],
                        [x["var_q75"] for x in b], color=style.INK, alpha=0.13,
                        linewidth=0, zorder=4)
        ax.plot([x["mean_mid"] for x in b], [x["var_med"] for x in b], lw=1.3,
                color=style.INK, zorder=6)
        for name, _ in SERIES:
            bb = st[name]["band"]
            ax.plot([x["mean_mid"] for x in bb], [x["var_med"] for x in bb],
                    lw=1.1, color=colours[name], zorder=5)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(1e-3, 30)
        ax.set_ylim(1e-4, 3e3)
        ax.set_title(ct, fontsize=style.FS_LABEL, pad=3)
        ax.set_xlabel("Per-gene mean count")
        # ---- ticket 115 s1: mechanical winner, drawn line by line ----------
        real_f = st["Real val"]["fano_median"]
        gaps = {n: abs(st[n]["fano_median"] - real_f) for n, _ in SERIES}
        best = min(round(g, DISPLAY_DP) for g in gaps.values())
        winners = sorted(n for n, g in gaps.items()
                         if round(g, DISPLAY_DP) == best)
        bold_rule[ct] = {
            "criterion": BOLD_CRITERION,
            "real_median_fano": real_f,
            "abs_gap_to_real": {n: gaps[n] for n, _ in SERIES},
            "abs_gap_rounded": {n: round(gaps[n], DISPLAY_DP) for n, _ in SERIES},
            "winners": winners,
            "tie": len(winners) > 1,
            "real_row_bold": False,
        }
        print(f"  [bold] {ct}: winner {winners} "
              + "  ".join(f"{n}={gaps[n]:.4f}" for n, _ in SERIES), flush=True)

        rows = [("median Fano", "", False, True)]
        rows.append(("real", f"{real_f:.2f}", False, False))
        for n, _ in SERIES:
            label = "OURS" if n.startswith("OURS") else n
            rows.append((label, f"{st[n]['fano_median']:.2f}",
                         n in winners, False))
        box_rows[ct] = rows
        drawn = []
        for i, (label, val, bold, header) in enumerate(rows):
            y = 0.962 - i * 0.072
            w = "bold" if bold else "normal"
            drawn.append(ax.text(0.045, y, label, transform=ax.transAxes,
                                 ha="left", va="top", fontweight=w,
                                 fontsize=style.FS_MIN - 1.5, color=style.INK,
                                 zorder=9))
            if val:
                drawn.append(ax.text(0.315, y, val, transform=ax.transAxes,
                                     ha="right", va="top", fontweight=w,
                                     fontsize=style.FS_MIN - 1.5,
                                     color=style.INK, zorder=9))
        box_texts[ct] = drawn
    # One rounded panel behind each set of lines. Drawn after the fact so it
    # wraps whatever the bold row actually measures, rather than a guess.
    from matplotlib.transforms import Bbox
    from matplotlib.patches import FancyBboxPatch
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    for ax, ct in zip(axes, types):
        bb = Bbox.union([t.get_window_extent(renderer=rend) for t in box_texts[ct]])
        bba = bb.transformed(ax.transAxes.inverted())
        pad = 0.020
        ax.add_patch(FancyBboxPatch(
            (bba.x0 - pad, bba.y0 - pad), bba.width + 2 * pad, bba.height + 2 * pad,
            transform=ax.transAxes, boxstyle="round,pad=0.012",
            fc="#ffffffe8", ec=style.GRID, lw=0.5, zorder=8))

    axes[0].set_ylabel("Per-gene variance")

    handles = [Line2D([0], [0], color=style.INK, lw=1.3)]
    labels = ["Real cells (scatter + IQR band)"]
    for name, _ in SERIES:
        handles.append(Line2D([0], [0], color=colours[name], lw=1.1))
        labels.append("\\ours{} (tilt+FK)".replace("\\ours{}", "OURS")
                      if name.startswith("OURS") else name)
    handles.append(Line2D([0], [0], color="#9a9892", lw=0.7, ls=(0, (3, 2))))
    labels.append("Poisson (var = mean)")
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.53, 0.905),
               ncol=6, handlelength=1.6, columnspacing=1.0, borderpad=0.0,
               fontsize=style.FS_MIN - 1, frameon=False)
    for i, ax in enumerate(axes):
        bb = ax.get_position()
        fig.text((bb.x0 + bb.x1) / 2, 0.020, f"({'abc'[i]})", ha="center",
                 va="bottom", fontsize=style.FS_LABEL, color=style.INK)

    FIGS.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(FIGS / f"{STEM}.{ext}", dpi=EXPORT_DPI, facecolor=style.BG)
    import dpi_selfcheck
    panels = FIGS / f"{STEM}.panels.json"
    dpi_selfcheck.record_panels(fig, panels)
    plt.close(fig)
    rep = dpi_selfcheck.report(FIGS / f"{STEM}.pdf", panels)
    lo_dpi = rep["dpi_lower_bound"]
    print(f"[dpi] measured lower bound {lo_dpi}  required [365, 400]")
    if not 365.0 <= lo_dpi <= 400.0:
        raise SystemExit(f"dpi {lo_dpi} outside [365, 400]")

    data["bold_rule"] = {
        "ticket": "115 s1",
        "criterion": BOLD_CRITERION,
        "display_dp": DISPLAY_DP,
        "winner_is_an_output": "the figure does not hard-code which method wins; "
                               "if a baseline were closest, that row would be the "
                               "bold one",
        "per_type": bold_rule,
    }
    data["measured_dpi_lower_bound"] = lo_dpi
    data["archived"] = archived
    data["md5"] = {f"{STEM}.{e}": md5(FIGS / f"{STEM}.{e}")
                   for e in ("pdf", "png", "svg")}
    data["md5"][f"{STEM}.panels.json"] = md5(panels)
    (FIGS / f"{STEM}.data.json").write_text(
        json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    # s1.1 says the winner is computed FROM the data file. Reread what was
    # just written and re-derive it there, so "why is this row bold" is
    # checkable by anyone with the json and no access to this script.
    written = json.loads((FIGS / f"{STEM}.data.json").read_text(encoding="utf-8"))
    for ct in types:
        rf = written["types"][ct]["arms"]["Real val"]["fano_median"]
        g = {n: abs(written["types"][ct]["arms"][n]["fano_median"] - rf)
             for n, _ in SERIES}
        b = min(round(x, DISPLAY_DP) for x in g.values())
        w = sorted(n for n, x in g.items() if round(x, DISPLAY_DP) == b)
        if w != written["bold_rule"]["per_type"][ct]["winners"]:
            raise SystemExit(f"bold rule not reproducible from the json for {ct}")
    print("[bold] rule re-derived from the written json for all "
          f"{len(types)} panels: identical")

    for k, v in data["md5"].items():
        print(f"[out] {k} md5={v}")
    print(f"[out] {STEM}.data.json")

    print(f"\n{'type':<14}{'real':>7}" + "".join(f"{n[:6]:>8}" for n, _ in SERIES))
    for ct in types:
        st = per_type[ct]["stats"]
        print(f"{ct:<14}{st['Real val']['fano_median']:>7.2f}"
              + "".join(f"{st[n]['fano_median']:>8.2f}" for n, _ in SERIES))


if __name__ == "__main__":
    main()
