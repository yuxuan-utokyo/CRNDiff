# -*- coding: utf-8 -*-
"""Shared figure style for every HVG2K figure.

WHY THIS FILE EXISTS
--------------------
Colour must follow the *entity*, not the figure: the same arm has to be the same
colour in F1, F2 and F3, or a reader comparing panels will draw wrong
conclusions.  Every `make_F*.py` imports from here and none of them defines a
colour of its own.  The palette and geometry are fixed by the taskbook; change
them here and every figure changes together.

CONVENTIONS
-----------
* Arm -> colour is a hard mapping (`ARM_COLOR`).
* Single-column width 3.4 in (F3 may use 7 in); minimum font size 7 pt.
* No titles inside the figure -- captions belong to LaTeX.
* All in-figure text is English.
* F2 uses a single blue ramp; F3 uses a blue/red diverging map with a neutral
  midpoint and a symmetric, shared colour limit.  jet/viridis are banned in F3
  because a non-diverging map hides the sign of a correlation error.

PITFALL PAID FOR
----------------
Matplotlib's default `savefig` leaves a transparent background; a reader pasting
the PNG onto a dark slide then sees black text on black.  `save()` writes an
explicit white background and also emits svg+pdf from the same figure object so
the three formats cannot drift apart.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ----------------------------------------------------------------- palette
INK = "#0b0b0b"          # primary text / axes
INK2 = "#52514e"         # secondary text
GRID = "#e1e0d9"         # hairline grid
BG = "#ffffff"

# "Real" is no longer black.  PLOTTING_RULES item 10 bans black as an identity
# colour (it competes with axes/text and carries no hue); the cloud ruling in
# comms/cloud_to_cc/006_answers.md made that ban global.  #7a5230 (deep umber)
# was chosen by MEASUREMENT, not taste: with the palettes it actually appears
# in (F7, F10c) it passes the lightness band, CVD separation and the
# normal-vision floor (worst pair 16.2), where black FAILED the lightness band
# outright (L 0.15 against a 0.43-0.77 band).
#
# EXCEPTION (Q, 2026-08-25, ticket 115 s2.2): F8's `real` reference CURVE
# keeps black. The 006 ruling banned black as an IDENTITY colour; F8's real
# line is a reference, not a cell identity, and Q ruled to leave it after
# being told about the rule. Recorded here so the rule and the figures stop
# contradicting each other. The original ban stands for identity colours.
#
# CAVEAT ON THE 0.43-0.77 BAND ITSELF (ticket 115 s3, executor's note):
# the band's UNITS AND SOURCE ARE NO LONGER RECOVERABLE. Measured under all
# three usual readings, black is L*=0, L*/100=0 and relative luminance 0 --
# never the 'L 0.15' cited above -- and REAL itself is L*/100 = 0.385, below
# the band it is said to pass. DO NOT USE THIS BAND AS A SELECTION
# CRITERION. Colour choice goes by measured CIEDE2000 + CVD separation
# instead (ticket 114 s3; see viz/palette_measure.py and
# results/figure_data/palette_mesothelial_selection.json). No colour value
# is changed by this note.
REAL = "#7a5230"

ARM_COLOR = {
    "Unconditional": "#898781",
    "Uncond": "#898781",
    "Product tilt": "#2a78d6",
    "Product tilt g=0.4": "#2a78d6",
    "Tilt": "#2a78d6",
    "Final": "#eb6834",
    "Real val": REAL,
    "Real cells": REAL,
}

# ----------------------------------------------------------------- type colours
# SECOND, SEPARATE mapping: colour by CELL TYPE, for the mechanism figures
# (F4-F7) whose series are types rather than methods.
#
# Relationship to ARM_COLOR, stated because the two share a palette family and a
# reader must not blend them: `#2a78d6` is "Product tilt" in F1-F3 and
# "Endothelial" in F4-F7; `#eb6834` is "Full method" there and "Myeloid" here.
# They are drawn from the same palette so the figures look like one system, but
# they are DIFFERENT SEMANTIC FAMILIES.  Every figure using TYPE_COLORS carries
# an explicit legend or direct labels, and no figure mixes the two mappings, so
# a colour is never ambiguous inside a single panel.  Do not build a figure that
# needs both at once; split it instead.
# ONE COLOUR, ONE MEANING (ticket 114, 2026-08-25).  Q ruled four changes and
# the registry below is now the single source for every family-A identity; no
# figure script may carry a family-A hex of its own (ticket 114 s4).
TYPE_COLORS = {
    # 114 ruling 1: the endothelial identity moves to the lighter blue that F20
    # / Figure 11 already used.  #2a78d6 is NO LONGER an endothelial identity
    # colour anywhere; it survives only as an ARM colour (product tilt) and as
    # the cool end of CMAP_DIV, neither of which is a cell type.
    "Endothelial": "#6aa6ee",
    "Myeloid": "#eb6834",
    "Neuronal": "#1baf7a",
    # 114 ruling 2 + s3: #eda100 had to go because it is essentially the same
    # amber as the UAP1-positive marker #e69f00.  The replacement was chosen by
    # measurement, not taste -- CIEDE2000 against every other fixed family-A
    # colour under normal vision, protan and deutan, winner = largest worst-case
    # minimum.  Selected by ticket 114 s3, minimum deltaE = 13.2 (protan, closest
    # opponent = REAL #7a5230); runners-up deep teal #00767a at 12.9 and deep
    # indigo #3f3d99 at 12.7.  Full matrix:
    # results/figure_data/palette_mesothelial_selection.json
    "Mesothelial": "#00695c",
    # 114 ruling 3: the magenta is now THE ventricular identity everywhere,
    # bar charts included.  See the note where TYPE_COLOR_SCATTER used to be.
    "Ventricular_Cardiomyocyte": "#d13b8a",
}

# TYPE_COLOR_SCATTER WAS REMOVED BY TICKET 114 (ruling 3), 2026-08-25.
#
# It existed because #8a5cd1 was fine for F10 (a bar chart, where the two types
# never touch) but failed badly wherever the two types are intermixed points:
# against Endothelial it measured deltaE 2.0 under deutan -- indistinguishable
# for red-green colourblind readers -- and only 12.5 for normal vision.  The fix
# at the time was a second map, which left one cell type carrying two hues
# depending on the view.  Q ended that double-track: #d13b8a is now the single
# ventricular identity in TYPE_COLORS above and the scatter variant is gone.
# Anything still importing TYPE_COLOR_SCATTER is out of date and should read
# TYPE_COLORS instead.

# Subset / status markers.  These are NOT a sixth and seventh cell type: they
# qualify cells that already have a type identity, so they live in their own
# mapping to keep family A's "one colour one meaning" exact.
SUBSET_COLORS = {
    # 114 ruling 2: the gene-condition marker keeps the amber it already had in
    # F12; it means "this endothelial cell expresses the requested gene".
    "UAP1_positive": "#e69f00",
    # 114 ruling 4: what used to be purple #9467bd.  Purple was carrying two
    # unrelated meanings across figures (neither-type, and "OURS generated"),
    # so both were retired; this neutral grey-blue takes the neither-type one.
    "neither": "#9db4c0",
}

# Non-identity background greys.  Kept here so figures stop re-declaring them,
# but they carry no semantics and a figure may still use its own background.
ATLAS_BG_WIDE = "#e7e6e1"
ATLAS_BG = "#ecebe7"


def assert_palette(**expected: str) -> None:
    """Drift guard for figure scripts (ticket 114 s4.3).

    Call with the values the figure believes it is drawing, e.g.
    ``style.assert_palette(Endothelial="#6aa6ee")``.  A mismatch is a hard stop:
    a figure that silently draws a different colour than the registry is exactly
    the failure this ticket exists to end.
    """
    known = {**TYPE_COLORS, **SUBSET_COLORS,
             "ATLAS_BG_WIDE": ATLAS_BG_WIDE, "ATLAS_BG": ATLAS_BG, "REAL": REAL}
    bad = []
    for name, want in expected.items():
        if name not in known:
            bad.append(f"{name} is not in the registry")
        elif known[name].lower() != str(want).lower():
            bad.append(f"{name}: registry {known[name]} but figure expects {want}")
    if bad:
        raise SystemExit("palette drift: " + "; ".join(bad))

# F2: single blue ramp
CMAP_SEQ = LinearSegmentedColormap.from_list("hvg2k_blue", ["#cde2fb", "#0d366b"])
# F3: diverging blue <-> red with a neutral midpoint
CMAP_DIV = LinearSegmentedColormap.from_list(
    "hvg2k_div", ["#2a78d6", "#f0efec", "#e34948"])

# ----------------------------------------------------------------- geometry
COL_W = 3.4       # single column, inches
WIDE_W = 7.0      # F3 triptych
FS_MIN = 7        # minimum font size, pt
FS_TICK = 7
FS_LABEL = 8
FS_ANNOT = 7


def apply_rc() -> None:
    plt.rcParams.update({
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "savefig.facecolor": BG,
        "savefig.edgecolor": BG,
        "font.family": "DejaVu Sans",
        "font.size": FS_TICK,
        "axes.labelsize": FS_LABEL,
        "axes.titlesize": FS_LABEL,
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "legend.fontsize": FS_MIN,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "svg.fonttype": "none",     # keep text editable in the SVG
        "pdf.fonttype": 42,         # TrueType, not Type-3
    })


def save(fig, outdir: Path, stem: str, dpi: int = 300) -> list[Path]:
    """Write png(dpi)+svg+pdf from one figure object; return the paths."""
    outdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("png", "svg", "pdf"):
        p = outdir / f"{stem}.{ext}"
        fig.savefig(p, dpi=dpi, bbox_inches="tight", facecolor=BG)
        paths.append(p)
    return paths
