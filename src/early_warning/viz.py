"""Shared plotting style for notebooks and reports (matplotlib).

Colors come from a CVD-validated categorical palette, assigned in fixed order:
outcome "passed" is always slot 1 (blue) and "failed" slot 2 (orange), in every
figure, so identity never changes between charts.
"""

from __future__ import annotations

import matplotlib as mpl

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
PASS, FAIL = SERIES[0], SERIES[1]
OUTCOME = {"passed": PASS, "failed": FAIL}
NEUTRAL = "#8a8985"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e6e5e1"
SURFACE = "#fcfcfb"
SEQUENTIAL = "Blues"


def set_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelcolor": TEXT_2,
            "axes.edgecolor": GRID,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.prop_cycle": mpl.cycler(color=SERIES),
            "xtick.color": TEXT_2,
            "ytick.color": TEXT_2,
            "text.color": TEXT,
            "legend.frameon": False,
            "lines.linewidth": 2,
        }
    )


def save(fig, name: str, folder="../reports/figures") -> None:
    from pathlib import Path

    Path(folder).mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(folder) / f"{name}.png")
