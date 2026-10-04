"""Figure 5 -- all five models, simulations against the fast-variable reduction.

Rows: pairwise SIS, pairwise SIR, higher-order SIS, adaptive SIS, interacting
contagions.  Columns: (a) prevalence, (b) fast variable(s) over time, (c) fast
variable against prevalence.  Each row is drawn by simulations/<model>_row.py into a
shared 5 x 3 grid sized for a 6.5-inch page, with one legend for the whole figure.

The runs are read from simulations/*.csv, written by simulations/<model>_nrm.py (see
simulations/README.md); the higher-order runs take the longest to generate.  The
adaptive and higher-order rows use networks 4x denser than Figures 2 and 3 (n x 4,
n_tri x 4).  Two tables below set what you may want to change: which runs each row
reads (DATA) and the prevalence that defines t = 0 in each row (ALIGN; None keeps the
simulated time axis).

Usage
-----
    uv run figure5.py [--align=0.01] [--align-ho=0.05] [--align-sis=none] ...
        --align=X        threshold for every row (default: ALIGN below)
        --align-KEY=X    threshold for one row; KEY in sis, sir, ho, adaptive,
                         interacting; X = none turns alignment off for that row
        --out=NAME       output name (default figure5)
        --verbose        print each row's fit diagnostics
"""

from __future__ import annotations

import sys

import matplotlib as mpl
import matplotlib.pyplot as plt

import style
from simulations import adaptive_row, interacting_row, simplicial_row, sir_row, sis_row

FONT = 6.5

# runs read by each row
DATA = {
    "sis": "simulations/sis_runs_test.csv",
    "sir": "simulations/sir_runs.csv",
    "ho": "simulations/simplicial_runs_x4b.csv",  # n = 20, n_tri = 12
    "adaptive": "simulations/adaptive_runs_x4.csv",  # n = 20
    "interacting": "simulations/interacting_runs.csv",
}
DEGREE_FACTOR = {
    "ho": 4,
    "adaptive": 4,
}  # density of the runs above relative to Figure 2

# prevalence at which each run's clock is set to t = 0 (None: keep simulated time)
ALIGN = {"sis": 0.01, "sir": 0.01, "ho": 0.05, "adaptive": 0.01, "interacting": 0.01}

TITLES = {
    "sis": "Pairwise SIS",
    "sir": "Pairwise SIR",
    "ho": "Higher-order SIS",
    "adaptive": "Adaptive network SIS",
    "interacting": "Interacting contagions",
}
COLUMN_TITLES = ("(a) prevalence", "(b) fast variable", "(c) slow manifold")

RC = {
    "font.size": FONT,
    "axes.labelsize": FONT,
    "xtick.labelsize": FONT - 0.5,
    "ytick.labelsize": FONT - 0.5,
    "legend.fontsize": FONT,
    "axes.linewidth": 0.5,
    "lines.linewidth": 1.0,
    "xtick.major.size": 2.0,
    "ytick.major.size": 2.0,
    "xtick.major.pad": 2,
    "ytick.major.pad": 2,
}


def draw_row(key: str, axes, align: float | None):
    """Draw one row and return its RowInfo."""
    csv = DATA[key]
    if key == "sis":
        return sis_row.draw(axes, csv, align=align)
    if key == "sir":
        return sir_row.draw(axes, csv, align=align)
    if key == "ho":
        return simplicial_row.draw(
            axes, csv, align=align, degree_factor=DEGREE_FACTOR["ho"]
        )
    if key == "adaptive":
        return adaptive_row.draw(
            axes, csv, align=align, degree_factor=DEGREE_FACTOR["adaptive"]
        )
    return interacting_row.draw(axes, csv, align=align)


def build(
    out: str = "figure5", align: dict | None = None, verbose: bool = False
) -> None:
    """Draw the five rows and save figures/<out>.pdf and .png."""
    align = {**ALIGN, **(align or {})}
    style.use_manuscript_style()
    with mpl.rc_context(RC):
        fig, axs = plt.subplots(5, 3, figsize=(6.5, 7.6))
        for r, key in enumerate(TITLES):
            ax = axs[r]
            info = draw_row(key, ax, align[key])
            if verbose:
                print(TITLES[key], info.notes)
            for a in ax:
                for ln in a.lines:
                    ln.set_linewidth(ln.get_linewidth() * 0.65)
                    ln.set_markersize(ln.get_markersize() * 0.6)
                    ln.set_markeredgewidth(ln.get_markeredgewidth() * 0.7)
                a.tick_params(top=True, right=True, length=2)
            ax[0].text(
                0.0,
                1.04,
                TITLES[key],
                transform=ax[0].transAxes,
                fontsize=FONT + 1,
                weight="bold",
                va="bottom",
            )
            ax[2].text(
                1.0,
                1.04,
                info.label,
                transform=ax[2].transAxes,
                fontsize=FONT - 0.5,
                color=style.C_ANNOTATION,
                va="bottom",
                ha="right",
            )
        handles = [
            plt.Line2D(
                [],
                [],
                marker="o",
                ls="",
                mfc="none",
                mec=style.C_FULL,
                ms=3,
                mew=0.6,
                label="simulations (survivors)",
            ),
            plt.Line2D(
                [],
                [],
                ls="--",
                color=style.C_REDUCED,
                label=r"analytic: 1D reduction, quasi-static $p^*(i)$",
            ),
            plt.Line2D(
                [],
                [],
                ls="-.",
                color=style.C_SET[1],
                label="1D reduction with quadratic fit of $p(i)$",
            ),
        ]
        fig.legend(
            handles=handles,
            loc="upper center",
            ncol=3,
            fontsize=FONT - 0.5,
            handlelength=2.2,
            columnspacing=1.2,
            bbox_to_anchor=(0.5, 0.999),
        )
        fig.tight_layout(h_pad=1.0, w_pad=0.6, rect=(0, 0, 1, 0.948))
        for c, t in enumerate(COLUMN_TITLES):
            pos = axs[0, c].get_position()
            fig.text(
                pos.x0 + pos.width / 2,
                0.962,
                t,
                ha="center",
                va="top",
                fontsize=FONT + 0.5,
                weight="bold",
            )
        fig.savefig(f"./figures/{out}.pdf")
        fig.savefig(f"./figures/{out}.png", dpi=250)


def _parse(argv: list[str]) -> tuple[dict, str]:
    """Read --align=X (all rows), --align-KEY=X (one row, wins) and --out=NAME."""
    over, out = {}, "figure5"

    def val(v: str) -> float | None:
        return None if v.lower() == "none" else float(v)

    for a in argv:
        if a.startswith("--align="):
            over.update({k: val(a.split("=")[1]) for k in TITLES})
    for a in argv:
        if a.startswith("--align-"):
            key, v = a[len("--align-") :].split("=")
            if key not in TITLES:
                raise SystemExit(
                    f"unknown row {key!r}; choose from {', '.join(TITLES)}"
                )
            over[key] = val(v)
        elif a.startswith("--out="):
            out = a.split("=")[1]
    return over, out


if __name__ == "__main__":
    overrides, name = _parse(sys.argv[1:])
    build(out=name, align=overrides, verbose="--verbose" in sys.argv)
