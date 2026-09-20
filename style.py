"""Shared Matplotlib style and colour palette for the manuscript figures.

Every figure calls :func:`use_manuscript_style` before creating axes, so the
four figures form one visual system: DejaVu Serif at 10 pt, inward ticks, hairline
spines and frameless legends.

The palette is organised by *role*, not by model.  A colour means the same thing
in every figure: dark blue is always the full model, red is always a reduction
derived analytically, gold is always the lowest-order empirical fit, and the
three ``C_SET`` hues always index a parameter sweep.
"""

from __future__ import annotations

import matplotlib as mpl

# Roles shared across figures.
C_FULL = "#1a3d6d"  # full model
C_REDUCED = "#c0392b"  # analytic reduction (1D or 2D)
C_GOLD = "#d0a215"  # lowest-order empirical fit; also the draft 1D SIR reduction

# Third, fourth, ... members of a parameter sweep, in sweep order.
C_SET: tuple[str, str, str] = ("#3b76af", "#4a9c5d", "#c4553f")

# Figure 1 draws one curve family per mechanism; Figure 3 one per fit order.
C_MECHANISM: tuple[str, str, str, str] = (C_FULL, *C_SET)
C_FIT_ORDER: tuple[str, str, str, str] = (C_GOLD, *C_SET)

# Neutral greys used for annotations and reference lines.
C_ANNOTATION = "0.35"
C_REFERENCE = "0.6"

_RC = {
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.linewidth": 0.7,
    "lines.linewidth": 1.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "mathtext.fontset": "dejavuserif",
    "legend.frameon": False,
}


def use_manuscript_style() -> None:
    """Install the shared rcParams.  Call once, before any axes are created."""
    mpl.rcParams.update(_RC)


def frame_axes(*axes: mpl.axes.Axes) -> None:
    """Draw ticks inward on all four sides, the house convention for these panels."""
    for ax in axes:
        ax.tick_params(top=True, right=True, length=3)
