"""Figure 1 -- the effective force of infection at threshold.

beta_eff(i)/gamma against prevalence i for seven model instances drawn from four
mechanisms.  Each instance is evaluated AT ITS OWN epidemic threshold tau = tau_c,
so every curve passes through (0, 1).  That common point is the universal threshold
condition beta_eff(0) = gamma of Eq. (3), not an axis normalisation applied after
the fact.

The sign of the slope at (0, 1) classifies the transition through the criterion of
Eq. (4):

    downward (solid)  -> continuous
    upward   (dashed) -> discontinuous, bistable

Three of the four mechanisms appear twice, once below and once above their own
critical parameter (beta*, w*, alpha*), which is what makes the collapse visible:
the same mechanism sits on either side of the divide depending on a single scalar.

Parameters: gamma = 1, n = 5 throughout, n_tri = 3 for the simplicial model.  The
interacting-contagion case is symmetric (gamma_c = gamma), so gamma_eff = gamma and
the slope of beta_eff alone classifies the transition.

Usage
-----
    uv run figure1.py              write figure1.pdf and figure1.png
    uv run figure1.py --check      numerical audit, no figure
    uv run figure1.py --invariant  overlay the exact interacting-contagion invariant
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import matplotlib.pyplot as plt
import numpy as np

import style
from models import (
    Adaptive,
    Interacting,
    PairwiseSIS,
    Scalar,
    Simplicial,
    ad_beta_eff,
    ad_fast,
    ad_slope_at_threshold,
    ad_tau_c,
    ad_w_star,
    ic_beta_eff,
    ic_fast,
    ic_invariant_fast,
    ic_slope_at_threshold,
    ic_tau_c,
    sim_beta_eff,
    sim_fast_grid,
    sim_tau_c,
    sis_beta_eff,
    sis_fast,
    sis_slope_at_threshold,
    sis_tau_c,
)

OUTFILE = "figure1"
GAM = 1.0
N = 5
NT = 3
IMAX = 0.45  # right edge of the plotted prevalence range

# Closed-form boundary of Eq. (64) for n = 5, n_tri = 3, gamma = 1.  Reported in the
# audit header only; Figure 2 locates the same boundary numerically.
BETA_STAR_CLOSED_FORM = 0.8209


# --------------------------------------------------------------------------- #
#  beta_eff(i) at threshold, one function per mechanism
# --------------------------------------------------------------------------- #
def sis_curve(i: Scalar, tau: float, par: PairwiseSIS) -> Scalar:
    """Pairwise SIS."""
    return sis_beta_eff(i, sis_fast(i, tau, par), tau, par)


def sim_curve(i: Scalar, tau: float, par: Simplicial) -> Scalar:
    """Simplicial SIS.  p(i) is an implicit root, so it is solved grid-wise."""
    grid = np.atleast_1d(i)
    return sim_beta_eff(grid, sim_fast_grid(grid, tau, par), tau, par)


def ad_curve(i: Scalar, tau: float, par: Adaptive) -> Scalar:
    """Adaptive network SIS."""
    return ad_beta_eff(i, ad_fast(i, tau, par), tau, par)


def ic_curve(i: Scalar, tau: float, par: Interacting) -> Scalar:
    """Interacting contagions, from the quasi-static root."""
    return ic_beta_eff(i, ic_fast(i, tau, par), tau, par)


def ic_invariant_curve(i: Scalar, tau: float, par: Interacting) -> Scalar:
    """Interacting contagions, from the exact invariant p = 1 - i instead.

    Same value and same slope at i = 0; the two part company at O(i^2).  Drawn only
    under --invariant, as a check that the quasi-static root is doing its job near
    threshold.
    """
    return ic_beta_eff(i, ic_invariant_fast(i), tau, par)


# --------------------------------------------------------------------------- #
#  Curve catalogue
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class Curve:
    """One plotted instance: label, colour, transition type, threshold, beta_eff.

    ``analytic_slope`` is the closed-form d beta_eff / di at threshold where Table 1
    provides one, and None for the simplicial model, whose beta_eff is only implicit.
    """

    label: str
    colour: str
    forward: bool
    tau_c: float
    beta_eff: Callable[[Scalar], Scalar]
    analytic_slope: float | None = None


def catalogue() -> list[Curve]:
    """The seven instances, in plotting order.

    ``partial`` binds tau_c and the parameter set at construction time.  A bare
    lambda closing over the loop variables would not: Python's late binding would
    give every curve in a family the last parameter value of that loop.
    """
    c_sis, c_sim, c_ad, c_ic = style.C_MECHANISM
    curves: list[Curve] = []

    par_sis = PairwiseSIS(n=N, gamma=GAM)
    curves.append(
        Curve(
            "Pairwise (always continuous)",
            c_sis,
            True,
            sis_tau_c(par_sis),
            partial(sis_curve, tau=sis_tau_c(par_sis), par=par_sis),
            sis_slope_at_threshold(par_sis),
        )
    )

    for beta, forward in ((0.4, True), (1.3, False)):
        par = Simplicial(n=N, n_tri=NT, beta=beta, gamma=GAM)
        rel = r"<\beta^*" if forward else r">\beta^*"
        curves.append(
            Curve(
                rf"Higher-order  $\beta={beta}{rel}$",
                c_sim,
                forward,
                sim_tau_c(par),
                partial(sim_curve, tau=sim_tau_c(par), par=par),
            )
        )

    for w, forward in ((0.8, True), (3.0, False)):
        par = Adaptive(n=N, w=w, gamma=GAM)
        rel = r"<w^*" if forward else r">w^*"
        curves.append(
            Curve(
                rf"Adaptive  $w={w}{rel}$",
                c_ad,
                forward,
                ad_tau_c(par),
                partial(ad_curve, tau=ad_tau_c(par), par=par),
                ad_slope_at_threshold(par),
            )
        )

    for alpha, forward in ((1.5, True), (3.0, False)):
        par = Interacting(alpha=alpha, gamma=GAM)
        rel = r"<\alpha^*" if forward else r">\alpha^*"
        curves.append(
            Curve(
                rf"Interacting  $\alpha={alpha}{rel}$",
                c_ic,
                forward,
                ic_tau_c(par),
                partial(ic_curve, tau=ic_tau_c(par), par=par),
                ic_slope_at_threshold(par),
            )
        )

    return curves


# --------------------------------------------------------------------------- #
#  Figure
# --------------------------------------------------------------------------- #
def make_figure(outfile: str = OUTFILE, show_invariant: bool = False) -> None:
    """Draw and write the figure."""
    style.use_manuscript_style()

    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    ig = np.linspace(0.0, IMAX, 400)

    ax.axhline(1.0, color=style.C_REFERENCE, lw=0.7, ls=":", zorder=1)

    for curve in catalogue():
        ax.plot(
            ig,
            curve.beta_eff(ig) / GAM,
            ls="-" if curve.forward else "--",
            color=curve.colour,
            lw=1.5 if curve.forward else 1.4,
            label=curve.label,
            zorder=3,
        )

    if show_invariant:
        for alpha in (1.5, 3.0):
            par = Interacting(alpha=alpha, gamma=GAM)
            ax.plot(
                ig,
                ic_invariant_curve(ig, ic_tau_c(par), par) / GAM,
                ls=(0, (1, 2)),
                color=style.C_ANNOTATION,
                lw=1.0,
                zorder=2,
            )

    ax.plot([0], [1.0], "o", ms=4.5, mfc="w", mec="k", mew=0.9, zorder=5)
    # The common point sits on the dotted line; label the line where it is clear.
    ax.text(
        IMAX * 0.985,
        0.96,
        r"$\beta_{\mathrm{eff}}(0)/\gamma=1$",
        ha="right",
        va="bottom",
        fontsize=10,
        color="0.45",
    )
    # Above the line: transmission intensifies with prevalence, hence bistability.
    ax.text(
        0.022,
        0.955,
        "discontinuous\n(bistable)",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        color=style.C_ANNOTATION,
        linespacing=1.25,
    )
    ax.text(
        0.022,
        0.045,
        "continuous",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        color=style.C_ANNOTATION,
        linespacing=1.25,
    )

    ax.set_xlabel(r"prevalence  $i$")
    ax.set_ylabel(r"$\beta_{\mathrm{eff}}(i)\,/\,\gamma$")
    ax.set_xlim(0, IMAX)
    ax.set_ylim(0.79, 1.21)
    style.frame_axes(ax)
    ax.text(
        0.22,
        1.02,
        r"$\gamma=\gamma_2=1$,  $n=5$,  $n_\Delta=3$;  each model at $\tau=\tau_c$",
        transform=ax.transAxes,
        fontsize=10,
        color=style.C_ANNOTATION,
    )

    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        handlelength=2.1,
        labelspacing=0.45,
        borderpad=0.2,
    )

    fig.tight_layout()
    fig.savefig(outfile + ".pdf", bbox_inches="tight")
    fig.savefig(outfile + ".png", dpi=200, bbox_inches="tight")


# --------------------------------------------------------------------------- #
#  Numerical audit  (--check)
# --------------------------------------------------------------------------- #
def checks() -> None:
    """Audit every curve against the two things that must hold.

    Each passes through (0, 1) to machine precision, and its slope there matches the
    closed form of Table 1 where one exists.
    """
    h = 1e-5
    print(
        f"w* = {ad_w_star(Adaptive(n=N, gamma=GAM)):.4f}   alpha* = 2   "
        f"beta* = {BETA_STAR_CLOSED_FORM:.4f} (n=5, n_tri=3)\n"
    )
    print(
        f"{'curve':38s} {'tau_c':>9s} {'beff(0)/g':>13s} {'slope':>9s} {'Table 1':>9s}"
    )

    for curve in catalogue():
        b0 = float(np.atleast_1d(curve.beta_eff(1e-12))[0])
        slope = (float(np.atleast_1d(curve.beta_eff(h))[0]) - b0) / h
        analytic = curve.analytic_slope
        shown = "     --" if analytic is None else f"{analytic:+9.5f}"
        flag = "" if (curve.forward == (slope < 0)) else "   <-- SIGN MISMATCH"
        print(
            f"{curve.label:38s} {curve.tau_c:9.6f} {b0 / GAM:13.10f} "
            f"{slope:+9.5f} {shown}{flag}"
        )

    print("\ninteracting contagions: quasi-static root vs exact invariant p=1-i")
    for alpha in (1.5, 3.0):
        par = Interacting(alpha=alpha, gamma=GAM)
        tau = ic_tau_c(par)
        gaps = [
            abs(ic_curve(i, tau, par) - ic_invariant_curve(i, tau, par))
            for i in (0.15, 0.45)
        ]
        print(
            f"  alpha={alpha}: |gap| = {gaps[0]:.2e} at i=0.15, {gaps[1]:.2e} at i=0.45"
        )


if __name__ == "__main__":
    if "--check" in sys.argv:
        checks()
    else:
        make_figure(show_invariant="--invariant" in sys.argv)
