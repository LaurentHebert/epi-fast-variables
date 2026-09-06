"""
fig1_skeleton.py
================
Figure 1 for ""Fast-variable reduction to 
                        complex contagion unifies epidemic models".

Effective force of infection beta_eff(i)/gamma against prevalence i for seven
model instances drawn from four mechanisms.  Each instance is evaluated AT ITS
OWN epidemic threshold tau = tau_c, so every curve passes through (0, 1): that
common point is the universal threshold condition beta_eff(0) = gamma, not an
axis normalisation applied after the fact.

The sign of the slope at (0, 1) classifies the transition:
    downward (solid)  -> continuous
    upward   (dashed) -> discontinuous, bistable

Parameters: gamma = 1, n = 5 throughout; n_triangle = 3 for the simplicial
model; the interacting-contagion case is symmetric (gamma_c = gamma), so
gamma_eff = gamma and the slope of beta_eff alone classifies the transition.

Model right-hand sides and slaved fast variables are imported from
fast_variable_figure.py so that this figure and Figs. 2-3 cannot drift apart.

Style follows fast_variable_figure.py / fitted_reduction_figure.py:
DejaVu Serif at 10 pt, inward ticks on all four sides, frameless legend, and
the same four-colour model palette.

Run:  python fig1_skeleton.py            (add --check for the numerical audit)
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from fast_variable_figure import (
    sis_p, sis_tauc,
    ho_p, ho_beff, ho_tauc,
    ad_p1, ad_tauc, ad_wstar,
    ic_p, ic_tauc,
)

GAM = 1.0
N = 5
NT = 3                      # triangle degree for the simplicial model
IMAX = 0.45                 # right edge of the plotted prevalence range


# ===================================================================== #
#  beta_eff(i) at threshold, one function per mechanism
# ===================================================================== #
def beff_sis(i, tau, n=N, gam=GAM):
    """Pairwise SIS, Eq. (24):  beta_eff = tau n p(i)."""
    return tau * n * sis_p(i, tau, n, gam)


def beff_ho(i, tau, beta, n=N, gam=GAM, nT=NT):
    """Simplicial SIS, Eq. (45).  p(i) is an implicit root of the quartic."""
    p = np.array([ho_p(x, tau, n, gam, beta, nT) for x in np.atleast_1d(i)])
    return ho_beff(np.atleast_1d(i), p, tau, n, gam, beta, nT)


def beff_ad(i, tau, w, n=N, gam=GAM):
    """Adaptive network SIS, Eq. (86):  beta_eff = tau n p1(i)."""
    return tau * n * ad_p1(i, tau, n, gam, w)


def beff_ic(i, tau, alpha, gam=GAM):
    """Interacting contagions, Eq. (100).

    NOTE the explicit susceptible factor (1 - i): in the well-mixed setting it
    is part of beta_eff itself.  `ic_beff` in fast_variable_figure.py returns
    only tau[alpha - (alpha-1)p] and leaves the (1 - i) to the prevalence
    equation, so it is NOT the quantity plotted here.  Dropping the factor
    flips the sign of the slope at i = 0 -- it would draw alpha = 1.5 as a
    backward case, contradicting alpha* = 2.
    """
    return tau * (alpha - (alpha - 1) * ic_p(i, tau, alpha, gam)) * (1 - i)


def beff_ic_invariant(i, tau, alpha, gam=GAM):
    """Same model, but using the exact invariant <II> = <I>^2, i.e. p = 1 - i,
    instead of the quasi-static root.  Same value and same slope at i = 0;
    they part company at O(i^2).  Not drawn by default -- see --invariant."""
    return tau * (alpha - (alpha - 1) * (1 - i)) * (1 - i)


# ===================================================================== #
#  Curve catalogue:  (label, colour, forward?, tau_c, beta_eff)
# ===================================================================== #
def catalogue(gam=GAM, n=N, nT=NT):
    C_SIS, C_HO, C_AD, C_IC = "#1a3d6d", "#3b76af", "#4a9c5d", "#c4553f"
    cur = []

    tc = sis_tauc(n, gam)
    cur.append(dict(lab="Pairwise (always continuous)", col=C_SIS, fwd=True,
                    tauc=tc, f=lambda i, tc=tc: beff_sis(i, tc)))

    for beta, fwd in ((0.4, True), (1.6, False)):
        tc = ho_tauc(beta, n, gam, nT)
        rel = r"<\beta^*" if fwd else r">\beta^*"
        cur.append(dict(lab=rf"Simplicial  $\beta={beta}{rel}$", col=C_HO,
                        fwd=fwd, tauc=tc,
                        f=lambda i, tc=tc, b=beta: beff_ho(i, tc, b)))

    for w, fwd in ((0.5, True), (3.0, False)):
        tc = ad_tauc(n, gam, w)
        rel = r"<w^*" if fwd else r">w^*"
        cur.append(dict(lab=rf"Adaptive  $w={w}{rel}$", col=C_AD, fwd=fwd,
                        tauc=tc, f=lambda i, tc=tc, ww=w: beff_ad(i, tc, ww)))

    for al, fwd in ((1.5, True), (3.0, False)):
        tc = ic_tauc(gam)
        rel = r"<\alpha^*" if fwd else r">\alpha^*"
        cur.append(dict(lab=rf"Interacting  $\alpha={al}{rel}$", col=C_IC,
                        fwd=fwd, tauc=tc,
                        f=lambda i, tc=tc, a=al: beff_ic(i, tc, a)))
    return cur


# ===================================================================== #
#  Figure
# ===================================================================== #
def make_figure(outfile="fig_beff_skeleton", show_invariant=False):
    mpl.rcParams.update({
        "font.family": "serif", "font.serif": ["DejaVu Serif"],
        "font.size": 10, "axes.labelsize": 10,
        "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
        "axes.linewidth": 0.7, "lines.linewidth": 1.5,
        "xtick.direction": "in", "ytick.direction": "in",
        "mathtext.fontset": "dejavuserif", "legend.frameon": False,
    })

    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    ig = np.linspace(0.0, IMAX, 400)

    ax.axhline(1.0, color="0.6", lw=0.7, ls=":", zorder=1)

    for c in catalogue():
        ax.plot(ig, c["f"](ig) / GAM, ls="-" if c["fwd"] else "--",
                color=c["col"], lw=1.5 if c["fwd"] else 1.4,
                label=c["lab"], zorder=3)

    if show_invariant:                      # sanity overlay, off by default
        for al in (1.5, 3.0):
            ax.plot(ig, beff_ic_invariant(ig, ic_tauc(GAM), al) / GAM,
                    ls=(0, (1, 2)), color="0.35", lw=1.0, zorder=2)

    ax.plot([0], [1.0], "o", ms=4.5, mfc="w", mec="k", mew=0.9, zorder=5)
    # the common point sits on the dotted line; label the line where it is clear
    ax.text(IMAX * 0.985, 0.96, r"$\beta_{\mathrm{eff}}(0)/\gamma=1$",
            ha="right", va="bottom", fontsize=10, color="0.45")
    # above the line = transmission intensifies with prevalence = backward
    ax.text(0.022, 0.955, "discontinuous\n(bistable)", transform=ax.transAxes,
            ha="left", va="top", fontsize=10, color="0.35", linespacing=1.25)
    ax.text(0.022, 0.045, "continuous", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=10, color="0.35",
            linespacing=1.25)

    ax.set_xlabel(r"prevalence  $i$")
    ax.set_ylabel(r"$\beta_{\mathrm{eff}}(i)\,/\,\gamma$")
    ax.set_xlim(0, IMAX)
    ax.set_ylim(0.79, 1.21)
    ax.tick_params(top=True, right=True, length=3)
    ax.text(0.28, 1.02, r"$\gamma=1$,  $n=5$,  $n_\Delta=3$;  each model at $\tau=\tau_c$",
            transform=ax.transAxes, fontsize=10, color="0.35")

    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
              handlelength=2.1, labelspacing=0.45, borderpad=0.2)

    fig.tight_layout()
    fig.savefig(outfile + ".pdf", bbox_inches="tight")
    fig.savefig(outfile + ".png", dpi=200, bbox_inches="tight")
    print(f"wrote {outfile}.pdf and {outfile}.png")


# ===================================================================== #
#  Numerical audit
# ===================================================================== #
def checks():
    """Two things must hold for every curve: it passes through (0, 1) to
    machine precision, and its slope there matches Table 1."""
    n, gam, nT = N, GAM, NT
    ana = {
        "Pairwise (always continuous)": -2 * gam * (n - 1)**2 / (n * (2*n - 1)),
        r"Adaptive  $w=0.8<w^*$":
            2 * gam * ((n - 1) * 0.8 - (n + 1) * gam) / (gam * (2*n + 1) + 0.8),
        r"Adaptive  $w=3.0>w^*$":
            2 * gam * ((n - 1) * 3.0 - (n + 1) * gam) / (gam * (2*n + 1) + 3.0),
        r"Interacting  $\alpha=1.5<\alpha^*$": gam * (1.5 - 2),
        r"Interacting  $\alpha=3.0>\alpha^*$": gam * (3.0 - 2),
    }
    h = 1e-5
    print(f"w* = {ad_wstar(n, gam):.4f}   alpha* = 2   "
          f"beta* = 0.8209 (n=5, n_tri=3)\n")
    print(f"{'curve':38s} {'tau_c':>9s} {'beff(0)/g':>13s} "
          f"{'slope':>9s} {'Table 1':>9s}")
    for c in catalogue():
        b0 = float(np.atleast_1d(c["f"](1e-12))[0])
        s = (float(np.atleast_1d(c["f"](h))[0]) - b0) / h
        a = ana.get(c["lab"])
        astr = "     --" if a is None else f"{a:+9.5f}"
        flag = "" if (c["fwd"] == (s < 0)) else "   <-- SIGN MISMATCH"
        print(f"{c['lab']:38s} {c['tauc']:9.6f} {b0/GAM:13.10f} "
              f"{s:+9.5f} {astr}{flag}")

    print("\ninteracting contagions: quasi-static root vs exact invariant p=1-i")
    for al in (1.5, 3.0):
        d = [abs(beff_ic(i, ic_tauc(GAM), al)
                 - beff_ic_invariant(i, ic_tauc(GAM), al)) for i in (0.15, 0.45)]
        print(f"  alpha={al}: |gap| = {d[0]:.2e} at i=0.15, {d[1]:.2e} at i=0.45")


if __name__ == "__main__":
    import sys
    if "--check" in sys.argv:
        checks()
    else:
        make_figure(show_invariant="--invariant" in sys.argv)
