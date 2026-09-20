"""Figure 2 -- fast-variable reduction of four canonical models.

Rows   : pairwise SIS | higher-order (simplicial) | adaptive network SIS |
         interacting contagions
Columns: (a) prevalence i(t), full model vs the 1D reduction
         (b) fast-variable relaxation p(t), full model vs the quasi-static p*(i)
         (c) bifurcation diagram in tau: full-model up-sweep (filled) and
             down-sweep (open) against the analytic endemic branch (dashed)

Column (c) is what makes the transition type visible: where up- and down-sweeps
separate, the model is bistable.  Each row sweeps a third parameter across the
model's own critical value, so every row shows both sides of the divide.

The reductions plotted here are derived in Appendix 6.2 and summarised in
Section 3; the model equations themselves live in ``models.py``.

Usage
-----
    uv run figure2.py            write figure2.pdf and figure2.png
    uv run figure2.py --check    numerical audit, no figure
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

import style
from models import (
    Adaptive,
    Interacting,
    ModelParams,
    PairwiseSIS,
    Simplicial,
    ad_branch,
    ad_fast,
    ad_fast_p2,
    ad_rhs,
    ad_rhs_1d,
    ad_tau_c,
    ad_w_star,
    ic_beta_eff,
    ic_fast,
    ic_rhs,
    ic_rhs_1d,
    ic_tau_c,
    sim_beta_eff,
    sim_beta_star,
    sim_fast,
    sim_rhs,
    sim_rhs_1d,
    sim_tau_c,
    sis_beta_eff,
    sis_fast,
    sis_rhs,
    sis_rhs_1d,
    sis_tau_c,
)

OUTFILE = "figure2"
GAM = 1.0
I0 = 1e-3

# Tight tolerances for the plotted trajectories; looser ones for the sweeps, which
# integrate to steady state many times over.
IVP = dict(rtol=1e-10, atol=1e-13, method="LSODA", dense_output=True)
SWP = dict(rtol=1e-9, atol=1e-12, method="LSODA")

# Simplicial boundary as located numerically by `--check` below.  The closed form of
# Eq. (64) gives 0.8209; the two differ in the fourth decimal and the legend rounds
# to two, so the curve drawn here is the numerically located one.
BSTAR = 0.8246


# --------------------------------------------------------------------------- #
#  Continuation helpers
# --------------------------------------------------------------------------- #
def endemic_branch(
    residual: Callable[[float, float], float],
    igrid: NDArray[np.float64],
    lo: float = 1e-5,
    hi: float = 6.0,
) -> NDArray[np.float64]:
    """For each prevalence i, solve residual(i, tau) = 0 for tau.

    Parameterising the branch by i rather than by tau traces the whole S-curve,
    unstable arm included, which is what makes the fold visible in column (c).
    Prevalences with no root in [lo, hi] come back as NaN and simply go undrawn.
    """
    out = []
    for i in igrid:
        try:
            out.append(brentq(lambda t, i=i: residual(i, t), lo, hi, xtol=1e-13))
        except ValueError:
            out.append(np.nan)
    return np.array(out)


def sweep(
    rhs: Callable[..., list[float]],
    seed: Sequence[float],
    taus: NDArray[np.float64],
    par: ModelParams,
    i_of: Callable[[NDArray[np.float64]], float],
    T: float = 6000,
    up: bool = True,
    cont: bool = True,
) -> NDArray[np.float64]:
    """Continuation sweep of the FULL model.

    Integrate to steady state at each tau, carrying the previous state forward when
    ``cont``.  Running the sweep up and then down exposes hysteresis wherever the
    bifurcation is discontinuous: the two passes coincide on a continuous branch and
    separate across a bistable window.
    """
    ys, out = None, []
    for t in taus if up else taus[::-1]:
        y0 = (
            np.array(seed, float)
            if (ys is None or not cont)
            else np.clip(np.nan_to_num(ys, nan=1e-7), 1e-10, 1.0)
        )
        ys = solve_ivp(rhs, [0, T], y0, args=(t, par), **SWP).y[:, -1]
        out.append(i_of(np.clip(np.nan_to_num(ys, nan=0.0), 0.0, 1.0)))
    out = np.array(out)
    return out if up else out[::-1]


def _sweep_pair(
    rhs: Callable[..., list[float]],
    up_seed: Sequence[float],
    down_seed: Sequence[float],
    taus: NDArray[np.float64],
    par: ModelParams,
    i_of: Callable[[NDArray[np.float64]], float],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Up-sweep from a near-disease-free seed and down-sweep from an endemic one."""
    return (
        sweep(rhs, up_seed, taus, par, i_of),
        sweep(rhs, down_seed, taus, par, i_of, up=False),
    )


def _plot_sweeps(
    ax: plt.Axes,
    taus: NDArray[np.float64],
    up: NDArray[np.float64],
    down: NDArray[np.float64] | None,
    colour: str,
    label: str,
) -> None:
    """Filled markers for the up-sweep, thinner open ones for the down-sweep."""
    ax.plot(taus, up, "o", ms=3.0, mfc="none", color=colour, alpha=0.7, label=label)
    if down is not None:
        ax.plot(taus, down, "o", ms=3.0, mfc="none", mew=0.55, color=colour, alpha=0.7)


# --------------------------------------------------------------------------- #
#  Rows
# --------------------------------------------------------------------------- #
def row_pairwise_sis(ax: NDArray, gam: float, i0: float) -> None:
    """Pairwise SIS -- continuous for every degree (Appendix 6.2.1, Step 4)."""
    par = PairwiseSIS(n=5, gamma=gam)
    tau, T = 1.3 * sis_tau_c(par), 70
    full = solve_ivp(sis_rhs, [0, T], [i0, (1 - i0) * i0], args=(tau, par), **IVP)
    red = solve_ivp(sis_rhs_1d, [0, T], [i0], args=(tau, par), **IVP)
    tt = np.linspace(0, T, 900)

    ax[0].plot(tt, full.sol(tt)[0], color=style.C_FULL, label="full model")
    ax[0].plot(tt, red.sol(tt)[0], "--", color=style.C_REDUCED, label="1D reduction")
    ax[0].legend(frameon=False, loc="center right")
    ax[1].plot(tt, full.sol(tt)[1] / full.sol(tt)[0], color=style.C_FULL)
    ax[1].plot(
        tt,
        sis_fast(np.clip(red.sol(tt)[0], 1e-12, 0.999), tau, par),
        "--",
        color=style.C_REDUCED,
    )

    a = ax[2]
    taus = np.linspace(0.02, 0.60, 90)
    ig = np.linspace(1e-4, 0.93, 350)
    for colour, n in zip(style.C_SET, [4, 6, 10], strict=True):
        p = PairwiseSIS(n=n, gamma=gam)
        # No continuation: each tau is seeded fresh, the branch being single-valued.
        up = sweep(sis_rhs, [1e-5, 1e-5], taus, p, lambda y: y[0], cont=False)
        _plot_sweeps(a, taus, up, None, colour, f"$n={n}$")
        a.plot(
            endemic_branch(
                lambda i, t, p=p: sis_beta_eff(i, sis_fast(i, t, p), t, p) - gam, ig
            ),
            ig,
            "--",
            color=colour,
            lw=1.1,
        )
    a.set_xlim(0, 0.6)
    a.set_ylim(-0.02, 1.1)
    a.legend(frameon=False, loc="upper left")
    a.set_title(r"(c) bifurcation, continuous for all $n$", loc="right", fontsize=10)


def row_simplicial(ax: NDArray, gam: float, i0: float) -> None:
    """Simplicial SIS -- discontinuous above beta* (Eq. (64))."""
    par = Simplicial(n=5, n_tri=3, beta=0.4, gamma=gam)
    tau, T = 1.3 * sim_tau_c(par), 80
    full = solve_ivp(sim_rhs, [0, T], [i0, 1 - i0], args=(tau, par), **IVP)
    red = solve_ivp(sim_rhs_1d, [0, T], [i0], args=(tau, par), **IVP)
    tt = np.linspace(0, T, 900)

    ax[0].plot(tt, full.sol(tt)[0], color=style.C_FULL)
    ax[0].plot(tt, red.sol(tt)[0], "--", color=style.C_REDUCED)
    ax[1].plot(tt, full.sol(tt)[1], color=style.C_FULL)
    ax[1].plot(
        tt,
        [sim_fast(max(x, 1e-12), tau, par) for x in red.sol(tt)[0]],
        "--",
        color=style.C_REDUCED,
    )

    a = ax[2]
    taus = np.linspace(0.02, 0.42, 110)
    ig = np.linspace(1e-4, 0.93, 350)
    for colour, beta in zip(style.C_SET, [0.4, BSTAR, 1.3], strict=True):
        p = Simplicial(n=par.n, n_tri=par.n_tri, beta=beta, gamma=gam)
        label = rf"$\beta={beta:.2f}$" + (
            r"  $(\beta^*)$" if abs(beta - BSTAR) < 1e-3 else ""
        )
        up, down = _sweep_pair(
            sim_rhs, [1e-6, 0.9], [0.55, 0.35], taus, p, lambda y: y[0]
        )
        _plot_sweeps(a, taus, up, down, colour, label)
        a.plot(
            endemic_branch(
                lambda i, t, p=p: sim_beta_eff(i, sim_fast(i, t, p), t, p) - gam,
                ig,
                hi=3.0,
            ),
            ig,
            "--",
            color=colour,
            lw=1.1,
        )
    a.set_xlim(0, 0.42)
    a.set_ylim(-0.02, 1.1)
    a.legend(frameon=False, loc="upper left")
    a.set_title(
        r"discontinuous for $\beta > \beta^* \approx 0.82$", loc="right", fontsize=10
    )


def row_adaptive(ax: NDArray, gam: float, i0: float) -> None:
    """Adaptive network SIS -- two fast variables, p1 and p2 (Appendix 6.2.3)."""
    par = Adaptive(n=5, w=0.8, gamma=gam)
    tau, T = 1.3 * ad_tau_c(par), 70
    full = solve_ivp(ad_rhs, [0, T], [i0, 1 - i0, i0], args=(tau, par), **IVP)
    red = solve_ivp(ad_rhs_1d, [0, T], [i0], args=(tau, par), **IVP)
    tt = np.linspace(0, T, 900)

    ax[0].plot(tt, full.sol(tt)[0], color=style.C_FULL)
    ax[0].plot(tt, red.sol(tt)[0], "--", color=style.C_REDUCED)

    ir = np.clip(red.sol(tt)[0], 1e-12, 0.999)
    p1r = ad_fast(ir, tau, par)
    ax[1].plot(tt, full.sol(tt)[1], color=style.C_FULL)
    ax[1].plot(tt, p1r, "--", color=style.C_REDUCED)
    ax[1].plot(tt, full.sol(tt)[2], color=style.C_FULL, lw=0.8, alpha=0.75)
    ax[1].plot(
        tt,
        ad_fast_p2(ir, p1r, tau, par),
        "--",
        color=style.C_REDUCED,
        lw=0.8,
        alpha=0.75,
    )
    ax[1].annotate(r"$p_1$", (0.35, 0.55), xycoords="axes fraction", fontsize=10)
    ax[1].annotate(
        r"$p_2$", (0.35, 0.3), xycoords="axes fraction", fontsize=10, alpha=0.7
    )

    a = ax[2]
    taus = np.linspace(0.08, 0.95, 120)
    ig = np.linspace(1e-4, 0.93, 400)
    for colour, w in zip(style.C_SET, [0.5, 1.5, 3.0], strict=True):
        p = Adaptive(n=par.n, w=w, gamma=gam)
        label = rf"$w={w}$" + (r"  $(w^*)$" if abs(w - ad_w_star(p)) < 1e-9 else "")
        up, down = _sweep_pair(
            ad_rhs, [1e-6, 1.0, 1e-6], [0.5, 0.4, 0.5], taus, p, lambda y: y[0]
        )
        _plot_sweeps(a, taus, up, down, colour, label)
        # Closed form here, so no root-finding: Eq. (91) gives tau(i*) directly.
        a.plot(ad_branch(ig, p), ig, "--", color=colour, lw=1.1)
    a.set_xlim(0.08, 0.95)
    a.set_ylim(-0.02, 1.1)
    a.legend(frameon=False, loc="upper left")
    a.set_title(
        r"discontinuous for $w>w^*=\gamma\frac{n+1}{n-1}$", loc="right", fontsize=10
    )


def row_interacting(ax: NDArray, gam: float, i0: float) -> None:
    """Interacting contagions -- discontinuous above alpha* = 2 (Eq. (112))."""
    par = Interacting(alpha=1.5, gamma=gam)
    tau, T = 1.3 * ic_tau_c(par), 45
    full = solve_ivp(ic_rhs, [0, T], [i0, 0.0], args=(tau, par), **IVP)
    red = solve_ivp(ic_rhs_1d, [0, T], [i0], args=(tau, par), **IVP)
    tt = np.linspace(0, T, 900)
    # The full model is integrated in (<SI>, <II>), so prevalence is their sum.
    ifull = full.sol(tt)[0] + full.sol(tt)[1]

    ax[0].plot(tt, ifull, color=style.C_FULL)
    ax[0].plot(tt, red.sol(tt)[0], "--", color=style.C_REDUCED)
    ax[1].plot(tt, full.sol(tt)[0] / ifull, color=style.C_FULL)
    ax[1].plot(
        tt,
        ic_fast(np.clip(red.sol(tt)[0], 1e-12, 0.999), tau, par),
        "--",
        color=style.C_REDUCED,
    )

    a = ax[2]
    taus = np.linspace(0.55, 1.45, 120)
    ig = np.linspace(1e-4, 0.95, 400)
    for colour, alpha in zip(style.C_SET, [1.5, 2.0, 3.0], strict=True):
        p = Interacting(alpha=alpha, gamma=gam)
        label = rf"$\alpha={alpha}$" + (r"  $(\alpha^*)$" if alpha == 2.0 else "")
        up, down = _sweep_pair(
            ic_rhs, [1e-6, 0.0], [0.4, 0.4], taus, p, lambda y: y[0] + y[1]
        )
        _plot_sweeps(a, taus, up, down, colour, label)
        a.plot(
            endemic_branch(
                lambda i, t, p=p: ic_beta_eff(i, ic_fast(i, t, p), t, p) - gam,
                ig,
                lo=0.05,
            ),
            ig,
            "--",
            color=colour,
            lw=1.1,
        )
    a.set_xlim(0.55, 1.45)
    a.set_ylim(-0.02, 1.1)
    a.legend(frameon=False, loc="upper left")
    a.set_title(r"discontinuous for $\alpha>\alpha^* = 2$", loc="right", fontsize=10)


# --------------------------------------------------------------------------- #
#  Figure
# --------------------------------------------------------------------------- #
ROW_TITLES = [
    "Pairwise SIS",
    "Higher-order (simplicial)",
    "Adaptive network SIS",
    "Interacting contagions",
]
ROW_PARAMS = [
    r"$n=5$",
    r"$n=5,\ n_\Delta=3,\ \beta=0.4$",
    r"$n=5,\ w=0.5$",
    r"$\gamma_2 = \gamma = 1,\ \alpha=1.5$",
]


def make_figure(outfile: str = OUTFILE) -> None:
    """Draw and write the figure."""
    style.use_manuscript_style()
    fig, ax = plt.subplots(4, 3, figsize=(9.8, 10.6))

    row_pairwise_sis(ax[0], GAM, I0)
    row_simplicial(ax[1], GAM, I0)
    row_adaptive(ax[2], GAM, I0)
    row_interacting(ax[3], GAM, I0)

    for r in range(4):
        ax[r, 0].set_ylabel("prevalence  $i$")
        ax[r, 1].set_ylabel("fast variable")
        ax[r, 2].set_ylabel(r"endemic  $i^*$")
        ax[r, 1].set_ylim(0, 1.08)
        ax[r, 0].text(
            -0.30,
            1.06,
            ROW_TITLES[r],
            transform=ax[r, 0].transAxes,
            fontsize=10,
            weight="bold",
            va="bottom",
        )
        ax[r, 0].text(
            0.97,
            0.06,
            ROW_PARAMS[r],
            transform=ax[r, 0].transAxes,
            fontsize=10,
            ha="right",
            color=style.C_ANNOTATION,
        )
    for c in range(3):
        ax[3, c].set_xlabel("time" if c < 2 else r"transmission rate  $\tau$")
    style.frame_axes(*ax.ravel())
    ax[0, 0].set_title("(a) prevalence", loc="right", fontsize=10)
    ax[0, 1].set_title("(b) fast variable relaxation", loc="right", fontsize=10)

    fig.tight_layout(h_pad=1.6)
    fig.savefig(outfile + ".pdf")
    fig.savefig(outfile + ".png", dpi=200)


# --------------------------------------------------------------------------- #
#  Numerical audit  (--check)
# --------------------------------------------------------------------------- #
def checks() -> None:
    """Confirm that the reductions reproduce the full models where they must."""
    gam = GAM
    print(
        "Equilibria of the FULL model vs the 1D reduction "
        "(quasi-static approx is exact at steady states):"
    )
    for tau in (0.30, 0.45):
        par = PairwiseSIS(n=5, gamma=gam)
        f = solve_ivp(sis_rhs, [0, 8e3], [0.05, 0.05 * 0.95], args=(tau, par), **SWP).y[
            0, -1
        ]
        r = solve_ivp(sis_rhs_1d, [0, 8e3], [0.05], args=(tau, par), **SWP).y[0, -1]
        print(f"  SIS  tau={tau}: full={f:.12f}  1D={r:.12f}  |diff|={abs(f - r):.1e}")
    for alpha in (1.5, 3.0):
        par = Interacting(alpha=alpha, gamma=gam)
        s = solve_ivp(ic_rhs, [0, 8e3], [0.02, 0.0], args=(1.2, par), **SWP)
        f = s.y[0, -1] + s.y[1, -1]
        r = solve_ivp(ic_rhs_1d, [0, 8e3], [0.02], args=(1.2, par), **SWP).y[0, -1]
        print(f"  IC   a={alpha}: full={f:.12f}  1D={r:.12f}  |diff|={abs(f - r):.1e}")

    print("\nAdaptive: closed-form branch vs quadratic root:")
    for w in (0.5, 3.0):
        par = Adaptive(n=5, w=w, gamma=gam)
        for i in (0.05, 0.6):
            closed = ad_branch(i, par)
            numeric = brentq(
                lambda t, i=i, par=par: t * 5 * ad_fast(i, t, par) - gam, 1e-4, 6.0
            )
            print(
                f"  w={w} i={i}: closed={closed:.10f}  numeric={numeric:.10f}  "
                f"|diff|={abs(closed - numeric):.1e}"
            )

    print("\nSimplicial: bistability boundary in beta (n=5, n_tri=3):")
    print(f"  beta* = {sim_beta_star(Simplicial(n=5, n_tri=3, gamma=gam)):.5f}")


if __name__ == "__main__":
    if "--check" in sys.argv:
        checks()
    else:
        make_figure()
