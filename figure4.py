"""Figure 4 -- pairwise SIR, where the reduction terminates at two dimensions.

Every other model in the paper collapses to a single equation because knowing the
prevalence i fixes the susceptible fraction s = 1 - i.  SIR does not: susceptible
depletion runs on the same timescale as prevalence itself, so s is a second slow
variable and survives the reduction.  What the recipe eliminates here is sigma,
exactly, through the invariant sigma = C s^{2(n-1)/n} of Eq. (130), and then p by
slaving (Eq. (131)) -- leaving a closed 2D system rather than a 1D one.

Panels: (a) prevalence i(t), (b) fast variable p(t), (c) final size r(inf) vs tau.

Three curves are compared throughout: the full pairwise SIR model, the 2D reduction,
and the draft 1D reduction p(i) = (n-2)(1-i)/(n-i), which applies two further
approximations (s -> 1 - i and sigma/s -> 1) on top of the 2D one.  The 1D curve is
shown because its failure is instructive: it produces a stable endemic plateau, which
an SIR epidemic cannot have.

Usage
-----
    uv run figure4.py            write figure4.pdf and figure4.png
    uv run figure4.py --check    numerical audit, no figure
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

import style
from models import (
    PairwiseSIR,
    sir_fast,
    sir_fast_1d,
    sir_initial_state,
    sir_rhs,
    sir_rhs_1d,
    sir_rhs_2d,
    sir_tau_c,
)

OUTFILE = "figure4"
GAM = 1.0
I0 = 1e-6
N_TIME_SERIES = 6  # degree used for panels (a) and (b)
T_END = 60.0
DEGREES = (4, 6, 10)  # degrees swept in panel (c)

# scipy exports no public type for a solve_ivp result.
Solution = Any

IVP = dict(rtol=1e-11, atol=1e-14, method="LSODA")
IVP_FINAL = dict(rtol=1e-10, atol=1e-14, method="LSODA")


# --------------------------------------------------------------------------- #
#  Trajectories
# --------------------------------------------------------------------------- #
def time_series(
    par: PairwiseSIR, tau: float, te: NDArray[np.float64]
) -> tuple[Solution, Solution, Solution, float]:
    """Integrate the full model and both reductions on a common time grid."""
    s0, SI0, SS0, C = sir_initial_state(I0, par)
    T = float(te[-1])
    full = solve_ivp(
        sir_rhs, [0, T], [s0, I0, SI0, SS0], args=(tau, par), t_eval=te, **IVP
    )
    red2 = solve_ivp(sir_rhs_2d, [0, T], [I0, s0], args=(tau, par, C), t_eval=te, **IVP)
    red1 = solve_ivp(sir_rhs_1d, [0, T], [I0], args=(tau, par), t_eval=te, **IVP)
    return full, red2, red1, C


def _extinction_event(index: int) -> Callable[..., float]:
    """Terminal event: prevalence, held at state ``index``, falls below 1e-11.

    SIR epidemics end rather than settle, so the final size is read off once the
    infected class has effectively emptied.
    """

    def event(t: float, y: list[float], *args: object) -> float:
        return y[index] - 1e-11

    event.terminal, event.direction = True, -1
    return event


def final_size(tau: float, par: PairwiseSIR, reduced: bool = False) -> float:
    """Final size r(inf) = 1 - s(inf) - i(inf), from the full model or the 2D one."""
    s0, SI0, SS0, C = sir_initial_state(I0, par)
    if not reduced:
        sol = solve_ivp(
            sir_rhs,
            [0, 5e4],
            [s0, I0, SI0, SS0],
            args=(tau, par),
            events=_extinction_event(1),
            **IVP_FINAL,
        )
        return 1.0 - sol.y[0, -1] - sol.y[1, -1]
    sol = solve_ivp(
        sir_rhs_2d,
        [0, 5e4],
        [I0, s0],
        args=(tau, par, C),
        events=_extinction_event(0),
        **IVP_FINAL,
    )
    return 1.0 - sol.y[1, -1] - sol.y[0, -1]


# --------------------------------------------------------------------------- #
#  Panels
# --------------------------------------------------------------------------- #
def panel_prevalence(
    ax: plt.Axes, par: PairwiseSIR, full: Solution, red2: Solution, red1: Solution
) -> None:
    """(a) prevalence i(t) for the full model and both reductions."""
    ax.plot(full.t, full.y[1], color=style.C_FULL, lw=2.0, label="full pairwise SIR")
    ax.plot(
        red2.t, red2.y[0], color=style.C_REDUCED, lw=1.6, ls="--", label="2D reduction"
    )
    ax.plot(red1.t, red1.y[0], color=style.C_GOLD, lw=1.6, ls=":", label="1D reduction")
    ax.set_xlabel("time")
    ax.set_ylabel(r"prevalence  $i$")
    ax.set_title("(a) prevalence", fontsize=10, loc="left")
    ax.legend(loc="upper left", fontsize=10)
    ax.text(
        0.97,
        1.03,
        rf"$n={par.n}$,  $\tau=1.3\tau_c$",
        transform=ax.transAxes,
        ha="right",
        color=style.C_ANNOTATION,
        fontsize=10,
    )


def panel_fast_variable(
    ax: plt.Axes,
    par: PairwiseSIR,
    full: Solution,
    red2: Solution,
    red1: Solution,
    C: float,
) -> None:
    """(b) the fast variable p(t), with its disease-free value marked."""
    p_full = full.y[2] / np.maximum(full.y[1], 1e-300)
    p_red2 = [sir_fast(i, s, C, par) for i, s in zip(red2.y[0], red2.y[1], strict=True)]

    ax.plot(full.t, p_full, color=style.C_FULL, lw=2.0)
    ax.plot(red2.t, p_red2, color=style.C_REDUCED, lw=1.6, ls="--")
    ax.plot(red1.t, sir_fast_1d(red1.y[0], par), color=style.C_GOLD, lw=1.6, ls=":")

    # p*(0) = (n-2)/n, Eq. (127): an infected node has one infected neighbour, the
    # one that infected it, and at most n-1 susceptible ones.
    ax.axhline((par.n - 2) / par.n, color=style.C_REFERENCE, lw=0.7, ls="-.")
    ax.text(
        T_END * 0.98,
        (par.n - 2) / par.n + 0.02,
        r"$p^*(0)=(n-2)/n$",
        ha="right",
        color="0.45",
        fontsize=10,
    )
    ax.set_xlabel("time")
    ax.set_ylabel(r"fast variable  $p$")
    ax.set_ylim(0, 1.05)
    ax.set_title("(b) fast variable", fontsize=10, loc="left")


def panel_final_size(ax: plt.Axes, gam: float) -> None:
    """(c) final size against tau, full model (solid) vs 2D reduction (dashed)."""
    for colour, n in zip(style.C_SET, DEGREES, strict=True):
        par = PairwiseSIR(n=n, gamma=gam)
        tc = sir_tau_c(par)
        taus = np.linspace(0.6 * tc, 3.2 * tc, 34)
        ax.plot(
            taus,
            [final_size(t, par) for t in taus],
            color=colour,
            lw=1.8,
            label=rf"$n={n}$",
        )
        ax.plot(
            taus,
            [final_size(t, par, reduced=True) for t in taus],
            color=colour,
            lw=1.4,
            ls="--",
        )
        ax.axvline(tc, color=colour, lw=0.7, ls=":", alpha=0.8)

    ax.set_xlabel(r"transmission rate  $\tau$")
    ax.set_ylabel(r"final size  $r(\infty)$")
    ax.set_title(r"(c) final size; dashed = 2D reduction", fontsize=10, loc="left")
    ax.set_xlim(0, 1.45)
    ax.set_ylim(-0.02, 1.0)
    ax.legend(loc="lower right", fontsize=10)
    # Right-anchored, like the other figures' in-axes annotations, so it cannot
    # overrun the axes when font metrics or layout shift.
    ax.text(
        0.985,
        0.4,
        r"dotted: $\tau=\gamma/(n-2)$",
        transform=ax.transAxes,
        ha="right",
        fontsize=10,
        color="0.4",
    )


# --------------------------------------------------------------------------- #
#  Figure
# --------------------------------------------------------------------------- #
def make_figure(outfile: str = OUTFILE) -> None:
    """Draw and write the figure."""
    style.use_manuscript_style()
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.1))

    par = PairwiseSIR(n=N_TIME_SERIES, gamma=GAM)
    tau = 1.3 * sir_tau_c(par)
    te = np.linspace(0, T_END, 1200)
    full, red2, red1, C = time_series(par, tau, te)

    panel_prevalence(ax[0], par, full, red2, red1)
    panel_fast_variable(ax[1], par, full, red2, red1, C)
    panel_final_size(ax[2], GAM)

    style.frame_axes(*ax)
    fig.tight_layout()
    fig.savefig('./figures/' + outfile + ".pdf", bbox_inches="tight")
    fig.savefig('./figures/' + outfile + ".png", dpi=200, bbox_inches="tight")


# --------------------------------------------------------------------------- #
#  Numerical audit  (--check)
# --------------------------------------------------------------------------- #
def checks() -> None:
    """How far each reduction sits from the full model, at the peak and at the end."""
    par = PairwiseSIR(n=N_TIME_SERIES, gamma=GAM)
    tau = 1.3 * sir_tau_c(par)
    te = np.linspace(0, T_END, 1200)
    full, red2, red1, _ = time_series(par, tau, te)

    print(f"n={par.n}, tau={tau:.4f}")
    peak = np.argmax(full.y[1])
    print(
        f"  peak i : full={full.y[1].max():.4f}  2D={red2.y[0].max():.4f}  "
        f"1D={red1.y[0].max():.4f}"
    )

    pre = full.t <= full.t[peak]
    err1 = np.abs(red1.y[0][pre] - full.y[1][pre]) / np.maximum(full.y[1][pre], 1e-12)
    err2 = np.abs(red2.y[0][pre] - full.y[1][pre]) / np.maximum(full.y[1][pre], 1e-12)
    print(
        f"  pre-peak rel. err  1D: {100 * err1.mean():.0f}% "
        f"(max {100 * err1.max():.0f}%)"
        f" | 2D: {100 * err2.mean():.1f}% (max {100 * err2.max():.1f}%)"
    )
    print(
        "  1D reduction endemic plateau i(T) =",
        f"{red1.y[0][-1]:.4f}",
        "(full model i(T) =",
        f"{full.y[1][-1]:.2e})",
    )

    print("\nfinal size r(inf): full vs 2D reduction")
    for n in DEGREES:
        p = PairwiseSIR(n=n, gamma=GAM)
        tc = sir_tau_c(p)
        for m in (1.5, 2.0, 3.0):
            a, b = final_size(m * tc, p), final_size(m * tc, p, reduced=True)
            print(
                f"  n={n:2d} tau={m:.1f}tau_c : full={a:.4f}  2D={b:.4f}"
                f"  err={100 * abs(b - a) / a:5.1f}%"
            )


if __name__ == "__main__":
    if "--check" in sys.argv:
        checks()
    else:
        make_figure()
