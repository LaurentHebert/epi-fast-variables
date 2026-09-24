"""Figure 3 -- the effective force of infection measured from trajectory data.

Where the quasi-static p(i) has no closed form, it can be measured instead: run the
full model once, discard the initial fast transient, and fit p as a polynomial in i.
Feeding that p(i) back through the model's own beta_eff closes a 1D reduction,
di/dt = i (beta_eff(i) - gamma).  This is the four-step numerical procedure of
Section 4, Eq. (7).

Four panels: pairwise SIS | higher-order (simplicial) | adaptive | interacting.
Main axes : p against i -- trajectory data, the analytic quasi-static p*(i), and
            constant / linear / quadratic fits.
Inset     : i(t) -- the full model against the 1D reduction built from each fit.

Fitting p(i) rather than di/dt as a function of i is the essential choice: p is the
quantity that equilibrates quickly, and its fitted constant and linear coefficients
are exactly the two numbers that fix the threshold and the transition type.

Note on the interacting-contagions panel: that model has an exact invariant,
<II> = <I>^2 (statistical independence of the two diseases), so the trajectory obeys
p = 1 - i exactly and the linear fit is exact to machine precision.  The quasi-static
root of the Step-3 quadratic is a different curve, which the trajectory does not
follow away from threshold; Figure 1 --invariant draws the two against each other.

Usage
-----
    uv run figure3.py              write figure3.pdf and figure3.png
    uv run figure3.py --verbose    also report where each transient was cut and
                                   the rms residual of each polynomial fit
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

import style
from models import (
    Adaptive,
    Interacting,
    ModelParams,
    PairwiseSIS,
    Simplicial,
    ad_rhs_1d,
    ic_rhs_1d,
    sim_rhs_1d,
    sis_rhs_1d,
    ad_beta_eff,
    ad_fast,
    ad_rhs,
    ad_tau_c,
    ic_beta_eff,
    ic_fast,
    ic_rhs,
    ic_tau_c,
    sim_beta_eff,
    sim_fast_grid,
    sim_rhs,
    sim_tau_c,
    sis_beta_eff,
    sis_fast,
    sis_rhs,
    sis_tau_c,
)

OUTFILE = "figure3"
GAM = 1.0
N = 5

# Numerical step 1 of Section 4: seed low, so the transient is over while i is still
# around 1e-5 and the low-prevalence region -- the one that decides the threshold and
# the transition type -- stays rich in data.
I0 = 1e-6

# Numerical step 2: keep the trajectory once |dp/dt| < KAPPA |di/dt|.  Crude, but the
# result is insensitive to it precisely because the timescales separate.
KAPPA = 5.0
ON_MANIFOLD_SAMPLES = 50

IVP = dict(rtol=1e-11, atol=1e-14, method="LSODA", dense_output=True)

# Fit orders that are drawn, with their line styles.  The cubic is fitted and
# reported but not drawn: by then the curves overlie the quadratic.  Colour index 2
# of the palette is deliberately skipped, so that adjacent orders stay
# distinguishable where they nearly coincide.
FIT_ORDERS = (0, 1, 2, 3)
DRAWN_ORDERS = (0, 1, 2)
FIT_LABELS = ("constant", "linear", "quadratic", "cubic")
FIT_LINESTYLE = {0: ":", 1: "-.", 2: "--"}
FIT_WIDTH_MAIN = {0: 1.15, 1: 2.0, 2: 1.15}
FIT_WIDTH_INSET = {0: 1.15, 1: 3.0, 2: 2.0}
FIT_COLOUR = {0: style.C_FIT_ORDER[0], 1: style.C_FIT_ORDER[1], 2: style.C_FIT_ORDER[3]}


@dataclass(frozen=True, slots=True)
class Panel:
    """Everything one panel needs: a model to run, and how to read p and i off it.

    ``i_of`` and ``p_of`` map the raw solution array to prevalence and fast variable,
    which differ per model -- the interacting-contagion system, for instance, is
    integrated in (<SI>, <II>), so prevalence is their sum.
    """

    title: str
    params_label: str
    rhs: Callable[..., list[float]]
    y0: list[float]
    par: ModelParams
    tau: float
    T: float
    i_of: Callable[[NDArray[np.float64]], NDArray[np.float64]]
    p_of: Callable[[NDArray[np.float64]], NDArray[np.float64]]
    quasi_static: Callable[[NDArray[np.float64]], NDArray[np.float64]]
    beta_eff: Callable[[float, float], float]
    rhs_1d: Callable[..., list[float]] | None = None
    fast_label: str = r"quasi-static $p^*(i)$"

    @property
    def args(self) -> tuple[float, ModelParams]:
        """Extra arguments every ``solve_ivp`` call on this panel needs."""
        return (self.tau, self.par)


def build_panels(gam: float = GAM) -> list[Panel]:
    """The four panels, each at 1.3 times its own threshold."""
    sis = PairwiseSIS(n=N, gamma=gam)
    tau_sis = 1.3 * sis_tau_c(sis)

    sim = Simplicial(n=N, n_tri=3, beta=1.3, gamma=gam)
    tau_sim = 1.3 * sim_tau_c(sim)

    ad = Adaptive(n=N, w=3.0, gamma=gam)
    tau_ad = 1.3 * ad_tau_c(ad)

    ic = Interacting(alpha=3.0, gamma=gam)
    tau_ic = 1.3 * ic_tau_c(ic)

    return [
        Panel(
            title="Pairwise SIS",
            params_label=rf"$n={N}$",
            rhs=sis_rhs,
            rhs_1d=sis_rhs_1d,
            y0=[I0, (1 - I0) * I0],
            par=sis,
            tau=tau_sis,
            T=140,
            i_of=lambda Y: Y[0],
            p_of=lambda Y: Y[1] / Y[0],
            quasi_static=lambda i: sis_fast(i, tau_sis, sis),
            beta_eff=lambda i, p: sis_beta_eff(i, p, tau_sis, sis),
        ),
        Panel(
            title="Higher-order (simplicial)",
            params_label=rf"$n={N},\ n_\Delta={sim.n_tri},\ \beta={sim.beta}$",
            rhs=sim_rhs,
            rhs_1d=sim_rhs_1d,
            y0=[I0, 1 - I0],
            par=sim,
            tau=tau_sim,
            T=170,
            i_of=lambda Y: Y[0],
            p_of=lambda Y: Y[1],
            quasi_static=lambda i: sim_fast_grid(i, tau_sim, sim),
            beta_eff=lambda i, p: sim_beta_eff(i, p, tau_sim, sim),
        ),
        Panel(
            title="Adaptive network SIS",
            params_label=rf"$n={N},\ w={ad.w}$",
            rhs=ad_rhs,
            rhs_1d=ad_rhs_1d,
            y0=[I0, 1 - I0, I0],
            par=ad,
            tau=tau_ad,
            T=170,
            i_of=lambda Y: Y[0],
            p_of=lambda Y: Y[1],
            quasi_static=lambda i: ad_fast(i, tau_ad, ad),
            beta_eff=lambda i, p: ad_beta_eff(i, p, tau_ad, ad),
            fast_label=r"quasi-static $p_1^*(i)$",
        ),
        Panel(
            title="Interacting contagions",
            params_label=rf"$\alpha={ic.alpha}$",
            rhs=ic_rhs,
            rhs_1d=ic_rhs_1d,
            y0=[I0, 0.0],
            par=ic,
            tau=tau_ic,
            T=100,
            i_of=lambda Y: Y[0] + Y[1],
            p_of=lambda Y: Y[0] / (Y[0] + Y[1]),
            quasi_static=lambda i: ic_fast(i, tau_ic, ic),
            beta_eff=lambda i, p: ic_beta_eff(i, p, tau_ic, ic),
        ),
    ]


# --------------------------------------------------------------------------- #
#  Trajectory -> (i, p) data
# --------------------------------------------------------------------------- #
def extract_manifold(
    panel: Panel, nsamp: int = 4000
) -> tuple[NDArray[np.float64], NDArray[np.float64], tuple[float, float]]:
    """Integrate the full model, drop the fast transient, resample uniformly in i.

    Resampling in i rather than in t matters: raw time samples over-weight the
    endpoints, where the trajectory dwells, and the fit would follow the plateau
    rather than the low-prevalence region that carries the bifurcation information.

    Returns the prevalence grid, p on it, and where the transient was judged to end.
    """
    sol = solve_ivp(panel.rhs, [0, panel.T], panel.y0, args=panel.args, **IVP)
    tt = np.linspace(0, panel.T, 40000)
    Y = sol.sol(tt)
    i, p = panel.i_of(Y), panel.p_of(Y)

    di, dp = np.gradient(i, tt), np.gradient(p, tt)
    on_manifold = np.abs(dp) < KAPPA * np.abs(di)
    window = np.ones(ON_MANIFOLD_SAMPLES) / ON_MANIFOLD_SAMPLES
    k0 = int(np.argmax(np.convolve(on_manifold.astype(float), window, "same") > 0.99))

    i, p, tt = i[k0:], p[k0:], tt[k0:]
    rising = np.r_[True, np.diff(i) > 0]
    i, p = i[rising], p[rising]
    igrid = np.linspace(i[0], i[-1], nsamp)
    return igrid, np.interp(igrid, i, p), (tt[0], i[0])


def rms(
    i: NDArray[np.float64], p: NDArray[np.float64], coef: NDArray[np.float64]
) -> float:
    """Root-mean-square residual of a polynomial fit to the (i, p) data."""
    return float(np.sqrt(np.mean((np.polyval(coef, i) - p) ** 2)))


# --------------------------------------------------------------------------- #
#  Fitted 1D reductions
# --------------------------------------------------------------------------- #
def reduced_rhs(panel: Panel, coef: NDArray[np.float64]) -> Callable[..., list[float]]:
    """1D reduction with p(i) taken from the fit and beta_eff from the model itself.

    This is numerical step 4 of Section 4.  Prevalence is held inside [0, 0.995] so
    that the integration stays clear of the 1/(1 - i) pole in the simplicial
    beta_eff; see :func:`cap_event` for the companion guard.
    """
    gam = panel.par.gamma

    def p_fit(i: float) -> float:
        return float(np.clip(np.polyval(coef, i), 0.0, 1.0))

    def rhs(t: float, y: list[float]) -> list[float]:
        i = min(max(float(y[0]), 0.0), 0.995)
        return [i * (panel.beta_eff(i, p_fit(i)) - gam)]

    return rhs


def cap_event(icap: float) -> Callable[..., float]:
    """Terminal event: stop as soon as a fitted reduction leaves the plotted range.

    The constant fit has no saturating term -- for the SIS-type models the reduced
    equation is di/dt = i(tau n a0 - gamma), unbounded exponential growth.  Left
    alone the solver chases i upward, stalls on zero-length steps, and raises
    "`ts` must be strictly increasing or decreasing".  The runaway is a genuine
    feature of the constant fit, and the reason it fails, so it is integrated safely
    rather than suppressed.  Stopping just above the top of the inset costs nothing
    visually.
    """

    def event(t: float, y: list[float]) -> float:
        return y[0] - icap

    event.terminal = True
    event.direction = 1
    return event


# --------------------------------------------------------------------------- #
#  Drawing
# --------------------------------------------------------------------------- #
def draw_panel(ax: plt.Axes, panel: Panel) -> tuple[tuple[float, float], dict]:
    """Draw one panel and its inset; return the transient cut and the fits."""
    igrid, pgrid, transient = extract_manifold(panel)
    fits = {d: np.polyfit(igrid, pgrid, d) for d in FIT_ORDERS}

    ax.plot(
        igrid[::45],
        pgrid[::45],
        "o",
        ms=5.0,
        mfc="none",
        mew=0.7,
        color="0.4",
        label="full model",
        zorder=2,
    )
    ax.plot(
        igrid,
        panel.quasi_static(igrid),
        "-",
        color="k",
        lw=1.7,
        alpha=0.85,
        label=panel.fast_label,
        zorder=3,
    )
    for d in DRAWN_ORDERS:
        ax.plot(
            igrid,
            np.polyval(fits[d], igrid),
            FIT_LINESTYLE[d],
            lw=FIT_WIDTH_MAIN[d],
            color=FIT_COLOUR[d],
            label=FIT_LABELS[d],
            zorder=4,
        )

    ax.set_xlabel("prevalence  $i$")
    ax.set_ylabel(r"fast variable  $p$")
    ax.set_title(panel.title, loc="left", fontsize=10.5, weight="bold")
    ax.text(
        0.985,
        1.055,
        panel.params_label,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        color=style.C_ANNOTATION,
    )
    ax.legend(
        frameon=False,
        loc="lower left",
        fontsize=10,
        handlelength=1.7,
        borderpad=0.2,
        labelspacing=0.3,
    )
    ax.set_xlim(0, igrid[-1] * 1.1)
    style.frame_axes(ax)

    axi = ax.inset_axes([0.615, 0.55, 0.355, 0.40])
    itop = 1.35 * igrid[-1]   # top of the inset (same value as in set_ylim)
    icap = 1.05 * itop        # stop just past the frame
    full = solve_ivp(panel.rhs, [0, panel.T], panel.y0, args=panel.args, **IVP)
    tt = np.linspace(0, panel.T, 900)
    tm = np.linspace(0, panel.T, 30)
    axi.plot(tm, panel.i_of(full.sol(tm)), "o", ms=3.0, mfc="none", mew=0.6,
             color="0.4", zorder=5)
    qs = solve_ivp(panel.rhs_1d, [0, panel.T], [I0], args=panel.args, t_eval=tt,
                   rtol=1e-9, atol=1e-12)
    axi.plot(qs.t, qs.y[0], "-", color="k", lw=0.9, zorder=6)
    for d in DRAWN_ORDERS:
        red = solve_ivp(
            reduced_rhs(panel, fits[d]),
            [0, panel.T],
            [I0],
            t_eval=tt,
            rtol=1e-9,
            atol=1e-12,
            method="LSODA",
            events=cap_event(icap),
        )
        t_red, i_red = red.t, red.y[0]
        if red.t_events[0].size:  # cap reached: add the exact crossing point
            t_red = np.append(t_red, red.t_events[0][0])
            i_red = np.append(i_red, red.y_events[0][0, 0])
        axi.plot(
            t_red,
            i_red,
            FIT_LINESTYLE[d],
            lw=FIT_WIDTH_INSET[d],
            color=FIT_COLOUR[d],
        )
    # The constant fit may leave the top of the inset; that is the point of it.
    axi.set_ylim(-0.03 * igrid[-1], 1.35 * igrid[-1])
    axi.set_xlabel("time", fontsize=10, labelpad=1)
    axi.set_ylabel("prevalence $i$", fontsize=10, labelpad=1)
    axi.tick_params(labelsize=10, length=2)
    current_yticks = axi.get_yticks()
    axi.set_yticks([y for y in current_yticks if y != 0])
    bottom, top = axi.get_ylim()
    axi.set_ylim([-0.05,top])

    return transient, {"igrid": igrid, "pgrid": pgrid, "fits": fits}


def report_fit_quality(
    panel: Panel, transient: tuple[float, float], data: dict
) -> None:
    """Report where the transient was cut and how well each polynomial fits."""
    t_rel, i_rel = transient
    igrid, pgrid, fits = data["igrid"], data["pgrid"], data["fits"]
    residuals = "  ".join(
        f"{FIT_LABELS[d][:5]}={rms(igrid, pgrid, fits[d]):.1e}" for d in FIT_ORDERS
    )
    print(
        f"{panel.title:26s} transient ends t={t_rel:5.2f} "
        f"(i={i_rel:.1e})  |  {residuals}"
    )


def make_figure(outfile: str = OUTFILE, verbose: bool = False) -> None:
    """Draw and write the figure."""
    style.use_manuscript_style()
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.8))

    for ax, panel in zip(axes.ravel(), build_panels(), strict=True):
        transient, data = draw_panel(ax, panel)
        if verbose:
            report_fit_quality(panel, transient, data)

    fig.tight_layout()
    fig.savefig('./figures/' + outfile + ".pdf")
    fig.savefig('./figures/' + outfile + ".png", dpi=200)


if __name__ == "__main__":
    make_figure(verbose="--verbose" in sys.argv)
