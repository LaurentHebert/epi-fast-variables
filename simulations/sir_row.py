"""Figure 5, pairwise SIR row: simulations against the 2D reduction and its fit.

Reads the CSV written by ``sir_nrm.py`` and draws (a) prevalence i(t), (b) the fast
variable p(t) and (c) p against i along the epidemic, for pairwise SIR on an n-regular
graph.  Unlike the endemic models, s is a second slow variable (Figure 4), so p depends
on (i, s): the reduction is 2D, and panel (c) shows a path, not a single curve.

Only outbreaks (peak of at least OUTBREAK_MIN infected nodes) are averaged.  The fit is
a bivariate quadratic p(i, s) over the pooled run-times of individual runs with at least
POOL_I_MIN infected, fed through the 2D reduction; the analytic curve is slaved through
the exact invariant sigma = C s^{2(n-1)/n} (Eqs. (130)-(131)).  The reductions are
integrated from the simulations' seeding, because C depends on it; when runs are
aligned in time, each curve is shifted so that it crosses the threshold at t = 0.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

import style
from models import PairwiseSIR, sir_fast, sir_initial_state, sir_rhs_2d, sir_tau_c
from simulations.common import RowInfo, align_to_prevalence, load_runs, transient_end

OUTBREAK_MIN = 100  # peak infected nodes for a run to count as an outbreak
POOL_I_MIN = 30  # pooled fit uses run-times with at least this many infected
SHOW_I_MIN = 5e-4  # mean points drawn only where mean prevalence is at least this
N_FAINT = 30

C_SIM, C_ANALYTIC, C_FIT = style.C_FULL, style.C_REDUCED, style.C_SET[1]

PAR = PairwiseSIR(n=6, gamma=1.0)
TAU = 1.3 * sir_tau_c(PAR)


def features(i, s) -> NDArray:
    """Bivariate quadratic basis in (i, s)."""
    i, s = np.asarray(i, float), np.asarray(s, float)
    return np.stack([np.ones_like(i), i, s, i * i, i * s, s * s], axis=-1)


def fit_p(i: NDArray, s: NDArray, p: NDArray) -> NDArray:
    """Least-squares bivariate quadratic p(i, s)."""
    return np.linalg.lstsq(features(i, s), p, rcond=None)[0]


def reduced_rhs_2d(coef: NDArray):
    """2D reduction closed with the fitted p(i, s)."""
    n, gam = PAR.n, PAR.gamma

    def rhs(t, y):
        i, s = max(float(y[0]), 0.0), float(y[1])
        p = float(np.clip(features(i, s) @ coef, 0.0, 1.0))
        return [i * (TAU * n * p - gam), -TAU * n * p * i]

    return rhs


def draw(ax, csvfile: str, align: float | None = None) -> RowInfo:
    """Draw the three panels of the row onto ``ax`` (three axes)."""
    n = PAR.n
    _, d, N = load_runs(csvfile, ("S", "I", "SI"))
    S, n_inf, SI = d["S"], d["I"], d["SI"]
    i0, t_end = n_inf[:, 0].mean() / N, n_inf.shape[1] - 1
    if align is not None:  # t = 0 when a run reaches prevalence `align`
        _, n_inf, S, SI = align_to_prevalence(
            np.arange(n_inf.shape[1]), n_inf, N, align, S, SI
        )
    ntimes = n_inf.shape[1]
    t = np.arange(ntimes)
    out = n_inf.max(1) >= OUTBREAK_MIN
    So, Io, SIo = S[out], n_inf[out], SI[out]

    i_mean = Io.mean(0) / N
    with np.errstate(invalid="ignore", divide="ignore"):
        p_mean = SIo.mean(0) / (n * Io.mean(0))
    ok = (Io.mean(0) > 0) & (i_mean >= SHOW_I_MIN)  # where mean points are drawn
    k0 = transient_end(i_mean, np.nan_to_num(p_mean))
    pool = (Io >= POOL_I_MIN) & (t[None, :] >= k0)
    coef = fit_p(
        (Io / N)[pool], (So / N)[pool], (SIo / (n * np.where(Io > 0, Io, 1)))[pool]
    )

    # the two reductions, integrated from the seeding and shifted to cross `align`
    C = sir_initial_state(i0, PAR)[3]
    tt = np.linspace(0, t[-1], 900)
    ivp = dict(rtol=1e-9, atol=1e-12, method="LSODA", dense_output=True)
    ana = solve_ivp(sir_rhs_2d, [0, t_end], [i0, 1 - i0], args=(TAU, PAR, C), **ivp)
    fit = solve_ivp(reduced_rhs_2d(coef), [0, t_end], [i0, 1 - i0], **ivp)

    def aligned(sol):
        if align is None:
            return sol.sol(tt)
        dense = np.linspace(0, t_end, 6000)
        tc = dense[np.argmax(sol.sol(dense)[0] >= align)]
        return sol.sol(np.clip(tt + tc, 0, t_end))

    i_ana, s_ana = aligned(ana)
    i_fit, s_fit = aligned(fit)
    p_ana = np.array(
        [sir_fast(a, b, C, PAR) for a, b in zip(i_ana, s_ana, strict=True)]
    )
    p_fit = np.clip(features(i_fit, s_fit) @ coef, 0, 1)

    # (a) prevalence and (b) fast variable against time
    faint = dict(color=C_SIM, lw=0.5, alpha=0.12, zorder=1)
    for r in np.flatnonzero(out)[:N_FAINT]:
        big = n_inf[r] >= POOL_I_MIN
        ax[0].plot(t, n_inf[r] / N, **faint)
        ax[1].plot(t[big], (SI[r] / (n * np.maximum(n_inf[r], 1)))[big], **faint)
    ax[0].plot(t, i_mean, "o", ms=3.0, mfc="none", mew=0.6, color=C_SIM, zorder=3)
    ax[1].plot(
        t[ok], p_mean[ok], "o", ms=3.0, mfc="none", mew=0.6, color=C_SIM, zorder=3
    )
    for a, y_ana, y_fit in ((ax[0], i_ana, i_fit), (ax[1], p_ana, p_fit)):
        a.plot(tt, y_ana, "--", color=C_ANALYTIC, lw=1.5, zorder=4)
        a.plot(tt, y_fit, "-.", color=C_FIT, lw=1.5, zorder=4)
    xlabel = "time" if align is None else f"time since $i={align:g}$"
    ax[0].set_xlim(0, 60)
    ax[0].set_ylim(-0.003, 0.075)
    ax[0].set_xlabel(xlabel)
    ax[0].set_ylabel("prevalence  $i$")
    ax[1].set_xlim(0, 60)
    ax[1].set_ylim(0, 1.08)
    ax[1].set_xlabel(xlabel)
    ax[1].set_ylabel("fast variable  $p$")

    # (c) fast variable against prevalence along the epidemic
    a = ax[2]
    a.plot(i_mean[: k0 + 1], p_mean[: k0 + 1], "-", color="0.75", lw=1.0, zorder=1)
    m = np.zeros(ntimes, bool)
    m[k0:] = ok[k0:]
    show = i_ana >= SHOW_I_MIN / 5
    a.plot(
        i_mean[m], p_mean[m], "o", ms=3.5, mfc="none", mew=0.6, color=C_SIM, zorder=3
    )
    a.plot(i_ana[show], p_ana[show], "--", color=C_ANALYTIC, lw=1.5, zorder=4)
    a.plot(i_fit[show], p_fit[show], "-.", color=C_FIT, lw=1.5, zorder=4)
    a.set_xlim(0, 0.045)
    a.set_xlabel("prevalence  $i$")
    a.set_ylabel("fast variable  $p$")

    resid = np.sqrt(
        np.mean(
            (
                features((Io / N)[pool], (So / N)[pool]) @ coef
                - (SIo / (n * np.where(Io > 0, Io, 1)))[pool]
            )
            ** 2
        )
    )
    return RowInfo(
        label=rf"$\tau={TAU},\ n={n}$",
        notes={
            "runs": n_inf.shape[0],
            "outbreaks": int(out.sum()),
            "N": N,
            "transient_end_t": float(t[k0]),
            "pooled_points": int(pool.sum()),
            "fit_rms": float(resid),
            "peak_i_sim": float((Io.max(1) / N).mean()),
            "peak_i_2d": float(i_ana.max()),
            "peak_i_fit": float(i_fit.max()),
        },
    )
