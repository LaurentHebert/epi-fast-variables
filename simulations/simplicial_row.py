"""Figure 5, higher-order SIS row: simulations against the 1D reduction and its fit.

Reads the CSV written by ``simplicial_nrm.py`` (simplicial SIS on a regular simplicial
complex) and draws (a) prevalence i(t), (b) the fast variable p(t) and (c) p against i.
The fast variable is p = <SI>/<I> with the per-stub density <SI> = SI/(kN), i.e.
p = SI / (k I), k the total pairwise degree.

Edges of triangles are also pairwise edges (inclusion property), so a node with n edges
outside triangles and n_tri triangles has k = n + 2 n_tri pairwise neighbours; the
model is evaluated with that k.  ``degree_factor`` multiplies n and n_tri of Figure 3
(n = 5, n_tri = 3) and must match the runs; tau = 1.3 tau_c of the model.  Means over
runs are ratios of means over survivors; the manifold is measured from pooled
individual runs, fitted with a quadratic and fed through beta_eff (Eq. (44)).  The
model starts from the measured i0 and p0, since seeding on triangles makes p0 differ
from 1 - i0.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.integrate import solve_ivp

import figure3 as f3
import style
from models import Simplicial, sim_beta_eff, sim_fast_grid, sim_rhs_1d, sim_tau_c
from simulations.common import (
    RowInfo,
    align_to_prevalence,
    load_runs,
    pooled_manifold,
    transient_end,
)

TAU_FACTOR = 1.3  # tau in units of the threshold, as in Figure 3
N_FAINT = 30  # individual surviving runs drawn faintly
SHOW_P_I_MIN = (
    30  # faint p lines drawn only while a run has at least this many infected
)

C_SIM, C_ANALYTIC, C_FIT = style.C_FULL, style.C_REDUCED, style.C_SET[1]


def draw(
    ax, csvfile: str, align: float | None = None, degree_factor: int = 1
) -> RowInfo:
    """Draw the three panels of the row onto ``ax`` (three axes)."""
    panel = f3.build_panels()[1]  # higher-order SIS (beta = 1.3 of Figure 3)
    n_extra, n_tri = panel.par.n * degree_factor, panel.par.n_tri * degree_factor
    par = Simplicial(
        n=n_extra + 2 * n_tri, n_tri=n_tri, beta=panel.par.beta, gamma=panel.par.gamma
    )
    n = par.n
    tau = TAU_FACTOR * sim_tau_c(par)
    panel = replace(
        panel,
        par=par,
        tau=tau,
        quasi_static=lambda i: sim_fast_grid(i, tau, par),
        beta_eff=lambda i, p: sim_beta_eff(i, p, tau, par),
    )

    t, d, N = load_runs(csvfile, ("I", "SI"))
    n_inf, SI = d["I"], d["SI"]
    if align is not None:  # t = 0 when a run reaches prevalence `align`
        t, n_inf, SI = align_to_prevalence(t, n_inf, N, align, SI)
    nruns = n_inf.shape[0]
    surv = n_inf[:, -1] > 0
    Is, SIs = n_inf[surv], SI[surv]

    i_mean = Is.mean(0) / N
    p_mean = SIs.mean(0) / (n * Is.mean(0))
    k0 = transient_end(i_mean, p_mean)
    i_man, p_man = pooled_manifold(Is, SIs, N, n, t, k0)
    coef = np.polyfit(i_man, p_man, 2)

    i0, tmax = i_mean[0], t[-1]
    tt = np.linspace(0, tmax, 900)
    ivp = dict(t_eval=tt, rtol=1e-9, atol=1e-12, method="LSODA")
    ana = solve_ivp(sim_rhs_1d, [0, tmax], [i0], args=panel.args, **ivp)
    fit = solve_ivp(
        f3.reduced_rhs(panel, coef), [0, tmax], [i0], events=f3.cap_event(1.05), **ivp
    )
    i_ana, i_fit = ana.y[0], fit.y[0]
    p_ana = sim_fast_grid(np.clip(i_ana, 1e-12, 0.999), tau, par)
    p_fit = np.clip(np.polyval(coef, i_fit), 0, 1)

    faint = dict(color=C_SIM, lw=0.5, alpha=0.12, zorder=1)
    for r in np.flatnonzero(surv)[:N_FAINT]:
        ax[0].plot(t, n_inf[r] / N, **faint)
        m = n_inf[r] >= SHOW_P_I_MIN
        ax[1].plot(t[m], (SI[r] / (n * np.maximum(n_inf[r], 1)))[m], **faint)
    ax[0].plot(t, i_mean, "o", ms=3.0, mfc="none", mew=0.6, color=C_SIM, zorder=3)
    ax[1].plot(t, p_mean, "o", ms=3.0, mfc="none", mew=0.6, color=C_SIM, zorder=3)
    for a, y_ana, y_fit in ((ax[0], i_ana, i_fit), (ax[1], p_ana, p_fit)):
        a.plot(tt, y_ana, "--", color=C_ANALYTIC, lw=1.5, zorder=4)
        a.plot(fit.t, y_fit, "-.", color=C_FIT, lw=1.5, zorder=4)
    xlabel = "time" if align is None else f"time since $i={align:g}$"
    ax[0].set_ylim(-0.02, 1.0)
    ax[0].set_xlabel(xlabel)
    ax[0].set_ylabel("prevalence  $i$")
    ax[1].set_xlabel(xlabel)
    ax[1].set_ylabel("fast variable  $p$")
    ax[1].set_ylim(0, 1.08)

    a = ax[2]
    a.plot(i_man, p_man, "o", ms=3.5, mfc="none", mew=0.6, color=C_SIM, zorder=3)
    ig = np.linspace(i_man.min(), i_man.max(), 300)
    a.plot(ig, sim_fast_grid(ig, tau, par), "--", color=C_ANALYTIC, lw=1.5, zorder=4)
    a.plot(ig, np.polyval(coef, ig), "-.", color=C_FIT, lw=1.5, zorder=4)
    a.set_xlabel("prevalence  $i$")
    a.set_ylabel("fast variable  $p$")

    resid = np.sqrt(np.mean((np.polyval(coef, i_man) - p_man) ** 2))
    return RowInfo(
        label=(
            rf"$\tau={tau:.3g}\ ({TAU_FACTOR:g}\tau_c),\ \beta={par.beta},\ "
            rf"n={n_extra},\ n_\Delta={n_tri}$"
        ),
        notes={
            "runs": nruns,
            "survivors": int(surv.sum()),
            "N": N,
            "transient_end_t": float(t[k0]),
            "fit_rms": float(resid),
            "endemic_i_sim": float(i_mean[-4:].mean()),
            "endemic_i_1d": float(i_ana[-1]),
            "endemic_i_fit": float(i_fit[-1]),
        },
    )
