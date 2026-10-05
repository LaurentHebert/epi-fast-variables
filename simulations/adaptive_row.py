"""Figure 5, adaptive-network SIS row: simulations against the 1D reduction and its fit.

Reads the CSV written by ``adaptive_nrm.py`` (Gross et al. rewiring SIS, Poisson
graphs) and draws (a) prevalence i(t), (b) the fast variables p1(t) = <SI>/<I> and
p2(t) = <II>/<I>, with the per-stub densities <SI> = SI/(nN), <II> = 2 II/(nN), i.e.
p1 = SI / (n I), p2 = 2 II / (n I), and (c) p1 against i.  Means over runs are ratios
of means over survivors (I > 0 at the last time).

Rewiring breaks <SI> + <II> = <I>, so the reduction carries two fast variables; only p1
enters beta_eff.  The manifold p1(i) is measured from the individual runs pooled into
bins uniform in i, fitted with a quadratic and fed through the model's own beta_eff
(figure3.reduced_rhs).  Parameters (w, gamma) come from figure3.build_panels()[2];
``degree_factor`` multiplies n for denser networks, and tau = 1.3 tau_c of that model.
"""

from __future__ import annotations

from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

import figure3 as f3
import style
from models import ad_beta_eff, ad_fast, ad_fast_p2, ad_rhs_1d, ad_tau_c
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
    20  # faint p lines drawn only while a run has at least this many infected
)

C_SIM, C_ANALYTIC, C_FIT = style.C_FULL, style.C_REDUCED, style.C_SET[1]


def draw(
    ax, csvfile: str, align: float | None = None, degree_factor: int = 1
) -> RowInfo:
    """Draw the three panels of the row onto ``ax`` (three axes)."""
    panel = f3.build_panels()[2]  # adaptive network SIS (w, n, gamma of Figure 3)
    par = replace(panel.par, n=panel.par.n * degree_factor)
    n = par.n
    tau = TAU_FACTOR * ad_tau_c(par)
    panel = replace(
        panel,
        par=par,
        tau=tau,
        quasi_static=lambda i: ad_fast(i, tau, par),
        beta_eff=lambda i, p: ad_beta_eff(i, p, tau, par),
    )

    t, d, N = load_runs(csvfile, ("I", "SI", "II"))
    n_inf, SI, II = d["I"], d["SI"], d["II"]
    if align is not None:  # t = 0 when a run reaches prevalence `align`
        t, n_inf, SI, II = align_to_prevalence(t, n_inf, N, align, SI, II)
    nruns = n_inf.shape[0]
    surv = n_inf[:, -1] > 0
    Is, SIs, IIs = n_inf[surv], SI[surv], II[surv]

    # ratios of means over survivors
    i_mean = Is.mean(0) / N
    p1_mean = SIs.mean(0) / (n * Is.mean(0))
    p2_mean = 2 * IIs.mean(0) / (n * Is.mean(0))
    k0 = transient_end(i_mean, p1_mean)
    i_man, p_man = pooled_manifold(Is, SIs, N, n, t, k0)
    coef = np.polyfit(i_man, p_man, 2)

    # the 1D models, from the simulations' initial prevalence
    i0, tmax = i_mean[0], t[-1]
    tt = np.linspace(0, tmax, 900)
    ivp = dict(t_eval=tt, rtol=1e-9, atol=1e-12, method="LSODA")
    ana = solve_ivp(ad_rhs_1d, [0, tmax], [i0], args=panel.args, **ivp)
    fit = solve_ivp(
        f3.reduced_rhs(panel, coef), [0, tmax], [i0], events=f3.cap_event(1.05), **ivp
    )
    i_ana, i_fit = ana.y[0], fit.y[0]
    p1_ana = panel.quasi_static(np.clip(i_ana, 1e-12, 0.999))
    p2_ana = ad_fast_p2(np.clip(i_ana, 1e-12, 0.999), p1_ana, tau, par)
    p1_fit = np.clip(np.polyval(coef, i_fit), 0, 1)

    # (a) prevalence and (b) fast variables against time
    faint = dict(color=C_SIM, lw=0.5, alpha=0.12, zorder=1)
    for r in np.flatnonzero(surv)[:N_FAINT]:
        ax[0].plot(t, n_inf[r] / N, **faint)
        m = n_inf[r] >= SHOW_P_I_MIN
        ax[1].plot(t[m], (SI[r] / (n * np.maximum(n_inf[r], 1)))[m], **faint)
    ax[0].plot(t, i_mean, "o", ms=3.0, mfc="none", mew=0.6, color=C_SIM, zorder=3)
    ax[1].plot(t, p1_mean, "o", ms=3.0, mfc="none", mew=0.6, color=C_SIM, zorder=3)
    ax[1].plot(
        t, p2_mean, "o", ms=2.5, mfc="none", mew=0.5, color=C_SIM, alpha=0.6, zorder=3
    )
    ax[0].plot(tt, i_ana, "--", color=C_ANALYTIC, lw=1.5, zorder=4)
    ax[0].plot(fit.t, i_fit, "-.", color=C_FIT, lw=1.5, zorder=4)
    ax[1].plot(tt, p1_ana, "--", color=C_ANALYTIC, lw=1.5, zorder=4)
    ax[1].plot(tt, p2_ana, "--", color=C_ANALYTIC, lw=0.9, alpha=0.75, zorder=4)
    ax[1].plot(fit.t, p1_fit, "-.", color=C_FIT, lw=1.5, zorder=4)
    xlabel = "time" if align is None else f"time since $i={align:g}$"
    ax[0].set_ylim(-0.02, 1.0)
    ax[0].set_xlabel(xlabel)
    ax[0].set_ylabel("prevalence  $i$")
    ax[1].set_xlabel(xlabel)
    ax[1].set_ylabel("fast variables")
    ax[1].set_ylim(0, 1.08)
    size = plt.rcParams["font.size"]
    ax[1].annotate(
        r"$p_2$", (0.55, 0.62), xycoords="axes fraction", fontsize=size, alpha=0.7
    )
    ax[1].annotate(r"$p_1$", (0.55, 0.2), xycoords="axes fraction", fontsize=size)

    # (c) p1 against prevalence
    a = ax[2]
    a.plot(i_mean[: k0 + 1], p1_mean[: k0 + 1], "-", color="0.75", lw=1.0, zorder=1)
    a.plot(i_man, p_man, "o", ms=3.5, mfc="none", mew=0.6, color=C_SIM, zorder=3)
    ig = np.linspace(i_man.min(), i_man.max(), 300)
    a.plot(ig, panel.quasi_static(ig), "--", color=C_ANALYTIC, lw=1.5, zorder=4)
    a.plot(ig, np.polyval(coef, ig), "-.", color=C_FIT, lw=1.5, zorder=4)
    a.set_ylim(0.1, 0.45)
    a.set_xlabel("prevalence  $i$")
    a.set_ylabel(r"fast variable  $p_1$")

    resid = np.sqrt(np.mean((np.polyval(coef, i_man) - p_man) ** 2))
    return RowInfo(
        label=rf"$\tau={tau:.3g}\ ({TAU_FACTOR:g}\tau_c),\ n={n},\ w={par.w}$",
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
