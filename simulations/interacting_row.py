"""Figure 5, interacting-contagions row: simulations, 1D reduction and its fit.

Reads the CSV written by ``interacting_nrm.py`` (well-mixed population, two SIS
diseases with co-infection enhancement alpha) and draws (a) prevalence i(t), (b) the
fast variable p(t) and (c) p against i.  The fast variable is the fraction of infected
individuals that carry only one disease, with the two diseases averaged (the model
is symmetric):

    i = (n10 + n01)/(2N) + n11/N        <SI> = (n10 + n01)/(2N)        p = <SI>/i

Means over runs are ratios of means over survivors: runs in which both diseases are
still present at the last time.  The model has the exact invariant <II> = <I>^2, so the
trajectory obeys p = 1 - i and a fitted p(i) is linear up to noise.  The manifold is
measured from the pooled individual runs, fitted with a quadratic and fed through the
model's own beta_eff (figure3.reduced_rhs).  Parameters (alpha, gamma) come from
figure3.build_panels()[3]; tau = 1.3 tau_c with tau_c = gamma.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.integrate import solve_ivp

import figure3 as f3
import style
from models import ic_beta_eff, ic_fast, ic_rhs_1d, ic_tau_c
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


def draw(ax, csvfile: str, align: float | None = None) -> RowInfo:
    """Draw the three panels of the row onto ``ax`` (three axes)."""
    panel = f3.build_panels()[3]  # interacting contagions (alpha, gamma of Figure 3)
    par = panel.par
    tau = TAU_FACTOR * ic_tau_c(par)
    panel = replace(
        panel,
        tau=tau,
        quasi_static=lambda i: ic_fast(i, tau, par),
        beta_eff=lambda i, p: ic_beta_eff(i, p, tau, par),
    )

    t, d, N = load_runs(csvfile, ("n10", "n01", "n11", "I", "SI"))
    n = np.stack([d["n10"], d["n01"], d["n11"]])
    n_inf, SI = d["I"], d["SI"]
    if align is not None:  # t = 0 when a run reaches prevalence `align`
        t, n_inf, n0, n1, n2, SI = align_to_prevalence(
            t, n_inf, N, align, n[0], n[1], n[2], SI
        )
        n = np.stack([n0, n1, n2])
    nruns = n_inf.shape[0]
    surv = ((n[0] + n[2])[:, -1] > 0) & ((n[1] + n[2])[:, -1] > 0)
    Is, SIs = n_inf[surv], SI[surv]

    i_mean = Is.mean(0) / N
    p_mean = SIs.mean(0) / Is.mean(0)
    k0 = transient_end(i_mean, p_mean)
    i_man, p_man = pooled_manifold(Is, SIs, N, 1, t, k0)
    coef = np.polyfit(i_man, p_man, 2)

    i0, tmax = i_mean[0], t[-1]
    tt = np.linspace(0, tmax, 900)
    ivp = dict(t_eval=tt, rtol=1e-9, atol=1e-12, method="LSODA")
    ana = solve_ivp(ic_rhs_1d, [0, tmax], [i0], args=panel.args, **ivp)
    fit = solve_ivp(
        f3.reduced_rhs(panel, coef), [0, tmax], [i0], events=f3.cap_event(1.05), **ivp
    )
    i_ana, i_fit = ana.y[0], fit.y[0]
    p_ana = ic_fast(np.clip(i_ana, 1e-12, 0.999), tau, par)
    p_fit = np.clip(np.polyval(coef, i_fit), 0, 1)

    faint = dict(color=C_SIM, lw=0.5, alpha=0.12, zorder=1)
    for r in np.flatnonzero(surv)[:N_FAINT]:
        ax[0].plot(t, n_inf[r] / N, **faint)
        m = n_inf[r] >= SHOW_P_I_MIN
        ax[1].plot(t[m], (SI[r] / np.maximum(n_inf[r], 1))[m], **faint)
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
    a.plot(ig, ic_fast(ig, tau, par), "--", color=C_ANALYTIC, lw=1.5, zorder=4)
    a.plot(ig, np.polyval(coef, ig), "-.", color=C_FIT, lw=1.5, zorder=4)
    a.set_xlabel("prevalence  $i$")
    a.set_ylabel("fast variable  $p$")

    resid = np.sqrt(np.mean((np.polyval(coef, i_man) - p_man) ** 2))
    return RowInfo(
        label=rf"$\tau={tau:.3g}\ ({TAU_FACTOR:g}\tau_c),\ \alpha={par.alpha}$",
        notes={
            "runs": nruns,
            "survivors": int(surv.sum()),
            "N": N,
            "transient_end_t": float(t[k0]),
            "fit_rms": float(resid),
            "linear_fit_slope": float(np.polyfit(i_man, p_man, 1)[0]),
            "endemic_i_sim": float(i_mean[-4:].mean()),
            "endemic_i_1d": float(i_ana[-1]),
            "endemic_i_fit": float(i_fit[-1]),
        },
    )
