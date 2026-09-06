"""
fitted_reduction_figure.py
==========================
Reproduces the 2x2 figure for "Fast-variable reduction to 
                            complex contagion unifies epidemic models".

Empirical fast-variable reduction.  Where the quasi-static p(i) has no closed
form, run the full model once, discard the initial fast transient, and fit p as
a polynomial in i.  Feed that p(i) back into the model's own prevalence equation
to close a 1D reduction:   di/dt = i (beta_eff(i) - gamma).

Four panels: pairwise SIS | higher-order (simplicial) | adaptive | interacting.
Main axes : p vs i  -- trajectory data, the analytic quasi-static p*(i), and
            constant / linear / quadratic / cubic fits.
Inset     : i(t) -- full model vs the 1D reduction built from each fit.

Note on the interacting-contagions panel: that model has an exact invariant,
<II> = <I>^2 (statistical independence of the two diseases), so the trajectory
obeys p = 1 - i EXACTLY and the linear fit is exact to machine precision.  The
quasi-static root of the Step-3 quadratic is a *different* curve, which the
trajectory does not follow away from threshold.

Run:  python fitted_reduction_figure.py
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from fast_variable_figure import (
    sis_full, sis_p, sis_tauc,
    ho_full, ho_p, ho_beff, ho_tauc,
    ad_full, ad_p1, ad_tauc,
    ic_full, ic_p, ic_tauc,
)

GAM = 1.0
I0 = 1e-6          # seed small so the transient ends while i is still ~1e-5
KAPPA = 5.0        # transient cut: keep once |dp/dt| < KAPPA |di/dt|
IVP = dict(rtol=1e-11, atol=1e-14, method="LSODA", dense_output=True)


def extract(rhs, y0, args, i_of, p_of, T, nsamp=4000):
    """Integrate the full model, drop the fast transient, resample uniformly in i.
    (Raw time samples would over-weight the endpoints, where the trajectory dwells.)"""
    s = solve_ivp(rhs, [0, T], y0, args=args, **IVP)
    tt = np.linspace(0, T, 40000)
    Y = s.sol(tt)
    i, p = i_of(Y), p_of(Y)

    di, dp = np.gradient(i, tt), np.gradient(p, tt)
    on_man = np.abs(dp) < KAPPA * np.abs(di)
    k0 = int(np.argmax(np.convolve(on_man.astype(float), np.ones(50) / 50, "same") > .99))

    i, p, tt = i[k0:], p[k0:], tt[k0:]
    m = np.r_[True, np.diff(i) > 0]
    i, p = i[m], p[m]
    ig = np.linspace(i[0], i[-1], nsamp)
    return ig, np.interp(ig, i, p), (tt[0], i[0])


def rms(i, p, c):
    return float(np.sqrt(np.mean((np.polyval(c, i) - p) ** 2)))


def make_1d(key, coef, args):
    """1D reduction with p(i) taken from the fit; beta_eff from the model itself.

    The constant fit gives a beta_eff with no prevalence dependence, so for the
    SIS-type models the reduced equation is di/dt = i (tau n a0 - gamma): pure
    exponential growth with no saturating term.  It is therefore *unbounded*.
    We freeze the flow at i = 1 so the solver cannot chase i to infinity and
    stall on zero-length steps (which raises 'ts must be strictly increasing').
    The runaway is a genuine feature of the constant fit, not a bug -- it is why
    the constant fit fails -- but it must be integrated safely."""
    pf = lambda i: float(np.clip(np.polyval(coef, i), 0.0, 1.0))

    def guard(f):
        def rhs(t, y):
            i = min(max(float(y[0]), 0.0), 0.995)   # stay off the 1/(1-i) pole
            return [f(i)]
        return rhs

    if key in ("sis", "ad"):
        tau, n, gam = args[0], args[1], args[2]
        return guard(lambda i: i * (tau * n * pf(i) - gam))
    if key == "ho":
        tau, n, gam, beta, nT = args
        return guard(lambda i: i * (ho_beff(i, pf(i), tau, n, gam, beta, nT) - gam))
    if key == "ic":
        tau, alpha, gam = args
        return guard(lambda i: i * (tau * (alpha - (alpha - 1) * pf(i)) * (1 - i) - gam))
    raise ValueError(key)


def cap_event(icap):
    """Terminal event: stop as soon as a fitted reduction leaves the plotted range.

    The constant fit has no saturating term -- for the SIS-type models the
    reduced equation is di/dt = i(tau n a0 - gamma), i.e. unbounded exponential
    growth.  Left alone the solver chases i upward, stalls on zero-length steps,
    and raises 'ValueError: `ts` must be strictly increasing or decreasing'.
    Stopping just above the top of the inset costs nothing visually and keeps the
    integration well away from the 1/(1-i) pole in the simplicial beta_eff."""
    def ev(t, y):
        return y[0] - icap
    ev.terminal = True
    ev.direction = 1
    return ev


def main():
    mpl.rcParams.update({
        "font.family": "serif", "font.serif": ["DejaVu Serif"],
        "font.size": 10, "axes.labelsize": 10,
        "xtick.labelsize": 10, "ytick.labelsize": 10,
        "axes.linewidth": .7, "lines.linewidth": 1.4,
        "xtick.direction": "in", "ytick.direction": "in",
        "mathtext.fontset": "dejavuserif",
    })
    gam = GAM
    # NB: the truth= lambdas below bind tau/beta/w/alpha as DEFAULT ARGUMENTS.
    # Without that, Python's late binding makes every lambda see the *last*
    # value of `tau` assigned in this function, and all four reference curves
    # get drawn at the wrong parameter.
    CF = ["#d0a215", "#3b76af", "#4a9c5d", "#c4553f"]
    LBL = ["constant", "linear", "quadratic", "cubic"]
    n = 5
    cfg = []

    tau = 1.3 * sis_tauc(n, gam)
    cfg.append(dict(key="sis", title="Pairwise SIS",
                    pars=rf"$n={n}$",
                    rhs=sis_full, y0=[I0, (1 - I0) * I0], args=(tau, n, gam), T=140,
                    i_of=lambda Y: Y[0], p_of=lambda Y: Y[1] / Y[0],
                    truth=lambda i, tau=tau, n=n, gam=gam: sis_p(i, tau, n, gam),
                    tlab=r"quasi-static $p^*(i)$", extra=None))

    beta, nT = 0.4, 3
    tau = 1.3 * ho_tauc(beta, n, gam, nT)
    cfg.append(dict(key="ho", title="Higher-order (simplicial)",
                    pars=rf"$n={n},\ n_\Delta={nT},\ \beta={beta}$",
                    rhs=ho_full, y0=[I0, 1 - I0], args=(tau, n, gam, beta, nT), T=170,
                    i_of=lambda Y: Y[0], p_of=lambda Y: Y[1],
                    truth=lambda i, tau=tau, n=n, gam=gam, beta=beta, nT=nT: np.array(
                        [ho_p(x, tau, n, gam, beta, nT) for x in np.atleast_1d(i)]),
                    tlab=r"quasi-static $p^*(i)$", extra=None))

    w = 0.5
    tau = 1.3 * ad_tauc(n, gam, w)
    cfg.append(dict(key="ad", title="Adaptive network SIS",
                    pars=rf"$n={n},\ w={w}$",
                    rhs=ad_full, y0=[I0, 1 - I0, I0], args=(tau, n, gam, w), T=170,
                    i_of=lambda Y: Y[0], p_of=lambda Y: Y[1],
                    truth=lambda i, tau=tau, n=n, gam=gam, w=w: ad_p1(i, tau, n, gam, w),
                    tlab=r"quasi-static $p_1^*(i)$", extra=None))

    alpha = 1.5
    tau = 1.3 * ic_tauc(gam)
    cfg.append(dict(key="ic", title="Interacting contagions",
                    pars=rf"$\alpha={alpha}$",
                    rhs=ic_full, y0=[I0, 0.0], args=(tau, alpha, gam), T=100,
                    i_of=lambda Y: Y[0] + Y[1], p_of=lambda Y: Y[0] / (Y[0] + Y[1]),
                    truth=lambda i, tau=tau, alpha=alpha, gam=gam: ic_p(i, tau, alpha, gam),
                    tlab=r"quasi-static $p^*(i)$",
                    extra=(lambda i: 1 - i, r"exact invariant  $p=1-i$")))

    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.8))

    for ax, c in zip(axes.ravel(), cfg):
        ig, pg, (t_rel, i_rel) = extract(c["rhs"], c["y0"], c["args"],
                                         c["i_of"], c["p_of"], c["T"])
        fits = {d: np.polyfit(ig, pg, d) for d in (0, 1, 2, 3)}

        ax.plot(ig[::45], pg[::45], "o", ms=5.0, mfc="none", mew=.7, color="0.4",
                label="full model", zorder=2)
        ax.plot(ig, c["truth"](ig), "-", color="k", lw=1.7, alpha=.85,
                label=c["tlab"], zorder=3)
        #if c["extra"] is not None:
        #    f, lab = c["extra"]
        #    ax.plot(ig, f(ig), ":", color="k", lw=2.2, alpha=.9, label=lab, zorder=3)
        for d in (0, 1, 2):
            if d==0:
                ax.plot(ig, np.polyval(fits[d], ig), ":", lw=1.15, color=CF[d],
                    label=rf"{LBL[d]}", zorder=4)#   (rms $=$ {rms(ig, pg, fits[d]):.0e})", zorder=4)
            elif d==1:
                ax.plot(ig, np.polyval(fits[d], ig), "-.", lw=2, color=CF[d],
                    label=rf"{LBL[d]}", zorder=4)#   (rms $=$ {rms(ig, pg, fits[d]):.0e})", zorder=4)
            elif d==2:
                ax.plot(ig, np.polyval(fits[d], ig), "--", lw=1.15, color=CF[d+1],
                    label=rf"{LBL[d]}", zorder=4)#   (rms $=$ {rms(ig, pg, fits[d]):.0e})", zorder=4)
            elif d==3:
                ax.plot(ig, np.polyval(fits[d], ig), ":", lw=1.35, color=CF[d],
                    label=rf"{LBL[d]}", zorder=4)#   (rms $=$ {rms(ig, pg, fits[d]):.0e})", zorder=4)

        ax.set_xlabel("prevalence  $i$")
        ax.set_ylabel(r"fast variable  $p$")
        ax.set_title(c["title"], loc="left", fontsize=10.5, weight="bold")
        ax.text(.985, 1.055, c["pars"], transform=ax.transAxes, ha="right", va="top",
                fontsize=10, color="0.35")
        ax.legend(frameon=False, loc="lower left", fontsize=10,
                  handlelength=1.7, borderpad=.2, labelspacing=.3)
        ax.set_xlim(0, ig[-1] * 1.03)
        ax.tick_params(top=True, right=True, length=3)

        axi = ax.inset_axes([0.615, 0.55, 0.355, 0.40])
        T = c["T"]
        icap = min(0.95, 1.45 * ig[-1])       # just above the top of the inset
        F = solve_ivp(c["rhs"], [0, T], c["y0"], args=c["args"], **IVP)
        tt = np.linspace(0, T, 900)
        axi.plot(tt, c["i_of"](F.sol(tt)), color="k", lw=1.15, alpha=.85, zorder=5)
        for d in (0, 1, 2):
            R = solve_ivp(make_1d(c["key"], fits[d], c["args"]), [0, T], [I0],
                          t_eval=tt, rtol=1e-9, atol=1e-12, method="LSODA",
                          events=cap_event(icap))
            if d==0:
                axi.plot(R.t, R.y[0], ":", lw=1.15, color=CF[d])
            elif d==1:
                axi.plot(R.t, R.y[0], "-.", lw=3, color=CF[d])
            elif d==2:
                axi.plot(R.t, R.y[0], "--", lw=2, color=CF[d+1])
            elif d==3:
                axi.plot(R.t, R.y[0], ":", lw=1, color=CF[d])
        axi.set_ylim(-0.03 * ig[-1], 1.35 * ig[-1])   # constant fit may exit the top
        axi.set_xlabel("time", fontsize=10, labelpad=1)
        axi.set_ylabel("prevalence $i$", fontsize=10, labelpad=1)
        axi.tick_params(labelsize=10, length=2)
        #axi.set_title("1D reduction from fit", fontsize=10, pad=2.5)

        print(f"{c['title']:26s} transient ends t={t_rel:5.2f} (i={i_rel:.1e})  |  "
              + "  ".join(f"{LBL[d][:5]}={rms(ig,pg,fits[d]):.1e}" for d in (0, 1, 2, 3)))

    fig.tight_layout()
    fig.savefig("fig_fitted_reduction.pdf")
    fig.savefig("fig_fitted_reduction.png", dpi=200)
    print("\nwrote fig_fitted_reduction.pdf / .png")


if __name__ == "__main__":
    main()
