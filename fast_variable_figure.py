"""
fast_variable_figure.py
=======================
Reproduces the 4x3 figure for "Fast-variable reduction to 
                            complex contagion unifies epidemic models".

Rows   : pairwise SIS | higher-order (simplicial) | adaptive network SIS |
         interacting contagions
Columns: (a) prevalence i(t), full vs 1D reduction
         (b) fast-variable relaxation p(t), full vs quasi-static p*(i)
         (c) bifurcation diagram in tau: full-model up-sweep (filled) and
             down-sweep (open) vs the analytic endemic branch (dashed)

Conventions follow the manuscript:
    gamma  recovery rate          tau    transmission rate
    n      degree                 p1     = <SI>/<I>  in [0,1]
    beta, n_triangle              simplicial rate / triangle degree
    w      rewiring rate          alpha  co-infection enhancement

Requires: numpy, scipy, matplotlib.   Run:  python fast_variable_figure.py
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

GAM = 1.0                      # recovery rate: sets the time unit
IVP = dict(rtol=1e-10, atol=1e-13, method="LSODA", dense_output=True)
SWP = dict(rtol=1e-9, atol=1e-12, method="LSODA")


# ===================================================================== #
#  1. CLASSICAL PAIRWISE SIS                        
# ===================================================================== #
def sis_full(t, y, tau, n, gam):
    """Minimum system in (<I>, <SI>)."""
    i, SI = y
    di = tau * n * SI - gam * i
    dSI = (tau * (n - 2) * SI - 2 * tau * (n - 1) * SI**2 / (1 - i)
           + gam * (i - 2 * SI))
    return [di, dSI]


def sis_p(i, tau, n, gam):
    """Slaved fast variable: positive root of  tau n A(i) p^2 -(tau(n-2)-gam)p -gam = 0."""
    A = (n + (n - 2) * i) / (n * (1 - i))
    a, b, c = tau * n * A, -(tau * (n - 2) - gam), -gam
    return (-b + np.sqrt(b * b - 4 * a * c)) / (2 * a)


def sis_1d(t, y, tau, n, gam):
    i = np.clip(y[0], 1e-14, 1 - 1e-9)
    return [i * (tau * n * sis_p(i, tau, n, gam) - gam)]


def sis_tauc(n, gam):
    return gam / (n - 1)                       # tau (n-1) = gamma


# ===================================================================== #
#  2. HIGHER-ORDER (SIMPLICIAL) PAIR-BASED SIS      
# ===================================================================== #
def ho_pdot(i, p, tau, n, gam, beta, nT):
    """RHS of the fast-variable equation."""
    om = 1.0 - i
    return (gam * (1 - p) + tau * (n - 2) * p - tau * n * p**2
            - 2 * tau * (n - 1) * p**2 * i / om
            + beta * nT * ((n - 4 - n * p) / n) * p**2 * (1 - p) / om
            - 2 * beta * nT * ((n - 2) / n) * p**3 * (1 - p) * i / om**2)


def ho_beff(i, p, tau, n, gam, beta, nT):
    """Effective force of infection."""
    return tau * n * p + beta * nT * p**2 * (1 - p) / (1 - i)


def ho_full(t, y, tau, n, gam, beta, nT):
    """Exact (i, p) system."""
    i = np.clip(y[0], 1e-14, 1 - 1e-9)
    p = np.clip(y[1], 0.0, 1.0)
    return [i * (ho_beff(i, p, tau, n, gam, beta, nT) - gam),
            ho_pdot(i, p, tau, n, gam, beta, nT)]


def ho_p(i, tau, n, gam, beta, nT):
    """Root in (0,1) of the quasi-static quartic; no closed form."""
    f = lambda p: ho_pdot(i, p, tau, n, gam, beta, nT)
    lo, hi = 1e-10, 1.0 - 1e-12
    if f(lo) <= 0 or f(hi) >= 0:
        return np.nan
    return brentq(f, lo, hi, xtol=1e-14, rtol=1e-15)


def ho_1d(t, y, tau, n, gam, beta, nT):
    i = np.clip(y[0], 1e-14, 1 - 1e-9)
    p = ho_p(i, tau, n, gam, beta, nT)
    return [0.0] if not np.isfinite(p) else \
           [i * (ho_beff(i, p, tau, n, gam, beta, nT) - gam)]


def ho_tauc(beta, n, gam, nT):
    """Threshold: invert the parametric bifurcation curve for given beta."""
    if beta <= 0:
        return gam / (n - 1)
    g = lambda p: gam * (n * (1 - p) - 1) / (nT * p**2 * (1 - p)) - beta
    p = brentq(g, 1 - 2 / n + 1e-9, 1 - 1 / n - 1e-12, xtol=1e-14)
    return gam * (2 + n * (p - 1)) / (n * p)


# ===================================================================== #
#  3. ADAPTIVE NETWORK SIS  (Gross-D'Lima-Blasius)
# ===================================================================== #
def ad_full(t, y, tau, n, gam, w):
    """Exact (i, p1, p2) system."""
    i = np.clip(y[0], 1e-14, 1 - 1e-9)
    p1, p2 = y[1], y[2]
    r = i / (1 - i)
    return [i * (tau * n * p1 - gam),
            gam * p2 + (tau * (n - 1) - w) * p1 - tau * n * p1**2
            + tau * n * p1 * (1 - 3 * p1 - p2) * r,
            -gam * p2 + 2 * tau * p1 - tau * n * p1 * p2 + 2 * tau * n * p1**2 * r]


def ad_p1(i, tau, n, gam, w):
    """Quasi-static p1: non-negative root of A p^2 + B p + C = 0."""
    A = (n * tau)**2 * (1 + i) / (1 - i)
    B = tau * n * (gam + w - tau * (n - 1) + i * (tau - w))
    C = gam * (w - tau * (n + 1) + i * (tau - w))
    return (-B + np.sqrt(np.maximum(B * B - 4 * A * C, 0.0))) / (2 * A)


def ad_p2(i, p1, tau, n, gam, w):
    """Quasi-static p2 from p1."""
    return (tau * n * p1**2 * (1 + i) / (1 - i)
            - p1 * ((tau - w) * (1 - i) + tau * (n - 2))) / gam


def ad_1d(t, y, tau, n, gam, w):
    i = np.clip(y[0], 1e-14, 1 - 1e-9)
    return [i * (tau * n * ad_p1(i, tau, n, gam, w) - gam)]


def ad_tauc(n, gam, w):
    return (gam + w) / n                       # tau n = gamma + w


def ad_branch(i, n, gam, w):
    """Endemic branch in closed form."""
    return (gam + w * (1 - i)**2) / ((1 - i) * (n - i))


def ad_wstar(n, gam):
    return gam * (n + 1) / (n - 1)             # continuous/discontinuous boundary


# ===================================================================== #
#  4. INTERACTING CONTAGIONS                      
# ===================================================================== #
def ic_full(t, y, tau, alpha, gam):
    """Minimum system in (<SI>, <II>)."""
    x, yy = y
    foi = tau * (x + alpha * yy)
    return [(1 - 2 * x - yy) * foi + gam * yy - gam * x - x * foi,
            2 * tau * x**2 + 2 * alpha * tau * yy * x - 2 * gam * yy]


def ic_p(i, tau, alpha, gam):
    """Quasi-static p = <SI>/<I>: the minus-sign branch."""
    A = tau * (alpha - 1) * (1 + i)
    B = -(gam + tau * (2 * alpha - 1 + i))
    C = gam + tau * alpha * (1 - i)
    return (-B - np.sqrt(np.maximum(B * B - 4 * A * C, 0.0))) / (2 * A)


def ic_beff(i, tau, alpha, gam):
    return tau * (alpha - (alpha - 1) * ic_p(i, tau, alpha, gam))


def ic_1d(t, y, tau, alpha, gam):
    i = np.clip(y[0], 1e-14, 1 - 1e-9)
    return [i * (ic_beff(i, tau, alpha, gam) * (1 - i) - gam)]


def ic_tauc(gam):
    return gam                                 # tau_c = gamma, independent of alpha


# ===================================================================== #
#  Generic helpers
# ===================================================================== #
def endemic_branch(res_fn, igrid, lo=1e-5, hi=6.0):
    """For each i, solve res_fn(i, tau) = 0 for tau. Traces the full S-curve,
    unstable branch included, because it is parameterised by i rather than tau."""
    out = []
    for i in igrid:
        try:
            out.append(brentq(lambda t: res_fn(i, t), lo, hi, xtol=1e-13))
        except Exception:
            out.append(np.nan)
    return np.array(out)


def sweep(rhs, seed, taus, argf, i_of, T=6000, up=True, cont=True):
    """Continuation sweep of the FULL model: integrate to steady state at each
    tau, carrying the previous state forward. Up- and down-sweeps expose
    hysteresis where the bifurcation is discontinuous."""
    ys, out = None, []
    for t in (taus if up else taus[::-1]):
        y0 = (np.array(seed(t), float) if (ys is None or not cont)
              else np.clip(np.nan_to_num(ys, nan=1e-7), 1e-10, 1.0))
        ys = solve_ivp(rhs, [0, T], y0, args=argf(t), **SWP).y[:, -1]
        out.append(i_of(np.clip(np.nan_to_num(ys, nan=0.0), 0.0, 1.0)))
    out = np.array(out)
    return out if up else out[::-1]


# ===================================================================== #
#  Figure
# ===================================================================== #
def make_figure(outfile="fig_fast_variable"):
    mpl.rcParams.update({
        "font.family": "serif", "font.serif": ["DejaVu Serif"],
        "font.size": 10, "axes.labelsize": 10,
        "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
        "axes.linewidth": 0.7, "lines.linewidth": 1.5,
        "xtick.direction": "in", "ytick.direction": "in",
        "mathtext.fontset": "dejavuserif",
    })
    C_FULL, C_1D = "#1a3d6d", "#c0392b"
    C_SET = ["#3b76af", "#4a9c5d", "#c4553f"]

    gam, i0 = GAM, 1e-3
    fig, ax = plt.subplots(4, 3, figsize=(9.8, 10.6))

    # ---------------- row 0: pairwise SIS ---------------------------- #
    n = 5
    tau, T = 1.3 * sis_tauc(n, gam), 70
    F = solve_ivp(sis_full, [0, T], [i0, (1 - i0) * i0], args=(tau, n, gam), **IVP)
    R = solve_ivp(sis_1d,   [0, T], [i0],                args=(tau, n, gam), **IVP)
    tt = np.linspace(0, T, 900)
    ax[0, 0].plot(tt, F.sol(tt)[0], color=C_FULL, label="full model")
    ax[0, 0].plot(tt, R.sol(tt)[0], "--", color=C_1D, label="1D reduction")
    ax[0, 0].legend(frameon=False, loc="center right")
    ax[0, 1].plot(tt, F.sol(tt)[1] / F.sol(tt)[0], color=C_FULL)
    ax[0, 1].plot(tt, sis_p(np.clip(R.sol(tt)[0], 1e-12, .999), tau, n, gam),
                  "--", color=C_1D)

    a = ax[0, 2]
    taus = np.linspace(0.02, 0.60, 90)
    ig = np.linspace(1e-4, 0.93, 350)
    for k, nn in enumerate([4, 6, 10]):
        up = sweep(sis_full, lambda t: [1e-5, 1e-5], taus,
                   lambda t: (t, nn, gam), lambda y: y[0], cont=False)
        a.plot(taus, up, "o", ms=2.2, color=C_SET[k], alpha=.7, label=f"$n={nn}$")
        a.plot(endemic_branch(lambda i, t: t * nn * sis_p(i, t, nn, gam) - gam, ig),
               ig, "--", color=C_SET[k], lw=1.1)
    a.set_xlim(0, .6); a.set_ylim(-.02, 1.1)
    a.legend(frameon=False, loc="upper left")
    a.set_title(r"(c) bifurcation, continuous for all $n$", loc="right", fontsize=10)

    # ---------------- row 1: higher-order (simplicial) --------------- #
    n, nT, beta, BSTAR = 5, 3, 0.4, 0.8246          # beta* found numerically
    tau, T = 1.3 * ho_tauc(beta, n, gam, nT), 80
    F = solve_ivp(ho_full, [0, T], [i0, 1 - i0], args=(tau, n, gam, beta, nT), **IVP)
    R = solve_ivp(ho_1d,   [0, T], [i0],         args=(tau, n, gam, beta, nT), **IVP)
    tt = np.linspace(0, T, 900)
    ax[1, 0].plot(tt, F.sol(tt)[0], color=C_FULL)
    ax[1, 0].plot(tt, R.sol(tt)[0], "--", color=C_1D)
    ax[1, 1].plot(tt, F.sol(tt)[1], color=C_FULL)
    ax[1, 1].plot(tt, [ho_p(max(x, 1e-12), tau, n, gam, beta, nT)
                       for x in R.sol(tt)[0]], "--", color=C_1D)

    a = ax[1, 2]
    taus = np.linspace(0.02, 0.42, 110)
    ig = np.linspace(1e-4, 0.93, 350)
    for k, b in enumerate([0.4, BSTAR, 1.6]):
        up = sweep(ho_full, lambda t: [1e-6, 0.9], taus,
                   lambda t: (t, n, gam, b, nT), lambda y: y[0])
        dn = sweep(ho_full, lambda t: [0.55, 0.35], taus,
                   lambda t: (t, n, gam, b, nT), lambda y: y[0], up=False)
        lab = rf"$\beta={b:.2f}$" + (r"  $(\beta^*)$" if abs(b - BSTAR) < 1e-3 else "")
        a.plot(taus, up, "o", ms=2.0, color=C_SET[k], alpha=.7, label=lab)
        a.plot(taus, dn, "o", ms=2.0, mfc="none", mew=.55, color=C_SET[k], alpha=.7)
        a.plot(endemic_branch(
            lambda i, t: ho_beff(i, ho_p(i, t, n, gam, b, nT), t, n, gam, b, nT) - gam,
            ig, hi=3.0), ig, "--", color=C_SET[k], lw=1.1)
    a.set_xlim(0, .42); a.set_ylim(-.02, 1.1)
    a.legend(frameon=False, loc="upper left")
    a.set_title(r"discontinuous for $\beta > \beta^* \approx 0.82$", loc="right", fontsize=10)

    # ---------------- row 2: adaptive network SIS -------------------- #
    n, w = 5, 0.5
    tau, T = 1.3 * ad_tauc(n, gam, w), 70
    F = solve_ivp(ad_full, [0, T], [i0, 1 - i0, i0], args=(tau, n, gam, w), **IVP)
    R = solve_ivp(ad_1d,   [0, T], [i0],             args=(tau, n, gam, w), **IVP)
    tt = np.linspace(0, T, 900)
    ax[2, 0].plot(tt, F.sol(tt)[0], color=C_FULL)
    ax[2, 0].plot(tt, R.sol(tt)[0], "--", color=C_1D)

    ir = np.clip(R.sol(tt)[0], 1e-12, .999)
    p1r = ad_p1(ir, tau, n, gam, w)
    ax[2, 1].plot(tt, F.sol(tt)[1], color=C_FULL)                       # p1 full
    ax[2, 1].plot(tt, p1r, "--", color=C_1D)                            # p1 slaved
    ax[2, 1].plot(tt, F.sol(tt)[2], color=C_FULL, lw=.8, alpha=.55)     # p2 full
    ax[2, 1].plot(tt, ad_p2(ir, p1r, tau, n, gam, w), "--", color=C_1D,
                  lw=.8, alpha=.55)                                     # p2 slaved
    ax[2, 1].annotate(r"$p_1$", (.35, .55), xycoords="axes fraction", fontsize=10)
    ax[2, 1].annotate(r"$p_2$", (.35, .3), xycoords="axes fraction", fontsize=10, alpha=.7)

    a = ax[2, 2]
    taus = np.linspace(0.08, 0.95, 120)
    ig = np.linspace(1e-4, 0.93, 400)
    for k, ww in enumerate([0.5, 1.5, 3.0]):
        up = sweep(ad_full, lambda t: [1e-6, 1.0, 1e-6], taus,
                   lambda t: (t, n, gam, ww), lambda y: y[0])
        dn = sweep(ad_full, lambda t: [0.5, 0.4, 0.5], taus,
                   lambda t: (t, n, gam, ww), lambda y: y[0], up=False)
        lab = rf"$w={ww}$" + (r"  $(w^*)$" if abs(ww - ad_wstar(n, gam)) < 1e-9 else "")
        a.plot(taus, up, "o", ms=2.0, color=C_SET[k], alpha=.7, label=lab)
        a.plot(taus, dn, "o", ms=2.0, mfc="none", mew=.55, color=C_SET[k], alpha=.7)
        a.plot(ad_branch(ig, n, gam, ww), ig, "--", color=C_SET[k], lw=1.1)
    a.set_xlim(.08, .95); a.set_ylim(-.02, 1.1)
    a.legend(frameon=False, loc="upper left")
    a.set_title(r"discontinuous for $w>w^*=\gamma\frac{n+1}{n-1}$", loc="right", fontsize=10)

    # ---------------- row 3: interacting contagions ------------------ #
    alpha = 1.5
    tau, T = 1.3 * ic_tauc(gam), 45
    F = solve_ivp(ic_full, [0, T], [i0, 0.0], args=(tau, alpha, gam), **IVP)
    R = solve_ivp(ic_1d,   [0, T], [i0],      args=(tau, alpha, gam), **IVP)
    tt = np.linspace(0, T, 900)
    ifull = F.sol(tt)[0] + F.sol(tt)[1]
    ax[3, 0].plot(tt, ifull, color=C_FULL)
    ax[3, 0].plot(tt, R.sol(tt)[0], "--", color=C_1D)
    ax[3, 1].plot(tt, F.sol(tt)[0] / ifull, color=C_FULL)
    ax[3, 1].plot(tt, ic_p(np.clip(R.sol(tt)[0], 1e-12, .999), tau, alpha, gam),
                  "--", color=C_1D)

    a = ax[3, 2]
    taus = np.linspace(0.55, 1.45, 120)
    ig = np.linspace(1e-4, 0.95, 400)
    for k, al in enumerate([1.5, 2.0, 3.0]):
        up = sweep(ic_full, lambda t: [1e-6, 0.0], taus,
                   lambda t: (t, al, gam), lambda y: y[0] + y[1])
        dn = sweep(ic_full, lambda t: [0.4, 0.4], taus,
                   lambda t: (t, al, gam), lambda y: y[0] + y[1], up=False)
        lab = rf"$\alpha={al}$" + (r"  $(\alpha^*)$" if al == 2.0 else "")
        a.plot(taus, up, "o", ms=2.0, color=C_SET[k], alpha=.7, label=lab)
        a.plot(taus, dn, "o", ms=2.0, mfc="none", mew=.55, color=C_SET[k], alpha=.7)
        a.plot(endemic_branch(lambda i, t: ic_beff(i, t, al, gam) * (1 - i) - gam,
                              ig, lo=.05), ig, "--", color=C_SET[k], lw=1.1)
    a.set_xlim(.55, 1.45); a.set_ylim(-.02, 1.1)
    a.legend(frameon=False, loc="upper left")
    a.set_title(r"discontinuous for $\alpha>\alpha^* = 2$", loc="right", fontsize=10)

    # ---------------- cosmetics -------------------------------------- #
    rows = ["Pairwise SIS", "Higher-order (simplicial)",
            "Adaptive network SIS", "Interacting contagions"]
    pars = [r"$n=5$", r"$n=5,\ n_\Delta=3,\ \beta=0.4$",
            r"$n=5,\ w=0.5$", r"$\alpha=1.5$"]
    for r in range(4):
        ax[r, 0].set_ylabel("prevalence  $i$")
        ax[r, 1].set_ylabel("fast variable")
        ax[r, 2].set_ylabel(r"endemic  $i^*$")
        ax[r, 1].set_ylim(0, 1.08)
        ax[r, 0].text(-0.30, 1.06, rows[r], transform=ax[r, 0].transAxes,
                      fontsize=10, weight="bold", va="bottom")
        ax[r, 0].text(0.97, 0.06, pars[r], transform=ax[r, 0].transAxes,
                      fontsize=10, ha="right", color="0.35")
    for c in range(3):
        ax[3, c].set_xlabel("time" if c < 2 else r"transmission rate  $\tau$")
    for a in ax.ravel():
        a.tick_params(top=True, right=True, length=3)
    ax[0, 0].set_title("(a) prevalence", loc="right", fontsize=10)
    ax[0, 1].set_title("(b) fast variable relaxation", loc="right", fontsize=10)

    fig.tight_layout(h_pad=1.6)
    fig.savefig(outfile + ".pdf")
    fig.savefig(outfile + ".png", dpi=200)
    print(f"wrote {outfile}.pdf and {outfile}.png")


# ===================================================================== #
#  Consistency checks (optional; run with --check; based on Claude Code)
# ===================================================================== #
def checks():
    gam = GAM
    print("Equilibria of the FULL model vs the 1D reduction "
          "(quasi-static approx is exact at steady states):")
    for tau in (0.30, 0.45):
        f = solve_ivp(sis_full, [0, 8e3], [.05, .05 * .95],
                      args=(tau, 5, gam), **SWP).y[0, -1]
        r = solve_ivp(sis_1d, [0, 8e3], [.05], args=(tau, 5, gam), **SWP).y[0, -1]
        print(f"  SIS  tau={tau}: full={f:.12f}  1D={r:.12f}  |diff|={abs(f-r):.1e}")
    for al in (1.5, 3.0):
        s = solve_ivp(ic_full, [0, 8e3], [.02, 0.], args=(1.2, al, gam), **SWP)
        f = s.y[0, -1] + s.y[1, -1]
        r = solve_ivp(ic_1d, [0, 8e3], [.02], args=(1.2, al, gam), **SWP).y[0, -1]
        print(f"  IC   a={al}: full={f:.12f}  1D={r:.12f}  |diff|={abs(f-r):.1e}")

    print("\nAdaptive: closed-form branch vs quadratic root:")
    for w in (0.5, 3.0):
        for i in (0.05, 0.6):
            c = ad_branch(i, 5, gam, w)
            nnum = brentq(lambda t: t * 5 * ad_p1(i, t, 5, gam, w) - gam, 1e-4, 6.)
            print(f"  w={w} i={i}: closed={c:.10f}  numeric={nnum:.10f}  "
                  f"|diff|={abs(c-nnum):.1e}")

    print("\nSimplicial: bistability boundary in beta (n=5, n_tri=3):")
    slope = lambda b, h=5e-3: (
        brentq(lambda t: ho_beff(h, ho_p(h, t, 5, gam, b, 3), t, 5, gam, b, 3) - gam,
               1e-5, 3.) - ho_tauc(b, 5, gam, 3)) / h
    print(f"  beta* = {brentq(slope, 0.5, 1.0, xtol=1e-8):.5f}")


if __name__ == "__main__":
    import sys
    if "--check" in sys.argv:
        checks()
    else:
        make_figure()
