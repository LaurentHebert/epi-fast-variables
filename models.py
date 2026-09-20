"""Model equations for the fast-variable reduction manuscript.

"Fast-variable reduction to complex contagion unifies epidemic models".

Five models appear in the paper.  Each is described here by the same small set of
functions, so that a figure can be written once and pointed at any of them:

    <m>_rhs        full model, in the coordinates the paper reduces it to
    <m>_fast       quasi-static (slaved) fast variable p(i), Step 3 of each section
    <m>_beta_eff   effective force of infection beta_eff(i, p)
    <m>_rhs_1d     the closed 1D reduction  di/dt = i (beta_eff - gamma)
    <m>_tau_c      epidemic threshold

with the prefixes ``sis_`` (pairwise SIS), ``sim_`` (higher-order/simplicial),
``ad_`` (adaptive network), ``ic_`` (interacting contagions) and ``sir_``
(pairwise SIR).  Model constants travel together in a frozen dataclass, so a call
reads ``sis_fast(i, tau, PairwiseSIS(n=5))`` rather than a bare tuple of scalars.

``beta_eff`` always takes ``p`` from the caller rather than computing it.  That is
what lets the analytic quasi-static root and a ``p(i)`` fitted from trajectory data
(Figure 3) pass through one and the same definition of beta_eff.

Equation and section numbers refer to the manuscript.  Full derivations are in
Appendix 6.2; the shared framework -- the universal threshold beta_eff(0) = gamma
and the slope criterion that classifies the transition -- is Section 2, Eqs. (1)-(4).

Recovery rate gamma sets the time unit; tau is the transmission rate; n the degree.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq

# Most quantities below are evaluated both at a single prevalence and over a grid
# of them.  Functions that are NOT vectorised say so in their docstring.
Scalar = float | NDArray[np.float64]

# Prevalence is clipped away from 0 and 1 before entering any expression with a
# 1/i or 1/(1-i) factor.
_I_MIN = 1e-14
_I_MAX = 1.0 - 1e-9


# --------------------------------------------------------------------------- #
#  1.  Classical pairwise SIS                                 Appendix 6.2.1   #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class PairwiseSIS:
    """Pairwise SIS on an n-regular network."""

    n: int = 5
    gamma: float = 1.0


def sis_rhs(t: float, y: list[float], tau: float, par: PairwiseSIS) -> list[float]:
    """Minimum system in (<I>, <SI>) -- Eqs. (15)-(16)."""
    n, gam = par.n, par.gamma
    i, SI = y
    di = tau * n * SI - gam * i
    dSI = tau * (n - 2) * SI - 2 * tau * (n - 1) * SI**2 / (1 - i) + gam * (i - 2 * SI)
    return [di, dSI]


def sis_fast(i: Scalar, tau: float, par: PairwiseSIS) -> Scalar:
    """Slaved p: positive root of tau n A(i) p^2 - (tau(n-2) - gamma) p - gamma = 0.

    Quadratic Eq. (24) with A(i) of Eq. (21); root Eq. (22).
    """
    n, gam = par.n, par.gamma
    A = (n + (n - 2) * i) / (n * (1 - i))
    a, b, c = tau * n * A, -(tau * (n - 2) - gam), -gam
    return (-b + np.sqrt(b * b - 4 * a * c)) / (2 * a)


def sis_beta_eff(i: Scalar, p: Scalar, tau: float, par: PairwiseSIS) -> Scalar:
    """beta_eff = tau n p -- Eq. (23).  Independent of i at fixed p."""
    return tau * par.n * p


def sis_rhs_1d(t: float, y: list[float], tau: float, par: PairwiseSIS) -> list[float]:
    """1D reduction di/dt = i (beta_eff(i) - gamma) -- Eq. (23)."""
    i = np.clip(y[0], _I_MIN, _I_MAX)
    return [i * (sis_beta_eff(i, sis_fast(i, tau, par), tau, par) - par.gamma)]


def sis_tau_c(par: PairwiseSIS) -> float:
    """Threshold tau_c (n - 1) = gamma -- Appendix 6.2.1, Step 2."""
    return par.gamma / (par.n - 1)


def sis_slope_at_threshold(par: PairwiseSIS) -> float:
    """Slope of beta_eff in i at threshold -- Eq. (28).

    d beta_eff / di at (i, tau) = (0, tau_c).  Negative for every n, so the pairwise
    transition is continuous whatever the degree.
    """
    n, gam = par.n, par.gamma
    return -2 * gam * (n - 1) ** 2 / (n * (2 * n - 1))


# --------------------------------------------------------------------------- #
#  2.  Higher-order (simplicial) pair-based SIS               Appendix 6.2.2   #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class Simplicial:
    """Pair-based simplicial SIS after Malizia et al., relabelled to paper notation.

    ``beta`` is the simplicial (triangle) transmission rate and ``n_tri`` the
    triangle degree; ``n`` and ``gamma`` keep their pairwise meaning.
    """

    n: int = 5
    n_tri: int = 3
    beta: float = 0.4
    gamma: float = 1.0


def sim_pdot(i: Scalar, p: Scalar, tau: float, par: Simplicial) -> Scalar:
    """Fast-variable equation dp/dt -- Eq. (37)."""
    n, gam, beta, nT = par.n, par.gamma, par.beta, par.n_tri
    om = 1.0 - i
    return (
        gam * (1 - p)
        + tau * (n - 2) * p
        - tau * n * p**2
        - 2 * tau * (n - 1) * p**2 * i / om
        + beta * nT * ((n - 4 - n * p) / n) * p**2 * (1 - p) / om
        - 2 * beta * nT * ((n - 2) / n) * p**3 * (1 - p) * i / om**2
    )


def sim_beta_eff(i: Scalar, p: Scalar, tau: float, par: Simplicial) -> Scalar:
    """beta_eff = tau n p + beta n_tri p^2 (1 - p) / (1 - i) -- Eq. (44).

    Cubic in p, which is why this model has no closed-form beta_eff(i): p itself is
    only an implicit root (see :func:`sim_fast`).
    """
    return tau * par.n * p + par.beta * par.n_tri * p**2 * (1 - p) / (1 - i)


def sim_rhs(t: float, y: list[float], tau: float, par: Simplicial) -> list[float]:
    """Exact (i, p) system -- Eqs. (36)-(37).

    Well defined at i = 0, unlike the (<I>, <SI>) form it is derived from.
    """
    i = np.clip(y[0], _I_MIN, _I_MAX)
    p = np.clip(y[1], 0.0, 1.0)
    return [i * (sim_beta_eff(i, p, tau, par) - par.gamma), sim_pdot(i, p, tau, par)]


def sim_fast(i: float, tau: float, par: Simplicial) -> float:
    """Root in (0, 1) of the quasi-static quartic -- Eq. (43).

    No usable closed form.

    Scalar-only: solved by bisection, one prevalence at a time.  Returns NaN where
    the bracket carries no sign change, which propagates visibly rather than
    silently returning a wrong root.
    """

    def f(p: float) -> float:
        return float(sim_pdot(i, p, tau, par))

    lo, hi = 1e-10, 1.0 - 1e-12
    if f(lo) <= 0 or f(hi) >= 0:
        return np.nan
    return brentq(f, lo, hi, xtol=1e-14, rtol=1e-15)


def sim_fast_grid(i: Scalar, tau: float, par: Simplicial) -> NDArray[np.float64]:
    """:func:`sim_fast` mapped over a grid of prevalences."""
    return np.array([sim_fast(x, tau, par) for x in np.atleast_1d(i)])


def sim_rhs_1d(t: float, y: list[float], tau: float, par: Simplicial) -> list[float]:
    """1D reduction -- Eq. (45).  Frozen where the quartic has no root in (0, 1)."""
    i = np.clip(y[0], _I_MIN, _I_MAX)
    p = sim_fast(i, tau, par)
    if not np.isfinite(p):
        return [0.0]
    return [i * (sim_beta_eff(i, p, tau, par) - par.gamma)]


def sim_tau_c(par: Simplicial) -> float:
    """Threshold for a given beta, by inverting the parametric bifurcation curve.

    Eqs. (38)-(39) cannot be eliminated for a single threshold relation, so the
    curve is parameterised by p_c: Eq. (41) gives beta(p_c) and Eq. (40) tau(p_c).
    Solve the former for p_c, then evaluate the latter.  p_c is confined to
    [1 - 2/n, 1 - 1/n], where beta and tau are both non-negative.
    """
    n, gam, beta, nT = par.n, par.gamma, par.beta, par.n_tri
    if beta <= 0:
        return gam / (n - 1)

    def g(p: float) -> float:
        return gam * (n * (1 - p) - 1) / (nT * p**2 * (1 - p)) - beta

    p = brentq(g, 1 - 2 / n + 1e-9, 1 - 1 / n - 1e-12, xtol=1e-14)
    return gam * (2 + n * (p - 1)) / (n * p)


def sim_beta_star(par: Simplicial, lo: float = 0.5, hi: float = 1.0) -> float:
    """Simplicial rate at which the transition changes character -- Eq. (64).

    Located numerically, as the beta whose threshold slope d tau_c / di vanishes at
    i = 0.  Closed form: beta* = (gamma / n_tri) h(n), with h(n) built from the root
    of the quartic Q_n and independent of both n_tri and gamma.
    """
    h = 5e-3

    def slope(b: float) -> float:
        p = Simplicial(n=par.n, n_tri=par.n_tri, beta=b, gamma=par.gamma)

        def residual(t: float) -> float:
            return float(sim_beta_eff(h, sim_fast(h, t, p), t, p) - p.gamma)

        return (brentq(residual, 1e-5, 3.0) - sim_tau_c(p)) / h

    return brentq(slope, lo, hi, xtol=1e-8)


# --------------------------------------------------------------------------- #
#  3.  Adaptive network SIS  (Gross, D'Lima & Blasius)        Appendix 6.2.3   #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class Adaptive:
    """Pairwise SIS with rewiring away from infected neighbours at rate ``w``.

    Rewiring breaks the stub identity <SI> + <II> = <I>, so the minimum system is
    genuinely three-dimensional and carries two fast variables, p1 and p2.  Unlike
    the other pairwise models here this one assumes a Poisson degree distribution,
    which is why its triple closures carry a factor n rather than n - 1
    (Appendix 6.2.3, Step 0).  That is the model, not a transcription slip.
    """

    n: int = 5
    w: float = 0.8
    gamma: float = 1.0


def ad_rhs(t: float, y: list[float], tau: float, par: Adaptive) -> list[float]:
    """Exact (i, p1, p2) system -- Eqs. (73)-(75)."""
    n, gam, w = par.n, par.gamma, par.w
    i = np.clip(y[0], _I_MIN, _I_MAX)
    p1, p2 = y[1], y[2]
    r = i / (1 - i)
    return [
        i * (tau * n * p1 - gam),
        gam * p2
        + (tau * (n - 1) - w) * p1
        - tau * n * p1**2
        + tau * n * p1 * (1 - 3 * p1 - p2) * r,
        -gam * p2 + 2 * tau * p1 - tau * n * p1 * p2 + 2 * tau * n * p1**2 * r,
    ]


def ad_fast(i: Scalar, tau: float, par: Adaptive) -> Scalar:
    """Quasi-static p1: non-negative root of A p^2 + B p + C = 0.

    Coefficients Eq. (83), root Eq. (84).  A > 0 and C < 0 near threshold, so the
    non-negative root is unique and lies in [0, 1].
    """
    n, gam, w = par.n, par.gamma, par.w
    A = (n * tau) ** 2 * (1 + i) / (1 - i)
    B = tau * n * (gam + w - tau * (n - 1) + i * (tau - w))
    C = gam * (w - tau * (n + 1) + i * (tau - w))
    return (-B + np.sqrt(np.maximum(B * B - 4 * A * C, 0.0))) / (2 * A)


def ad_fast_p2(i: Scalar, p1: Scalar, tau: float, par: Adaptive) -> Scalar:
    """Second quasi-static fast variable p2, from p1.

    Appendix 6.2.3, Step 3: the relation for gamma p2 derived just above Eq. (83).
    """
    n, gam, w = par.n, par.gamma, par.w
    return (
        tau * n * p1**2 * (1 + i) / (1 - i) - p1 * ((tau - w) * (1 - i) + tau * (n - 2))
    ) / gam


def ad_beta_eff(i: Scalar, p1: Scalar, tau: float, par: Adaptive) -> Scalar:
    """beta_eff = tau n p1 -- Eq. (85)."""
    return tau * par.n * p1


def ad_rhs_1d(t: float, y: list[float], tau: float, par: Adaptive) -> list[float]:
    """1D reduction -- Eq. (85)."""
    i = np.clip(y[0], _I_MIN, _I_MAX)
    return [i * (ad_beta_eff(i, ad_fast(i, tau, par), tau, par) - par.gamma)]


def ad_tau_c(par: Adaptive) -> float:
    """Threshold tau_c n = gamma + w -- Eq. (80).  Rewiring raises it linearly."""
    return (par.gamma + par.w) / par.n


def ad_branch(i: Scalar, par: Adaptive) -> Scalar:
    """Endemic branch tau(i*) in closed form -- Eq. (91).

    Parameterised by prevalence rather than by tau, so it traces the whole S-curve,
    unstable arm included.
    """
    n, gam, w = par.n, par.gamma, par.w
    return (gam + w * (1 - i) ** 2) / ((1 - i) * (n - i))


def ad_w_star(par: Adaptive) -> float:
    """Rewiring rate separating continuous from discontinuous transitions.

    Appendix 6.2.3, Step 4, following Eq. (90):  w* = gamma (n + 1) / (n - 1).
    """
    return par.gamma * (par.n + 1) / (par.n - 1)


def ad_slope_at_threshold(par: Adaptive) -> float:
    """Slope of beta_eff in i at threshold -- Eq. (89).  Changes sign at w = w*."""
    n, gam, w = par.n, par.gamma, par.w
    return 2 * gam * ((n - 1) * w - (n + 1) * gam) / (gam * (2 * n + 1) + w)


# --------------------------------------------------------------------------- #
#  4.  Interacting contagions                                 Appendix 6.2.4   #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class Interacting:
    """Two well-mixed SIS contagions with co-infection enhancement ``alpha``.

    Symmetric case gamma_c = gamma: co-infected individuals clear each infection at
    the same rate as singly-infected ones.  The manuscript's asymmetric coefficients
    (Eq. (104)) reduce to the ones used here when gamma_c = gamma.
    """

    alpha: float = 1.5
    gamma: float = 1.0


def ic_rhs(t: float, y: list[float], tau: float, par: Interacting) -> list[float]:
    """Minimum system in (<SI>, <II>) -- Eqs. (95)-(96).

    Force of infection Phi is Eq. (94); prevalence is i = <SI> + <II>.
    """
    gam = par.gamma
    x, yy = y
    foi = tau * (x + par.alpha * yy)
    return [
        (1 - 2 * x - yy) * foi + gam * yy - gam * x - x * foi,
        2 * tau * x**2 + 2 * par.alpha * tau * yy * x - 2 * gam * yy,
    ]


def ic_fast(i: Scalar, tau: float, par: Interacting) -> Scalar:
    """Quasi-static p = <SI>/<I>: the minus-sign branch -- Eq. (105).

    Coefficients are Eq. (104) at gamma_c = gamma.  A + B + C = -2 tau i vanishes at
    i = 0, which puts the physical root at p*(0) = 1: at vanishing prevalence there
    is no co-infection, so alpha is not yet felt.
    """
    gam, alpha = par.gamma, par.alpha
    A = tau * (alpha - 1) * (1 + i)
    B = -(gam + tau * (2 * alpha - 1 + i))
    C = gam + tau * alpha * (1 - i)
    return (-B - np.sqrt(np.maximum(B * B - 4 * A * C, 0.0))) / (2 * A)


def ic_beta_eff(i: Scalar, p: Scalar, tau: float, par: Interacting) -> Scalar:
    """beta_eff = tau [alpha - (alpha - 1) p] (1 - i) -- Eq. (99).

    The susceptible factor (1 - i) belongs to beta_eff itself in this well-mixed
    setting, where it counts the susceptible neighbours of infected nodes, so the
    prevalence equation is di/dt = i [beta_eff - gamma_eff] with no factor outside.
    Keeping the factor here is what makes d beta_eff / di at i = 0 equal
    gamma (alpha - 2) and change sign at alpha* = 2 (Eq. (112)); without it the
    slope would be gamma (alpha - 1) > 0, misclassifying every alpha > 1 as
    discontinuous.
    """
    return tau * (par.alpha - (par.alpha - 1) * p) * (1 - i)


def ic_rhs_1d(t: float, y: list[float], tau: float, par: Interacting) -> list[float]:
    """1D reduction -- Eq. (106)."""
    i = np.clip(y[0], _I_MIN, _I_MAX)
    return [i * (ic_beta_eff(i, ic_fast(i, tau, par), tau, par) - par.gamma)]


def ic_tau_c(par: Interacting) -> float:
    """Threshold tau_c = gamma -- Eq. (103).

    Independent of alpha: each disease invades as an ordinary SIS process, there
    being no co-infection yet to enhance.
    """
    return par.gamma


def ic_alpha_star(par: Interacting) -> float:
    """Co-infection enhancement separating continuous from discontinuous.

    Eq. (112); equals 2 in the symmetric case.
    """
    return 2.0


def ic_slope_at_threshold(par: Interacting) -> float:
    """Slope of beta_eff in i at threshold -- Eq. (109) at gamma_c = gamma."""
    return par.gamma * (par.alpha - 2)


def ic_invariant_fast(i: Scalar) -> Scalar:
    """Exact invariant p = 1 - i, from <II> = <I>^2.

    The two diseases are statistically independent, so a trajectory follows this
    line exactly.  It agrees with :func:`ic_fast` in value and slope at i = 0 and
    parts company at O(i^2): the quasi-static root is a different curve, which the
    trajectory does not follow away from threshold.
    """
    return 1 - i


# --------------------------------------------------------------------------- #
#  5.  Pairwise SIR                                           Appendix 6.2.5   #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class PairwiseSIR:
    """Pairwise SIR on an n-regular network.

    The reduction terminates at two dimensions rather than one: susceptible
    depletion runs on the same timescale as prevalence, so s is a second slow
    variable and cannot be eliminated (Appendix 6.2.5, preamble).
    """

    n: int = 6
    gamma: float = 1.0


def sir_rhs(t: float, y: list[float], tau: float, par: PairwiseSIR) -> list[float]:
    """Full system in (s, i, <SI>, <SS>) -- Eqs. (119)-(122), closures Eq. (118)."""
    n, gam = par.n, par.gamma
    s, i, SI, SS = y
    s = max(s, 1e-300)
    tri_SSI = (n - 1) * SS * SI / s
    tri_ISI = (n - 1) * SI * SI / s
    return [
        -tau * n * SI,
        tau * n * SI - gam * i,
        tau * tri_SSI - tau * tri_ISI - tau * SI - gam * SI,
        -2.0 * tau * tri_SSI,
    ]


def sir_initial_state(i0: float, par: PairwiseSIR) -> tuple[float, float, float, float]:
    """Random seeding (s0, <SI>0, <SS>0) and the invariant constant C.

    C is fixed by the initial data rather than by the model -- C = s0^(2/n), which
    tends to 1 as s0 -> 1 -- so it is carried alongside the parameters, not inside
    :class:`PairwiseSIR`.
    """
    s0 = 1 - i0
    SI0, SS0 = s0 * i0, s0**2
    C = SS0 / s0 ** (2 * (par.n - 1) / par.n)  # = s0^{2/n}
    return s0, SI0, SS0, C


def sir_fast(i: float, s: float, C: float, par: PairwiseSIR) -> float:
    """Slaved p from the exact invariant sigma = C s^{2(n-1)/n} -- Eqs. (130)-(131).

    Eliminating sigma involves no approximation; only p is slaved.  Scalar-only,
    because of the max() guards.
    """
    n = par.n
    s = max(s, 1e-300)
    ratio = C * s ** ((n - 2) / n)
    return max(((n - 1) * ratio - 1.0) / (n + (n - 1) * i / s), 0.0)


def sir_rhs_2d(
    t: float, y: list[float], tau: float, par: PairwiseSIR, C: float
) -> list[float]:
    """2D reduction in (i, s) -- Eqs. (123)-(124) closed by Eq. (131)."""
    i, s = y
    p = sir_fast(i, s, C, par)
    return [i * (tau * par.n * p - par.gamma), -tau * par.n * p * i]


def sir_fast_1d(i: Scalar, par: PairwiseSIR) -> Scalar:
    """The draft 1D closure p(i) = (n - 2)(1 - i)/(n - i).

    Obtained from Eq. (131) by the two further approximations s -> 1 - i and
    sigma/s -> 1.  Retained for comparison: it produces a spurious stable endemic
    plateau, since an SIR epidemic must die out.
    """
    n = par.n
    return (n - 2) * (1 - i) / (n - i)


def sir_rhs_1d(t: float, y: list[float], tau: float, par: PairwiseSIR) -> list[float]:
    """The draft 1D reduction built on :func:`sir_fast_1d`."""
    i = y[0]
    return [i * (tau * par.n * sir_fast_1d(i, par) - par.gamma)]


def sir_tau_c(par: PairwiseSIR) -> float:
    """Threshold tau_c (n - 2) = gamma -- Eq. (128).

    The n - 2 rather than n - 1 of pairwise SIS: at the disease-free state
    p*(0) = (n - 2)/n (Eq. (127)), one neighbour of a newly infected node being the
    one that infected it.
    """
    return par.gamma / (par.n - 2)


# Any of the five parameter sets, for helpers that are model-agnostic.
ModelParams = PairwiseSIS | Simplicial | Adaptive | Interacting | PairwiseSIR
