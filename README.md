# Figures for *Fast-variable reduction to complex contagion unifies epidemic models*

Four figure scripts based on the model equations. Project by L. Hébert-Dufresne, P. Simon, and I. Kiss.

## Running

The project is managed with [uv](https://docs.astral.sh/uv/). One-time setup:

```sh
uv sync
```

Then each figure builds itself:

```sh
uv run figure1.py     # figure1.pdf, figure1.png -- beta_eff at threshold
uv run figure2.py     # figure2.pdf, figure2.png -- 4x3 reduction of four models
uv run figure3.py     # figure3.pdf, figure3.png -- reduction fitted from trajectories
uv run figure4.py     # figure4.pdf             -- pairwise SIR, 2D reduction
```

A successful run prints nothing. Diagnostics are available behind flags:

| flag | script | what it prints |
|---|---|---|
| `--check` | 1, 2, 4 | numerical audits: thresholds, slopes against Table 1, closed forms against numerical roots, reduction error |
| `--verbose` | 3 | where each transient was cut, and the rms residual of each polynomial fit |
| `--invariant` | 1 | also overlays the exact interacting-contagion invariant p = 1 − i |

`uv run figure1.py --check` and friends do not write a figure.

After `uv sync` the scripts also run under the project's interpreter directly
(`.venv/bin/python figure1.py`). They will *not* run under a bare system `python`
unless numpy, scipy and matplotlib happen to be installed there.

## Layout

```
models.py     all five models: full RHS, quasi-static fast variable, beta_eff,
              thresholds, critical parameters.  No matplotlib.
style.py      shared rcParams and the colour palette, by role.
figure1..4.py one figure each: layout, and the data pipeline that figure needs.
```

`models.py` groups each model's constants in a frozen dataclass — `PairwiseSIS`,
`Simplicial`, `Adaptive`, `Interacting`, `PairwiseSIR` — and gives every model the
same verbs:

```python
from models import Simplicial, sim_fast, sim_beta_eff, sim_tau_c

par = Simplicial(n=5, n_tri=3, beta=0.4)
p = sim_fast(i=0.1, tau=0.3, par=par)
b = sim_beta_eff(0.1, p, 0.3, par)
```

`beta_eff` always takes `p` from the caller rather than computing it, which is what
lets Figure 2's analytic quasi-static root and Figure 3's fitted `p(i)` go through
the same function definition.

Comments cite the manuscript by section and equation — for example
`Eq. (44)`, `Appendix 6.2.3, Step 3`.
