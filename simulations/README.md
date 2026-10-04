# simulations/

Exact continuous-time (next reaction method) simulations behind Figure 5, kept apart
from the figure scripts because some take a while to generate (higher-order SIS the
longest). `figure5.py` reads the CSVs in this folder; it never runs a simulation.

| model | simulator | runs read by Figure 5 |
|---|---|---|
| pairwise SIS | `sis_nrm.py` | `sis_runs_test.csv` |
| pairwise SIR | `sir_nrm.py` | `sir_runs.csv` |
| higher-order SIS | `simplicial_nrm.py` | `simplicial_runs_x4b.csv` |
| adaptive SIS | `adaptive_nrm.py` | `adaptive_runs_x4.csv` |
| interacting contagions | `interacting_nrm.py` | `interacting_runs.csv` |

`*_row.py` draw one row of Figure 5 (simulation vs. reduction); `common.py` holds what
they and the simulators share (`IndexedSet`, CSV loading, time alignment, manifold fit).

## Regenerating the data (run from the repository root)

    python -m simulations.sis_nrm [--verbose]       # writes simulations/sis_runs.csv

The other simulators work the same way; parameters are arguments of `main()`. The runs
in the folder were produced with:

    # higher-order, n = 20, n_tri = 12 (4x denser than n = 5, n_tri = 3), 200 seeds
    python -c "import simulations.simplicial_nrm as S; S.main(tau=0.015461, beta=1.3, n=20, n_tri=12, n_seeds=200, tmax=40, dt=0.25, seed=23, outfile='simulations/simplicial_runs_x4b.csv')"
    # adaptive, n = 20
    python -c "import simulations.adaptive_nrm as A; A.main(tau=0.26, w=3.0, n=20, tmax=15, dt=0.25, seed=22, outfile='simulations/adaptive_runs_x4.csv')"

The SIS, SIR and interacting runs use the `main()` defaults (the SIS and SIR files hold
5 networks x 20 runs); the filenames differ from the defaults, so pass `outfile=`.
