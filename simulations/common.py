"""Helpers shared by the simulators and the figure rows in this folder.

Simulators use :class:`IndexedSet`; the figure rows share the transient cut, the pooled
measurement of the slow manifold, and the alignment of runs in time.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

import figure3 as f3

KAPPA = f3.KAPPA  # transient cut: |dp/dt| < KAPPA |di/dt| ...
WINDOW = 3  # ... on this many consecutive recording times
NBINS = 40  # bins uniform in i for the pooled manifold
MIN_INFECTED = 30  # run-times with fewer infected nodes are too noisy to pool
MIN_BIN_COUNT = 5  # bins with fewer pooled run-times are dropped


@dataclass(frozen=True, slots=True)
class RowInfo:
    """What a figure row reports back: its parameter label and fit diagnostics."""

    label: str
    notes: dict[str, float] = field(default_factory=dict)


class IndexedSet:
    """Set with O(1) add, remove and uniform random choice."""

    def __init__(self):
        """Create an empty set."""
        self.items = []
        self.pos = {}

    def __len__(self):
        """Number of elements."""
        return len(self.items)

    def add(self, x):
        """Insert x (must be absent)."""
        self.pos[x] = len(self.items)
        self.items.append(x)

    def remove(self, x):
        """Remove x (must be present)."""
        i = self.pos.pop(x)
        last = self.items.pop()
        if i < len(self.items):  # x was not the last element
            self.items[i] = last
            self.pos[last] = i

    def choice(self, rng):
        """Uniformly random element."""
        return self.items[int(rng.random() * len(self.items))]


def load_runs(
    path: str, columns: tuple[str, ...]
) -> tuple[NDArray, dict[str, NDArray], int]:
    """Read a long-format run CSV: times, {column: (runs, times)} and the system size N.

    Rows are ordered by (network, run, t); files without a network column by (run, t).
    N is recovered from the first row as I / i.
    """
    raw = np.genfromtxt(path, delimiter=",", names=True, dtype=float)
    keys = [k for k in ("network", "run", "t") if k in raw.dtype.names]
    raw = raw[np.lexsort([raw[k] for k in reversed(keys)])]
    first = np.ones(len(raw), bool)
    for k in keys[:-1]:
        first &= raw[k] == raw[k][0]
    ntimes = int(first.sum())
    if len(raw) % ntimes:
        raise ValueError("CSV is not a complete set of runs of equal length")
    data = {c: raw[c].reshape(-1, ntimes) for c in columns}
    return raw["t"][:ntimes], data, round(raw["I"][0] / raw["i"][0])


def align_to_prevalence(
    t: NDArray, n_inf: NDArray, N: int, thresh: float, *arrays: NDArray
) -> tuple[NDArray, ...]:
    """Shift each run so that t = 0 is its first point with I / N >= thresh.

    Runs that never get there are dropped, and all runs are cut to the length the
    latest-starting run still has.  Returns (t, I, *arrays) with the shifted data;
    the arrays are (runs, times) and share I's time grid.
    """
    reach = n_inf >= thresh * N
    keep = reach.any(1)
    start = reach.argmax(1)[keep]
    length = n_inf.shape[1] - start.max()
    shifted = [
        np.stack(
            [
                a[r, k : k + length]
                for r, k in zip(np.flatnonzero(keep), start, strict=True)
            ]
        )
        for a in (n_inf, *arrays)
    ]
    return (t[:length], *shifted)


def transient_end(i: NDArray, p: NDArray) -> int:
    """First index from which the mean trajectory sits on the slow manifold."""
    di, dp = np.gradient(i), np.gradient(p)
    on = np.abs(dp) < KAPPA * np.abs(di)
    for k in range(len(on) - WINDOW + 1):
        if on[k : k + WINDOW].all():
            return k
    raise RuntimeError("no on-manifold stretch found; pass a longer run")


def pooled_manifold(
    n_inf: NDArray, SI: NDArray, N: int, n: int, t: NDArray, k0: int
) -> tuple[NDArray, NDArray]:
    """Measure p(i) from individual runs: (mean i, p) per uniform bin in i."""
    sel = (n_inf >= MIN_INFECTED) & (t[None, :] >= t[k0])
    i, Isel, SIsel = (n_inf / N)[sel], n_inf[sel], SI[sel]
    edges = np.linspace(i.min(), i.max(), NBINS + 1)
    idx = np.clip(np.digitize(i, edges) - 1, 0, NBINS - 1)
    ib, pb = [], []
    for b in range(NBINS):
        m = idx == b
        if m.sum() >= MIN_BIN_COUNT:
            ib.append(i[m].mean())
            pb.append(SIsel[m].sum() / (n * Isel[m].sum()))
    return np.array(ib), np.array(pb)
