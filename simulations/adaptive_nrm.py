"""Adaptive-network SIS (Gross et al.) on Poisson graphs, next reaction method.

Adaptive network SIS (Gross, D'Lima & Blasius) on Poisson random graphs, simulated
exactly in continuous time with the next reaction method (same machinery as
sis_nrm.py).

Model
-----
Erdos-Renyi graph G(N, M) with M = n N / 2 edges (Poisson degrees, mean degree n).
Every S-I edge carries two competing processes:

  infection : the S end becomes infected                    rate tau
  rewiring  : the S end drops the edge and attaches it to a
              uniformly random other susceptible node       rate w

and every infected node recovers at rate gamma. The number of edges is conserved, but
infected nodes lose links, so the graph changes during the run (each run works on its
own copy of the adjacency).

Events are chosen as in the next reaction method:  R = gamma I + (tau + w) SI;
recovery with probability gamma I / R, otherwise a uniformly random S-I edge, which
infects with probability tau / (tau + w) and is rewired otherwise.

Output
------
One long-format CSV, one row per run per recording time t = 0, dt, 2 dt, ..., tmax:

  network, run, t, I, SI, II, i, si, ii, si_over_i, ii_over_i

  I, SI, II        raw counts (infected nodes, S-I edges, I-I edges)
  i, si, ii        the same divided by N
  si_over_i        SI / I   (empty when I = 0)
  ii_over_i        II / I   (empty when I = 0)

The model's <SI> and <II> are per-stub densities, <SI> = SI / (n N) and
<II> = 2 II / (n N), so the fast variables are
  p1 = <SI>/<I> = SI / (n I)      and      p2 = <II>/<I> = 2 II / (n I).
The columns si and ii below are plain per-node densities, SI / N and II / N.
State at recording time t is the state just before the first event after t. Extinct
runs (I = 0, absorbing) are padded with zeros.
"""

import csv
import math
import random
import sys

import networkx as nx

from simulations.common import IndexedSet


def simulate_adaptive(
    base_adj, tau, gamma, w, n_seeds, tmax, dt, rng, return_state=False
):
    """Simulate and write the runs CSV (columns in the module docstring)."""
    """Run one adaptive-SIS simulation; list of (I, SI, II) at t = 0, dt, ..."""
    N = len(base_adj)
    adj = [set(a) for a in base_adj]  # rewiring modifies the graph
    state = [0] * N  # 0 = susceptible, 1 = infected
    infected = IndexedSet()
    susceptible = IndexedSet()
    si = IndexedSet()  # S-I edges, encoded as inf_node * N + sus_node
    p_inf = tau / (tau + w)

    seeds = rng.sample(range(N), n_seeds)
    for v in seeds:
        state[v] = 1
    for v in range(N):
        (infected if state[v] else susceptible).add(v)
    nII = 0
    for v in seeds:
        for x in adj[v]:
            if state[x] == 0:
                si.add(v * N + x)
            else:
                nII += 1
    nII //= 2

    nrec = round(tmax / dt) + 1  # number of recording times
    out = []
    t = 0.0
    k = 0  # next recording index
    while True:
        R_rec = gamma * len(infected)
        R_si = (tau + w) * len(si)
        R = R_rec + R_si
        t_next = t + rng.expovariate(R) if R > 0 else math.inf

        while k < nrec and k * dt < t_next:
            out.append((len(infected), len(si), nII))
            k += 1
        if k >= nrec or R == 0:
            break
        t = t_next

        if rng.random() * R < R_rec:  # recovery
            v = infected.choice(rng)
            state[v] = 0
            infected.remove(v)
            susceptible.add(v)
            for x in adj[v]:
                if state[x] == 1:
                    nII -= 1
                    si.add(x * N + v)
                else:
                    si.remove(v * N + x)
        else:  # an S-I edge: infection or rewiring
            e = si.choice(rng)
            u, v = divmod(e, N)  # u infected, v susceptible
            if rng.random() < p_inf:  # infection of v
                state[v] = 1
                susceptible.remove(v)
                infected.add(v)
                for x in adj[v]:
                    if state[x] == 1:
                        si.remove(x * N + v)
                        nII += 1
                    else:
                        si.add(v * N + x)
            else:  # v rewires its link to u onto a random other susceptible node
                for _ in range(100):
                    x = susceptible.choice(rng)
                    if x != v and x not in adj[v]:
                        adj[u].discard(v)
                        adj[v].discard(u)
                        si.remove(e)
                        adj[v].add(x)
                        adj[x].add(v)
                        break

    while len(out) < nrec:
        out.append((0, 0, 0))
    return (out, state, adj) if return_state else out


def main(
    tau=1.04,  # transmission rate (per S-I edge); 1.3 x tau_c for w = 3
    gamma=1.0,  # recovery rate
    w=3.0,  # rewiring rate (per S-I edge)
    n=5,  # mean degree
    number_of_networks=5,
    number_of_simulations=20,
    network_size=10000,
    n_seeds=20,
    tmax=30,
    dt=0.25,
    seed=None,
    outfile="simulations/adaptive_runs.csv",
    verbose=False,
):
    """Simulate and write the runs CSV (columns in the module docstring)."""
    rng = random.Random(seed)
    N = network_size
    with open(outfile, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(
            [
                "network",
                "run",
                "t",
                "I",
                "SI",
                "II",
                "i",
                "si",
                "ii",
                "si_over_i",
                "ii_over_i",
            ]
        )
        for net in range(number_of_networks):
            G = nx.gnm_random_graph(N, N * n // 2, seed=rng.randrange(2**31))
            base_adj = [list(G[v]) for v in range(N)]  # nodes are 0..N-1
            for run in range(number_of_simulations):
                traj = simulate_adaptive(
                    base_adj, tau, gamma, w, n_seeds, tmax, dt, rng
                )
                for k, (n_inf, SI, II) in enumerate(traj):
                    wr.writerow(
                        [
                            net,
                            run,
                            f"{k * dt:g}",
                            n_inf,
                            SI,
                            II,
                            f"{n_inf / N:.6g}",
                            f"{SI / N:.6g}",
                            f"{II / N:.6g}",
                            f"{SI / n_inf:.6g}" if n_inf > 0 else "",
                            f"{II / n_inf:.6g}" if n_inf > 0 else "",
                        ]
                    )
            if verbose:
                print(f"network {net + 1}/{number_of_networks} done", file=sys.stderr)


if __name__ == "__main__":
    main(verbose="--verbose" in sys.argv)
