"""SIR on degree-regular graphs, next reaction method.

SIR on degree-regular graphs, simulated exactly in continuous time with the
next reaction method (same machinery as sis_nrm.py).

Events
------
  recovery  : each infected node recovers (I -> R) at rate gamma  -> total gamma * I
  infection : each S-I edge transmits at rate tau                 -> total tau * SI

Infected nodes and S-I edges are kept in indexed sets (O(1) add / remove / uniform
choice). Recovery never creates pairs, infection creates S-I edges only towards the
susceptible neighbours of the newly infected node. S-S edges are tracked by a counter.

Output
------
One long-format CSV, one row per run per integer time t = 0, 1, ..., tmax:

  network, run, t, S, I, SI, SS, s, i, si, ss, si_over_i

  S, I, SI, SS   raw counts (S nodes, I nodes, S-I edges, S-S edges)
  s, i, si, ss   the same divided by N
  si_over_i      SI / I (empty when I = 0)

State at integer time t is the state just before the first event after t. Once the
epidemic is over (I = 0, absorbing) the final state is repeated up to tmax.
"""

import csv
import math
import random
import sys

import networkx as nx

from simulations.common import IndexedSet


def simulate_sir(adj, tau, gamma, n_seeds, tmax, rng, return_state=False):
    """Run one SIR simulation. Returns a list of (S, I, SI, SS) at t = 0..tmax."""
    N = len(adj)
    S_, I_, R_ = 0, 1, 2
    state = [S_] * N
    infected = IndexedSet()
    si = IndexedSet()  # S-I edges, encoded as inf_node * N + sus_node

    for v in rng.sample(range(N), n_seeds):
        state[v] = I_
        infected.add(v)
    for v in infected.items:
        for w in adj[v]:
            if state[w] == S_:
                si.add(v * N + w)
    nS = N - n_seeds
    nSS = (
        sum(1 for v in range(N) if state[v] == S_ for w in adj[v] if state[w] == S_)
        // 2
    )

    out = []
    t = 0.0
    next_record = 0
    while True:
        R_rec = gamma * len(infected)
        R_inf = tau * len(si)
        R = R_rec + R_inf
        t_next = t + rng.expovariate(R) if R > 0 else math.inf

        while next_record <= tmax and next_record < t_next:
            out.append((nS, len(infected), len(si), nSS))
            next_record += 1
        if next_record > tmax or R == 0:
            break
        t = t_next

        if rng.random() * R < R_rec:  # recovery
            v = infected.choice(rng)
            state[v] = R_
            infected.remove(v)
            for w in adj[v]:
                if state[w] == S_:
                    si.remove(v * N + w)
        else:  # infection along a uniformly random S-I edge
            e = si.choice(rng)
            v = e % N
            state[v] = I_
            infected.add(v)
            nS -= 1
            for w in adj[v]:
                if state[w] == I_:
                    si.remove(w * N + v)
                elif state[w] == S_:
                    si.add(v * N + w)
                    nSS -= 1

    while len(out) < tmax + 1:
        out.append(out[-1])
    return (out, state) if return_state else out


def main(
    tau=0.325,  # transmission rate (per S-I edge)
    gamma=1.0,  # recovery rate
    n=6,  # degree
    number_of_networks=5,
    number_of_simulations=20,
    network_size=10000,
    n_seeds=20,
    tmax=100,
    seed=None,
    outfile="simulations/sir_runs.csv",
    verbose=False,
):
    """Simulate and write the runs CSV (columns in the module docstring)."""
    rng = random.Random(seed)
    N = network_size
    with open(outfile, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "network",
                "run",
                "t",
                "S",
                "I",
                "SI",
                "SS",
                "s",
                "i",
                "si",
                "ss",
                "si_over_i",
            ]
        )
        for net in range(number_of_networks):
            G = nx.random_regular_graph(d=n, n=N, seed=rng.randrange(2**31))
            adj = [list(G[v]) for v in range(N)]
            for run in range(number_of_simulations):
                traj = simulate_sir(adj, tau, gamma, n_seeds, tmax, rng)
                for t, (S, n_inf, SI, SS) in enumerate(traj):
                    w.writerow(
                        [
                            net,
                            run,
                            t,
                            S,
                            n_inf,
                            SI,
                            SS,
                            f"{S / N:.6g}",
                            f"{n_inf / N:.6g}",
                            f"{SI / N:.6g}",
                            f"{SS / N:.6g}",
                            f"{SI / n_inf:.6g}" if n_inf > 0 else "",
                        ]
                    )
            if verbose:
                print(f"network {net + 1}/{number_of_networks} done", file=sys.stderr)


if __name__ == "__main__":
    main(verbose="--verbose" in sys.argv)
