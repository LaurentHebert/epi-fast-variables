"""SIS on degree-regular graphs, next reaction method.

SIS on degree-regular graphs, simulated exactly in continuous time with the
next reaction method (Gillespie-style: total rate R, exponential waiting time,
event picked proportionally to its rate).

Events
------
  recovery  : each infected node recovers at rate gamma      -> total gamma * I
  infection : each S-I edge transmits at rate tau            -> total tau * SI

Both event types are picked in O(1): a recovery by choosing a uniformly random
infected node, an infection by choosing a uniformly random S-I edge. Both sets
are kept in "indexed sets" (list + position dict) so insertion, removal and
uniform sampling are all O(1). After each event only the neighbours of the
node that changed state are touched.

Output
------
One long-format CSV (one row per run per integer time t = 0, 1, ..., tmax):

  network, run, t, I, SI, i, si, si_over_i

  I, SI       raw counts (infected nodes, S-I edges)
  i, si       I/N and SI/N       (prevalence <I>; the model's <SI> = si / n)
  si_over_i   SI/I               (mean number of S neighbours per infected node)

State at integer time t is the state of the system just before the first event
occurring after t. Extinct runs (I = 0, absorbing) are padded with zeros, and
si_over_i is empty (NaN) there.
"""

import csv
import math
import random
import sys

import networkx as nx

from simulations.common import IndexedSet


def simulate_sis(adj, tau, gamma, n_seeds, tmax, rng, return_state=False):
    """Run one SIS simulation. Returns a list of (I, SI) at t = 0..tmax.

    With return_state=True, returns (list, final node states) instead.
    """
    N = len(adj)
    state = [0] * N  # 0 = susceptible, 1 = infected
    infected = IndexedSet()
    si = IndexedSet()  # S-I edges, encoded as inf_node * N + sus_node

    # initial conditions: n_seeds distinct infected nodes
    for v in rng.sample(range(N), n_seeds):
        state[v] = 1
        infected.add(v)
    for v in infected.items:
        for w in adj[v]:
            if state[w] == 0:
                si.add(v * N + w)

    out = []
    t = 0.0
    next_record = 0
    while True:
        R_rec = gamma * len(infected)
        R_inf = tau * len(si)
        R = R_rec + R_inf
        t_next = t + rng.expovariate(R) if R > 0 else math.inf

        # record the (unchanged) state at every integer time before the next event
        while next_record <= tmax and next_record < t_next:
            out.append((len(infected), len(si)))
            next_record += 1
        if next_record > tmax or R == 0:
            break
        t = t_next

        if rng.random() * R < R_rec:  # recovery of a uniformly random infected node
            v = infected.choice(rng)
            state[v] = 0
            infected.remove(v)
            for w in adj[v]:
                if state[w] == 1:
                    si.add(w * N + v)  # w (I) -- v (now S)
                else:
                    si.remove(v * N + w)  # v was I, w S: pair disappears
        else:  # infection along a uniformly random S-I edge
            e = si.choice(rng)
            v = e % N  # susceptible endpoint
            state[v] = 1
            infected.add(v)
            for w in adj[v]:
                if state[w] == 1:
                    si.remove(w * N + v)  # pair (w I, v S) disappears
                else:
                    si.add(v * N + w)  # new pair (v I, w S)

    # extinct before tmax: absorbing state, pad with zeros
    while len(out) < tmax + 1:
        out.append((0, 0))
    return (out, state) if return_state else out


def main(
    tau=0.325,  # transmission rate (per S-I edge)
    gamma=1.0,  # recovery rate
    n=5,  # degree
    number_of_networks=100,
    number_of_simulations=100,
    network_size=10000,
    n_seeds=20,
    tmax=100,
    seed=None,
    outfile="simulations/sis_runs.csv",
    verbose=False,
):
    """Simulate and write the runs CSV (columns in the module docstring)."""
    rng = random.Random(seed)
    with open(outfile, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["network", "run", "t", "I", "SI", "i", "si", "si_over_i"])
        for net in range(number_of_networks):
            G = nx.random_regular_graph(d=n, n=network_size, seed=rng.randrange(2**31))
            adj = [list(G[v]) for v in range(network_size)]  # nodes are 0..N-1
            for run in range(number_of_simulations):
                traj = simulate_sis(adj, tau, gamma, n_seeds, tmax, rng)
                for t, (n_inf, SI) in enumerate(traj):
                    w.writerow(
                        [
                            net,
                            run,
                            t,
                            n_inf,
                            SI,
                            f"{n_inf / network_size:.6g}",
                            f"{SI / network_size:.6g}",
                            f"{SI / n_inf:.6g}" if n_inf > 0 else "",
                        ]
                    )
            if verbose:
                print(f"network {net + 1}/{number_of_networks} done", file=sys.stderr)


if __name__ == "__main__":
    main(verbose="--verbose" in sys.argv)
