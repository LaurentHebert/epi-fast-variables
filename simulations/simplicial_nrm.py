"""Simplicial SIS (Malizia et al.) on a random hypergraph, next reaction method.

Simplicial SIS (Malizia et al.) on a degree-regular hypergraph with a fixed number of
pairwise edges and triangles per node, simulated exactly in continuous time with the
next reaction method (same machinery as sis_nrm.py).

Hypergraph
----------
Simplicial complex with the inclusion property (Malizia et al.): every node has n_tri
triangles (stubs paired in groups of three, configuration style), and the three edges of
each triangle are also pairwise edges that transmit at tau. In addition every node has n
pairwise edges that belong to no triangle. The total pairwise degree is therefore
n + 2 n_tri (= 11 for n = 5, n_tri = 3), and that is the "n" the pair-based model sees.
The model assumes no overlap: any structure that would share two nodes with one already
accepted (triangles first, then the extra edges), as well as self-loops and repeated
nodes, is deleted. At N = 10^4 this removes about 0.1% of the structures; the builder
reports what was dropped. ``inclusion=False`` gives the older construction with
triangles disjoint from the edges.

Dynamics
--------
  recovery            each infected node recovers at rate gamma
  edge infection      each S-I edge (triangle edges included) transmits at rate tau
  triangle infection  each triangle with exactly two infected nodes infects its
                      susceptible node at rate beta

Events: R = gamma I + tau SI + beta T2, with SI the number of S-I edges, T2 the number
of triangles with exactly two infected nodes. A recovery picks a uniform infected node,
an infection a uniform S-I edge or a uniform two-infected triangle; all three sets are
indexed sets (O(1) add / remove / choice). Only the edges and triangles of the node that
changes state are touched.

Initial condition
-----------------
n_seeds infected nodes, by default placed as two-infected triangles (n_seeds / 2
triangles, two nodes each), so that the three-body channel is active from the start and
runs are not lost to early stochastic extinction.

Output
------
One long-format CSV, one row per run per recording time t = 0, dt, 2 dt, ..., tmax:

  network, run, t, I, SI, T2, i, si, si_over_i

  I, SI, T2   raw counts (infected nodes, S-I edges, two-infected triangles)
  i, si       I/N, SI/N      (prevalence <I>; the model's <SI> = si / k)
  si_over_i   SI / I         (empty when I = 0); the model's fast variable is
                             p = SI / (k I), k = n + 2 n_tri the total degree

State at recording time t is the state just before the first event after t. Extinct runs
(I = 0, absorbing) are padded with zeros.
"""

import csv
import math
import random
import sys
from multiprocessing import Pool

from simulations.common import IndexedSet


def build_hypergraph(N, n, n_tri, rng, inclusion="extra"):
    """Random hypergraph with n edges and n_tri triangles per node, overlaps deleted.

    Returns (adj, tris, tri_of, stats): adj[v] the edge neighbours of v, tris the list
    of triangles (a, b, c), tri_of[v] the indices of the triangles containing v.
    """
    if (N * n_tri) % 3 or (N * (n - 2 * n_tri if inclusion is True else n)) % 2:
        raise ValueError("N * n_tri must be divisible by 3 and N * n by 2")

    def pair(a, b):
        return (a * N + b) if a < b else (b * N + a)

    occupied = set()  # node pairs shared by an accepted structure

    stubs = [v for v in range(N) for _ in range(n_tri)]
    rng.shuffle(stubs)
    tris, dropped_tri = [], 0
    for k in range(0, len(stubs), 3):
        a, b, c = stubs[k : k + 3]
        if a == b or b == c or a == c:
            dropped_tri += 1
            continue
        ps = (pair(a, b), pair(b, c), pair(a, c))
        if any(p in occupied for p in ps):
            dropped_tri += 1
            continue
        occupied.update(ps)
        tris.append((a, b, c))

    adj = [[] for _ in range(N)]
    n_extra = n
    if (
        inclusion
    ):  # simplicial complex: the three edges of a triangle are pairwise edges
        if inclusion != "extra" and n < 2 * n_tri:
            raise ValueError("inclusion=True needs n >= 2 * n_tri")
        for a, b, c in tris:
            for x, y in ((a, b), (b, c), (a, c)):
                adj[x].append(y)
                adj[y].append(x)
        # 'extra': n more pairwise edges outside triangles (degree n + 2 n_tri);
        # True: n counts the triangle edges too
        n_extra = n if inclusion == "extra" else n - 2 * n_tri
    stubs = [v for v in range(N) for _ in range(n_extra)]
    rng.shuffle(stubs)
    dropped_edge = 0
    for k in range(0, len(stubs), 2):
        a, b = stubs[k], stubs[k + 1]
        if a == b or pair(a, b) in occupied:
            dropped_edge += 1
            continue
        occupied.add(pair(a, b))
        adj[a].append(b)
        adj[b].append(a)

    tri_of = [[] for _ in range(N)]
    for idx, tri in enumerate(tris):
        for v in tri:
            tri_of[v].append(idx)
    stats = dict(
        n_edges=sum(len(a) for a in adj) // 2,
        n_tris=len(tris),
        dropped_edges=dropped_edge,
        dropped_tris=dropped_tri,
        mean_edge_degree=sum(len(a) for a in adj) / N,
        mean_tri_degree=3 * len(tris) / N,
    )
    return adj, tris, tri_of, stats


def simulate_simplicial(
    hg,
    tau,
    beta,
    gamma,
    n_seeds,
    tmax,
    dt,
    rng,
    return_state=False,
    seed_triangles=False,
):
    """Simulate and write the runs CSV (columns in the module docstring)."""
    """Run one simulation. Returns a list of (I, SI, T2) at t = 0, dt, 2 dt, ..."""
    adj, tris, tri_of, _ = hg
    N = len(adj)
    state = [0] * N  # 0 = susceptible, 1 = infected
    infected = IndexedSet()
    si = IndexedSet()  # S-I edges, encoded as inf_node * N + sus_node
    iis = IndexedSet()  # triangles with exactly two infected nodes
    tri_inf = [0] * len(tris)  # infected nodes per triangle

    if seed_triangles:  # n_seeds nodes as n_seeds/2 triangles, two infected nodes each
        chosen = set()
        for d in rng.sample(range(len(tris)), len(tris)):
            if len(chosen) >= n_seeds:
                break
            if any(v in chosen for v in tris[d]):
                continue
            chosen.update(rng.sample(tris[d], 2))
        seeds = chosen
    else:
        seeds = rng.sample(range(N), n_seeds)
    for v in seeds:
        state[v] = 1
        infected.add(v)
    for v in infected.items:
        for w in adj[v]:
            if state[w] == 0:
                si.add(v * N + w)
        for d in tri_of[v]:
            tri_inf[d] += 1
    for d, c in enumerate(tri_inf):
        if c == 2:
            iis.add(d)

    def infect(v):
        state[v] = 1
        infected.add(v)
        for w in adj[v]:
            if state[w] == 1:
                si.remove(w * N + v)
            else:
                si.add(v * N + w)
        for d in tri_of[v]:
            old = tri_inf[d]
            if old == 2:
                iis.remove(d)
            tri_inf[d] = old + 1
            if old + 1 == 2:
                iis.add(d)

    nrec = round(tmax / dt) + 1
    out = []
    t = 0.0
    k = 0
    while True:
        R_rec = gamma * len(infected)
        R_edge = tau * len(si)
        R_tri = beta * len(iis)
        R = R_rec + R_edge + R_tri
        t_next = t + rng.expovariate(R) if R > 0 else math.inf

        while k < nrec and k * dt < t_next:
            out.append((len(infected), len(si), len(iis)))
            k += 1
        if k >= nrec or R == 0:
            break
        t = t_next

        x = rng.random() * R
        if x < R_rec:  # recovery
            v = infected.choice(rng)
            state[v] = 0
            infected.remove(v)
            for w in adj[v]:
                if state[w] == 1:
                    si.add(w * N + v)
                else:
                    si.remove(v * N + w)
            for d in tri_of[v]:
                old = tri_inf[d]
                if old == 2:
                    iis.remove(d)
                tri_inf[d] = old - 1
                if old - 1 == 2:
                    iis.add(d)
        elif x < R_rec + R_edge:  # infection along an S-I edge
            infect(si.choice(rng) % N)
        else:  # infection inside a two-infected triangle
            a, b, c = tris[iis.choice(rng)]
            infect(a if state[a] == 0 else b if state[b] == 0 else c)

    while len(out) < nrec:
        out.append((0, 0, 0))
    return (out, state, tri_inf) if return_state else out


def _one_network(args):
    net, seed, N, n, n_tri, tau, beta, gamma, n_seeds, tmax, dt, nsim, seed_tri = args
    rng = random.Random(seed)
    hg = build_hypergraph(N, n, n_tri, rng)
    rows = []
    for run in range(nsim):
        traj = simulate_simplicial(
            hg, tau, beta, gamma, n_seeds, tmax, dt, rng, seed_triangles=seed_tri
        )
        for k, (n_inf, SI, T2) in enumerate(traj):
            rows.append(
                [
                    net,
                    run,
                    f"{k * dt:g}",
                    n_inf,
                    SI,
                    T2,
                    f"{n_inf / N:.6g}",
                    f"{SI / N:.6g}",
                    f"{SI / n_inf:.6g}" if n_inf > 0 else "",
                ]
            )
    return net, rows, hg[3]


def main(
    # edge transmission rate: 1.3 x tau_c of the model, n = 11, n_tri = 3, beta = 1.3
    tau=0.08469,
    beta=1.3,  # simplicial (triangle) transmission rate
    gamma=1.0,  # recovery rate
    n=5,  # pairwise degree
    n_tri=3,  # triangle degree
    number_of_networks=5,
    number_of_simulations=20,
    network_size=10000,
    n_seeds=20,  # infected at t = 0, placed as two-infected triangles
    seed_triangles=True,
    tmax=100,
    dt=0.5,
    seed=None,
    n_jobs=2,
    outfile="simulations/simplicial_runs.csv",
    verbose=False,
):
    """Simulate and write the runs CSV (columns in the module docstring)."""
    master = random.Random(seed)
    jobs = [
        (
            net,
            master.randrange(2**63),
            network_size,
            n,
            n_tri,
            tau,
            beta,
            gamma,
            n_seeds,
            tmax,
            dt,
            number_of_simulations,
            seed_triangles,
        )
        for net in range(number_of_networks)
    ]
    with open(outfile, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["network", "run", "t", "I", "SI", "T2", "i", "si", "si_over_i"])
        with Pool(n_jobs) as pool:
            for net, rows, st in pool.imap(_one_network, jobs):
                w.writerows(rows)
                if verbose:
                    print(
                        f"network {net + 1}/{number_of_networks} done  "
                        f"(dropped {st['dropped_edges']} edges, "
                        f"{st['dropped_tris']} triangles; mean degrees "
                        f"{st['mean_edge_degree']:.3f}, {st['mean_tri_degree']:.3f})",
                        file=sys.stderr,
                    )


if __name__ == "__main__":
    main(verbose="--verbose" in sys.argv)
