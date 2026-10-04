"""Two interacting SIS contagions in a well-mixed population, next reaction method.

Two interacting SIS contagions in a well-mixed (fully connected) population, simulated
exactly in continuous time with the next reaction method.

Model (Hébert-Dufresne et al.; Appendix 6.2.4 of the manuscript, symmetric case)
-----
Each of N individuals has a joint state (disease 1, disease 2) in {S, I}^2. Let n10,
n01, n11 be the numbers of individuals infected with disease 1 only, disease 2 only,
and both (n00 = N - n10 - n01 - n11).

  infection  : every individual susceptible to disease d is infected at rate
                 Phi_d = tau (n_d_only + alpha n11) / N,
               i.e. a co-infected carrier transmits each disease alpha times faster.
  recovery   : a singly-infected individual recovers at rate gamma; a co-infected one
               clears each of its two infections at rate gamma_c.

Eight event channels (00->10, 01->11, 00->01, 10->11, 10->00, 01->00, 11->01, 11->10);
total rate R, exponential waiting time, channel chosen proportionally to its rate.
The initial condition seeds n_seeds individuals with disease 1 only and n_seeds others
with disease 2 only, so each disease starts at prevalence n_seeds / N, no co-infection.

Output
------
One long-format CSV, one row per run per recording time t = 0, dt, 2 dt, ..., tmax:

  run, t, n10, n01, n11, I, SI, i, si, si_over_i

  I   = (n10 + n01)/2 + n11   infected count with one disease, averaged over the two
  SI  = (n10 + n01)/2         singly-infected count (the model's <SI> N), same average
  i, si                       I/N, SI/N      (prevalence <I> and <SI>)
  si_over_i                   SI / I = p, the fast variable (empty when I = 0)

State at recording time t is the state just before the first event after t. Runs in
which both diseases are extinct (absorbing) are padded with zeros.
"""

import csv
import math
import random
import sys
from multiprocessing import Pool


def simulate_interacting(N, tau, gamma, gamma_c, alpha, n_seeds, tmax, dt, rng):
    """Run one simulation. Returns a list of (n10, n01, n11) at t = 0, dt, ..."""
    n10, n01, n11 = n_seeds, n_seeds, 0
    n00 = N - 2 * n_seeds
    nrec = round(tmax / dt) + 1
    out = []
    t = 0.0
    k = 0
    inv_N = 1.0 / N
    while True:
        phi1 = tau * (n10 + alpha * n11) * inv_N
        phi2 = tau * (n01 + alpha * n11) * inv_N
        r1 = n00 * phi1  # 00 -> 10
        r2 = n01 * phi1  # 01 -> 11
        r3 = n00 * phi2  # 00 -> 01
        r4 = n10 * phi2  # 10 -> 11
        r5 = gamma * n10  # 10 -> 00
        r6 = gamma * n01  # 01 -> 00
        r7 = gamma_c * n11  # 11 -> 01 (disease 1 cleared)
        r8 = r7  # 11 -> 10 (disease 2 cleared)
        R = r1 + r2 + r3 + r4 + r5 + r6 + r7 + r8
        t_next = t + rng.expovariate(R) if R > 0 else math.inf

        while k < nrec and k * dt < t_next:
            out.append((n10, n01, n11))
            k += 1
        if k >= nrec or R == 0:
            break
        t = t_next

        x = rng.random() * R
        if x < r1:
            n00 -= 1
            n10 += 1
        elif x < r1 + r2:
            n01 -= 1
            n11 += 1
        elif x < r1 + r2 + r3:
            n00 -= 1
            n01 += 1
        elif x < r1 + r2 + r3 + r4:
            n10 -= 1
            n11 += 1
        elif x < r1 + r2 + r3 + r4 + r5:
            n10 -= 1
            n00 += 1
        elif x < r1 + r2 + r3 + r4 + r5 + r6:
            n01 -= 1
            n00 += 1
        elif x < r1 + r2 + r3 + r4 + r5 + r6 + r7:
            n11 -= 1
            n01 += 1
        else:
            n11 -= 1
            n10 += 1

    while len(out) < nrec:
        out.append((0, 0, 0))
    return out


def _one_run(args):
    run, seed, N, tau, gamma, gamma_c, alpha, n_seeds, tmax, dt = args
    rng = random.Random(seed)
    return run, simulate_interacting(
        N, tau, gamma, gamma_c, alpha, n_seeds, tmax, dt, rng
    )


def main(
    tau=1.3,  # transmission rate; tau_c = gamma, so this is 1.3 tau_c
    gamma=1.0,  # recovery rate of singly-infected individuals
    gamma_c=1.0,  # clearance rate of each infection when co-infected
    alpha=3.0,  # co-infection enhancement
    number_of_simulations=100,
    population_size=10000,
    n_seeds=20,
    tmax=50,
    dt=0.25,
    seed=None,
    n_jobs=2,
    outfile="simulations/interacting_runs.csv",
    verbose=False,
):
    """Simulate and write the runs CSV (columns in the module docstring)."""
    N = population_size
    master = random.Random(seed)
    jobs = [
        (r, master.randrange(2**63), N, tau, gamma, gamma_c, alpha, n_seeds, tmax, dt)
        for r in range(number_of_simulations)
    ]
    with open(outfile, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "t", "n10", "n01", "n11", "I", "SI", "i", "si", "si_over_i"])
        with Pool(n_jobs) as pool:
            for run, traj in pool.imap(_one_run, jobs):
                for k, (n10, n01, n11) in enumerate(traj):
                    SI = (n10 + n01) / 2
                    n_inf = SI + n11
                    w.writerow(
                        [
                            run,
                            f"{k * dt:g}",
                            n10,
                            n01,
                            n11,
                            f"{n_inf:g}",
                            f"{SI:g}",
                            f"{n_inf / N:.6g}",
                            f"{SI / N:.6g}",
                            f"{SI / n_inf:.6g}" if n_inf > 0 else "",
                        ]
                    )
                if verbose:
                    print(
                        f"run {run + 1}/{number_of_simulations} done", file=sys.stderr
                    )


if __name__ == "__main__":
    main(verbose="--verbose" in sys.argv)
