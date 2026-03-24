"""
Outer Loop — Topology Search Strategies

Strategy 1: Random sampling (baseline)
Strategy 2: Evolutionary (1+1)-ES mutation search

Both use the same inner-optimizer interface:
  optimize_topology(topology) → {best_score, best_params, ...}
"""

import time
import numpy as np

from grammar import Topology, enumerate_valid_topologies, mutate_topology
from inner_optimizer import optimize_topology
from archive import Archive, SearchResult
import config as cfg


# ──────────────────────────────────────────────────────────
# Strategy 1: Random Search
# ──────────────────────────────────────────────────────────

def run_random_search(
    n_candidates: int = cfg.RANDOM_SEARCH_N,
    max_evals_inner: int = cfg.INNER_MAX_EVALS,
    n_retries: int = cfg.INNER_NUM_RETRIES,
    seed: int = 0,
) -> Archive:
    """
    Evaluate n randomly-chosen valid topologies.

    This is the simplest possible baseline: no learning, pure sampling.
    Its purpose is to establish the expected score distribution
    before more sophisticated strategies are tried.
    """
    rng = np.random.default_rng(seed)
    all_valid = enumerate_valid_topologies()
    rng.shuffle(all_valid)
    candidates = all_valid[:n_candidates]

    archive = Archive()
    total_t0 = time.perf_counter()

    for i, topo in enumerate(candidates):
        print(f"[Random {i+1}/{n_candidates}] {topo.to_label()}")
        result = optimize_topology(
            topo, max_evals=max_evals_inner, n_retries=n_retries,
            verbose=True,
        )
        archive.add(SearchResult(
            topology=topo,
            score=result["best_score"],
            params=result["best_params"],
            iteration=i,
            wall_time=result["wall_time"],
            strategy="random",
        ))
        print(f"  → score = {result['best_score']:.4f}")
        if (i + 1) % 10 == 0:
            print(f"\n{archive.summary()}\n")

    total_wall = time.perf_counter() - total_t0
    print(f"\nRandom search done in {total_wall:.1f}s")
    print(archive.summary())
    return archive


# ──────────────────────────────────────────────────────────
# Strategy 2: Evolutionary (1+1)-ES over topology encodings
# ──────────────────────────────────────────────────────────

def run_evolutionary_search(
    n_iterations: int = cfg.EVO_SEARCH_N,
    max_evals_inner: int = cfg.INNER_MAX_EVALS,
    n_retries: int = cfg.INNER_NUM_RETRIES,
    seed: int = 0,
) -> Archive:
    """
    Simple (1+1)-ES over topologies:
      1. Start from a random valid topology.
      2. Each iteration: mutate one edge, optimise, keep if score improves.
      3. All evaluations go into the global archive.

    This is the first strategy that actually *searches*, as opposed
    to random sampling.  It should converge faster when the topology
    landscape has local structure (which gene-circuit spaces do).
    """
    rng = np.random.default_rng(seed)
    all_valid = enumerate_valid_topologies()

    # Random start
    current = all_valid[rng.integers(len(all_valid))]
    result = optimize_topology(
        current, max_evals=max_evals_inner, n_retries=n_retries, verbose=True,
    )
    current_score = result["best_score"]

    archive = Archive()
    archive.add(SearchResult(
        current, current_score, result["best_params"], 0,
        result["wall_time"], "evo",
    ))
    print(f"[Evo 0/{n_iterations}] Start: {current.to_label()}  "
          f"score={current_score:.4f}")

    total_t0 = time.perf_counter()

    for i in range(1, n_iterations):
        # Mutate until valid (up to MAX_MUTATION_TRIES attempts)
        candidate = None
        for _ in range(cfg.MAX_MUTATION_TRIES):
            c = mutate_topology(current, rng)
            if c.is_valid() and not archive.already_evaluated(c):
                candidate = c
                break
        if candidate is None:
            # Fall back to a random valid topology not yet seen
            unseen = [t for t in all_valid if not archive.already_evaluated(t)]
            if not unseen:
                print("  All topologies evaluated. Stopping early.")
                break
            candidate = unseen[rng.integers(len(unseen))]

        print(f"[Evo {i}/{n_iterations}] {candidate.to_label()}")
        result = optimize_topology(
            candidate, max_evals=max_evals_inner, n_retries=n_retries,
            verbose=True,
        )
        score = result["best_score"]
        archive.add(SearchResult(
            candidate, score, result["best_params"], i,
            result["wall_time"], "evo",
        ))
        print(f"  → score = {score:.4f}  (current best = {current_score:.4f})")

        if score > current_score:
            current = candidate
            current_score = score
            print(f"  ★ New best topology!")

        if (i + 1) % 10 == 0:
            print(f"\n{archive.summary()}\n")

    total_wall = time.perf_counter() - total_t0
    print(f"\nEvolutionary search done in {total_wall:.1f}s")
    print(archive.summary())
    return archive


# ──────────────────────────────────────────────────────────
# Quick test
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Running a tiny random search (3 topologies, small budget)...")
    archive = run_random_search(
        n_candidates=3,
        max_evals_inner=300,
        n_retries=2,
    )
    print("\nDone.")
    print(archive.summary())
