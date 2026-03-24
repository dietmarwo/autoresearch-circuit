"""
Inner Optimizer — fcmaes parameter tuning for one topology.

For each topology T, find:
    x* = argmin_x  -phenotype_score(simulate(T, x))

Key fcmaes features exploited:
  - Low per-evaluation overhead (C++/Eigen backend)
  - Coordinated parallel retry (many restarts, keep global best)
  - Handles noisy objectives gracefully
  - Bite_cpp: fast derivative-free solver for noisy landscapes
"""

import time
import numpy as np

from fcmaes import retry
from fcmaes.optimizer import Bite_cpp

from grammar import Topology
from model_builder import build_param_bounds
from evaluator import evaluate_topology
import config as cfg


def make_objective(topology: Topology,
                   n_seeds: int = cfg.INNER_N_SEEDS,
                   t_end: float = cfg.SIM_T_END):
    """
    Create a callable objective(x) → float for fcmaes.

    The objective is NEGATED because fcmaes minimises.
    Returns PENALTY_VALUE on any failure.
    """
    def objective(x: np.ndarray) -> float:
        try:
            score = evaluate_topology(topology, x, n_seeds=n_seeds, t_end=t_end)
            return -score  # fcmaes minimises
        except Exception:
            return cfg.PENALTY_VALUE
    return objective


def optimize_topology(topology: Topology,
                      max_evals: int = cfg.INNER_MAX_EVALS,
                      n_retries: int = cfg.INNER_NUM_RETRIES,
                      n_seeds: int = cfg.INNER_N_SEEDS,
                      verbose: bool = False) -> dict:
    """
    Run fcmaes inner optimisation for a single topology.

    Uses coordinated parallel retry with Bite_cpp: multiple independent
    restarts are run and the globally best solution is kept.

    Args:
        topology:   The regulatory network topology.
        max_evals:  Maximum objective evaluations per retry.
        n_retries:  Number of independent restarts.
        n_seeds:    SSA seeds per evaluation during optimisation.
        verbose:    Print progress info.

    Returns:
        dict with keys:
          best_score   (float, positive — higher is better)
          best_params  (np.ndarray)
          num_evals    (int, total evaluations across all retries)
          topology     (Topology)
          wall_time    (float, seconds)
    """
    lower, upper = build_param_bounds(topology)
    objective = make_objective(topology, n_seeds=n_seeds)

    t0 = time.perf_counter()

    result = retry.minimize(
        objective,
        bounds=retry.Bounds(lower, upper),
        num_retries=n_retries,
        max_evaluations=max_evals,
        optimizer=Bite_cpp(max_evals),
    )

    wall = time.perf_counter() - t0

    best_score = -result.fun  # un-negate
    if verbose:
        print(f"    fcmaes: score={best_score:.4f}  "
              f"evals={max_evals * n_retries}  wall={wall:.1f}s")

    return {
        "best_score": best_score,
        "best_params": result.x.copy(),
        "num_evals": max_evals * n_retries,
        "topology": topology,
        "wall_time": wall,
    }


# ── Quick self-test ──────────────────────────────────────────

if __name__ == "__main__":
    from grammar import REPRESSILATOR, GOODWIN_LOOP, TOGGLE_SWITCH_AB

    for label, topo in [
        ("Repressilator", REPRESSILATOR),
        ("Goodwin loop",  GOODWIN_LOOP),
        ("Toggle switch", TOGGLE_SWITCH_AB),
    ]:
        print(f"\n{'─'*50}")
        print(f"Optimising {label}: {topo.to_label()}")
        print(f"  params: {topo.num_params}")
        res = optimize_topology(
            topo,
            max_evals=500,     # small budget for quick test
            n_retries=4,
            n_seeds=2,
            verbose=True,
        )
        print(f"  BEST SCORE: {res['best_score']:.4f}")
