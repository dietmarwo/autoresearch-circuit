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
np.set_printoptions(legacy='1.25')

from fcmaes import retry
from fcmaes.optimizer import Bite_cpp

import sys
from loguru import logger
logger.remove()
logger.add(sys.stdout, format="{time:HH:mm:ss.SS} | {process} | {level} | {message}", level="INFO")

from grammar import Topology
from model_builder import build_param_bounds
from evaluator import evaluate_topology_details
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
            metrics = evaluate_topology_details(
                topology,
                x,
                n_seeds=n_seeds,
                t_end=t_end,
                seed_offset=cfg.TRAIN_SEED_OFFSET,
            )
            return -metrics["raw_score"]  # fcmaes minimises
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
          best_score   (float, validation-based ranking score)
          best_params  (np.ndarray)
          best_raw_score (float, raw training score optimised by fcmaes)
          train_score  (float, reported score on optimisation seeds)
          validation_score (float, reported holdout score on longer simulations)
          generalization_gap (float, abs(train_score - validation_score))
          train_period / validation_period (float, median oscillation period)
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

    best_params = result.x.copy()
    best_raw_score = -result.fun  # un-negate raw objective
    train_metrics = evaluate_topology_details(
        topology,
        best_params,
        n_seeds=n_seeds,
        t_end=cfg.SIM_T_END,
        seed_offset=cfg.TRAIN_SEED_OFFSET,
    )
    validation_metrics = evaluate_topology_details(
        topology,
        best_params,
        n_seeds=cfg.VALID_N_SEEDS,
        t_end=cfg.VALID_T_END,
        seed_offset=cfg.VALID_SEED_OFFSET,
    )
    generalization_gap = abs(train_metrics["score"] - validation_metrics["score"])
    best_score = (
        validation_metrics["score"]
        - cfg.GENERALIZATION_GAP_PENALTY * generalization_gap
    )

    if verbose:
        period_str = (
            f"{validation_metrics['period']:.1f}"
            if validation_metrics["period"] > 0.0 else "n/a"
        )
        print(
            f"    fcmaes: raw={best_raw_score:.4f}  "
            f"train={train_metrics['score']:.4f}  "
            f"val={validation_metrics['score']:.4f}  "
            f"gap={generalization_gap:.4f}  "
            f"rank={best_score:.4f}  "
            f"period={period_str}  "
            f"evals={max_evals * n_retries}  wall={wall:.1f}s"
        )

    return {
        "best_score": best_score,
        "best_params": best_params,
        "best_raw_score": best_raw_score,
        "train_score": train_metrics["score"],
        "train_raw_score": train_metrics["raw_score"],
        "train_period": train_metrics["period"],
        "validation_score": validation_metrics["score"],
        "validation_raw_score": validation_metrics["raw_score"],
        "validation_period": validation_metrics["period"],
        "generalization_gap": generalization_gap,
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
        print(
            f"  BEST RANK: {res['best_score']:.4f}  "
            f"(train={res['train_score']:.4f}  val={res['validation_score']:.4f})"
        )
