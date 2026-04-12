"""
Inner Optimizer — fcmaes parameter tuning for one topology.

For each topology T, find:
    x* = argmin_x  -phenotype_score(simulate(T, x))

Key fcmaes features exploited:
  - Low per-evaluation overhead (C++/Eigen backend)
  - DE ask/tell with a reusable evaluator pool for larger worker counts
  - Bite_cpp fallback for smaller worker counts
  - Handles noisy objectives gracefully
"""

import time
from typing import Sequence

import numpy as np
np.set_printoptions(legacy='1.25')

import fcmaes
from fcmaes import retry
from fcmaes.optimizer import Bite_cpp

import sys
import threadpoolctl
from loguru import logger
logger.remove()
logger.add(sys.stdout, format="{time:HH:mm:ss.SS} | {process} | {level} | {message}", level="INFO")

from grammar import Topology
from model_builder import build_param_bounds
from evaluator import evaluate_topology_details
import config as cfg


def _compute_metrics(
    topology: Topology,
    x: np.ndarray,
    *,
    n_seeds: int,
    t_end: float,
    seed_offset: int,
    knockout_samples: int,
    knockdown_samples: int,
    param_perturb_samples: int,
) -> dict:
    with threadpoolctl.threadpool_limits(limits=1, user_api="blas"):
        return evaluate_topology_details(
            topology,
            x,
            n_seeds=n_seeds,
            t_end=t_end,
            seed_offset=seed_offset,
            knockout_samples=knockout_samples,
            knockdown_samples=knockdown_samples,
            param_perturb_samples=param_perturb_samples,
            batch_workers=None,
        )


def make_metric_fn(
    topology: Topology,
    *,
    metric_name: str,
    n_seeds: int = cfg.INNER_N_SEEDS,
    t_end: float = cfg.SIM_T_END,
    seed_offset: int = cfg.TRAIN_SEED_OFFSET,
    knockout_samples: int = cfg.TRAIN_KNOCKOUT_SAMPLES,
    knockdown_samples: int = cfg.TRAIN_KNOCKDOWN_SAMPLES,
    param_perturb_samples: int = cfg.TRAIN_PARAM_PERTURB_SAMPLES,
    penalty_value: float = 0.0,
):
    """Create a scalar metric callable(x) for pooled batch evaluation."""

    def metric_fn(x: np.ndarray) -> float:
        try:
            metrics = _compute_metrics(
                topology,
                x,
                n_seeds=n_seeds,
                t_end=t_end,
                seed_offset=seed_offset,
                knockout_samples=knockout_samples,
                knockdown_samples=knockdown_samples,
                param_perturb_samples=param_perturb_samples,
            )
            return float(metrics[metric_name])
        except Exception:
            return penalty_value

    return metric_fn


def make_objective(topology: Topology,
                   n_seeds: int = cfg.INNER_N_SEEDS,
                   t_end: float = cfg.SIM_T_END,
                   knockout_samples: int = cfg.TRAIN_KNOCKOUT_SAMPLES,
                   knockdown_samples: int = cfg.TRAIN_KNOCKDOWN_SAMPLES,
                   param_perturb_samples: int = cfg.TRAIN_PARAM_PERTURB_SAMPLES):
    """
    Create a callable objective(x) → float for fcmaes.

    The objective is NEGATED because fcmaes minimises.
    Returns PENALTY_VALUE on any failure.
    """
    metric_fn = make_metric_fn(
        topology,
        metric_name="raw_score",
        n_seeds=n_seeds,
        t_end=t_end,
        seed_offset=cfg.TRAIN_SEED_OFFSET,
        knockout_samples=knockout_samples,
        knockdown_samples=knockdown_samples,
        param_perturb_samples=param_perturb_samples,
        penalty_value=-cfg.PENALTY_VALUE,
    )

    def objective(x: np.ndarray) -> float:
        try:
            return -metric_fn(x)  # fcmaes minimises
        except Exception:
            return cfg.PENALTY_VALUE

    return objective

fcmaes_evaluator = None
fcmaes_evaluator_key = None
fcmaes_evaluator_workers = None


def _stop_fcmaes_evaluator() -> None:
    global fcmaes_evaluator, fcmaes_evaluator_key, fcmaes_evaluator_workers
    if fcmaes_evaluator is not None:
        fcmaes_evaluator.stop()
        fcmaes_evaluator = None
        fcmaes_evaluator_key = None
        fcmaes_evaluator_workers = None


def _ensure_fcmaes_evaluator(objective, key, n_workers: int):
    global fcmaes_evaluator, fcmaes_evaluator_key, fcmaes_evaluator_workers
    if (
        fcmaes_evaluator is not None
        and (fcmaes_evaluator_key != key or fcmaes_evaluator_workers != n_workers)
    ):
        _stop_fcmaes_evaluator()
    if fcmaes_evaluator is None:
        fcmaes_evaluator = fcmaes.evaluator.parallel(objective, workers=n_workers)
        fcmaes_evaluator_key = key
        fcmaes_evaluator_workers = n_workers
    return fcmaes_evaluator


def evaluate_params_batch(
    topology: Topology,
    xs: Sequence[np.ndarray],
    *,
    metric_name: str,
    n_workers: int,
    n_seeds: int,
    t_end: float,
    seed_offset: int,
    knockout_samples: int = 0,
    knockdown_samples: int = 0,
    param_perturb_samples: int = 0,
) -> np.ndarray:
    """
    Evaluate many parameter vectors for one topology using the reusable fcmaes pool.

    This is primarily used for post-optimisation stress tests such as local
    parameter perturbations. The worker pool is recreated only when the
    topology, worker count, or evaluation settings change.
    """
    if len(xs) == 0:
        return np.empty(0, dtype=np.float64)

    if n_workers < 2:
        metric_fn = make_metric_fn(
            topology,
            metric_name=metric_name,
            n_seeds=n_seeds,
            t_end=t_end,
            seed_offset=seed_offset,
            knockout_samples=knockout_samples,
            knockdown_samples=knockdown_samples,
            param_perturb_samples=param_perturb_samples,
            penalty_value=0.0,
        )
        return np.asarray([metric_fn(np.asarray(x, dtype=np.float64)) for x in xs], dtype=np.float64)

    metric_fn = make_metric_fn(
        topology,
        metric_name=metric_name,
        n_seeds=n_seeds,
        t_end=t_end,
        seed_offset=seed_offset,
        knockout_samples=knockout_samples,
        knockdown_samples=knockdown_samples,
        param_perturb_samples=param_perturb_samples,
        penalty_value=0.0,
    )
    key = (
        "batch",
        topology.edges,
        metric_name,
        int(n_workers),
        int(n_seeds),
        float(t_end),
        int(seed_offset),
        int(knockout_samples),
        int(knockdown_samples),
        int(param_perturb_samples),
    )
    evaluator = _ensure_fcmaes_evaluator(metric_fn, key, n_workers)
    return evaluator(np.asarray(xs, dtype=np.float64))

def shutdown():
    _stop_fcmaes_evaluator()

def optimize_topology(topology: Topology,
                      max_evals: int = cfg.INNER_MAX_EVALS,
                      n_workers: int = cfg.INNER_NUM_WORKERS,
                      n_seeds: int = cfg.INNER_N_SEEDS,
                      verbose: bool = False) -> dict:
    global fcmaes_evaluator
    """create_doc
    Run fcmaes inner optimisation for a single topology.

    Uses DE with a reusable worker pool when enough workers are available.
    For smaller worker counts or very small budgets, falls back to coordinated
    Bite_cpp retry.

    Args:
        topology:   The regulatory network topology.
        max_evals:  Total inner objective budget per topology.
        n_workers:  Number of parallel workers.
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
          num_evals    (int, total evaluations across all workers)
          topology     (Topology)
          wall_time    (float, seconds)create_doc
    """
    lower, upper = build_param_bounds(topology)
    objective = make_objective(
        topology,
        n_seeds=n_seeds,
        knockout_samples=cfg.TRAIN_KNOCKOUT_SAMPLES,
        knockdown_samples=cfg.TRAIN_KNOCKDOWN_SAMPLES,
        param_perturb_samples=cfg.TRAIN_PARAM_PERTURB_SAMPLES,
    )

    t0 = time.perf_counter()
    if n_workers >= 8 and max_evals >= n_workers:
        key = (
            "opt",
            topology.edges,
            int(n_workers),
            int(n_seeds),
            float(cfg.SIM_T_END),
            int(cfg.TRAIN_SEED_OFFSET),
            int(cfg.TRAIN_KNOCKOUT_SAMPLES),
            int(cfg.TRAIN_KNOCKDOWN_SAMPLES),
            int(cfg.TRAIN_PARAM_PERTURB_SAMPLES),
        )
        evaluator = _ensure_fcmaes_evaluator(objective, key, n_workers)
        es = fcmaes.de.DE(len(lower), retry.Bounds(lower, upper), keep=20, popsize=n_workers)
        # alternatively, since fcmaes 2.0.2 you can apply Biteopt with ask / tell interface 
        # es = fcmaes.bitecpp.Bite_C(
        #    len(lower),
        #    retry.Bounds(lower, upper),
        #    batch_size=n_workers,
        #    max_evaluations=max_evals,
        # )       
        eval_batches = 0
        for _ in range(max(1, max_evals // n_workers)):
            xs = es.ask()
            ys = evaluator(xs)
            eval_batches += 1
            stop = es.tell(ys)
            if stop:
                break
        result = es.result()
        num_evals = eval_batches * n_workers
    else:
        per_worker_evals = max(1, max_evals // n_workers)
        result = retry.minimize(
            objective,
            bounds=retry.Bounds(lower, upper),
            num_retries=n_workers,
            max_evaluations=per_worker_evals,
            optimizer=Bite_cpp(per_worker_evals),
        )
        num_evals = int(getattr(result, "nfev", per_worker_evals * n_workers))

    wall = time.perf_counter() - t0

    best_params = result.x.copy()
    best_raw_score = -result.fun  # un-negate raw objective
    train_metrics = evaluate_topology_details(
        topology,
        best_params,
        n_seeds=n_seeds,
        t_end=cfg.SIM_T_END,
        seed_offset=cfg.TRAIN_SEED_OFFSET,
        knockout_samples=cfg.TRAIN_KNOCKOUT_SAMPLES,
        knockdown_samples=cfg.TRAIN_KNOCKDOWN_SAMPLES,
        param_perturb_samples=cfg.TRAIN_PARAM_PERTURB_SAMPLES,
        batch_workers=n_workers,
    )
    validation_metrics = evaluate_topology_details(
        topology,
        best_params,
        n_seeds=cfg.VALID_N_SEEDS,
        t_end=cfg.VALID_T_END,
        seed_offset=cfg.VALID_SEED_OFFSET,
        knockout_samples=cfg.VALID_KNOCKOUT_SAMPLES,
        knockdown_samples=cfg.VALID_KNOCKDOWN_SAMPLES,
        param_perturb_samples=cfg.VALID_PARAM_PERTURB_SAMPLES,
        batch_workers=n_workers,
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
        extra = ""
        if validation_metrics["knockout_score"] is not None:
            extra = (
                f"  full={validation_metrics['full_score']:.4f}"
                f"  ko={validation_metrics['knockout_score']:.4f}"
                f"  pass={validation_metrics['knockout_pass_rate']:.2f}"
            )
        if validation_metrics["knockdown_score"] is not None:
            extra += f"  kd={validation_metrics['knockdown_score']:.4f}"
        if validation_metrics["param_perturb_score"] is not None:
            extra += f"  pert={validation_metrics['param_perturb_score']:.4f}"
        print(
            f"    fcmaes: raw={best_raw_score:.4f}  "
            f"train={train_metrics['score']:.4f}  "
            f"val={validation_metrics['score']:.4f}  "
            f"gap={generalization_gap:.4f}  "
            f"rank={best_score:.4f}  "
            f"period={period_str}  "
            f"{extra}  "
            f"evals={num_evals}  wall={wall:.1f}s"
        )

    return {
        "best_score": best_score,
        "best_params": best_params,
        "best_raw_score": best_raw_score,
        "train_score": train_metrics["score"],
        "train_raw_score": train_metrics["raw_score"],
        "train_period": train_metrics["period"],
        "train_full_score": train_metrics["full_score"],
        "train_knockout_score": train_metrics["knockout_score"],
        "train_knockout_pass_rate": train_metrics["knockout_pass_rate"],
        "train_knockdown_score": train_metrics["knockdown_score"],
        "train_param_perturb_score": train_metrics["param_perturb_score"],
        "validation_score": validation_metrics["score"],
        "validation_raw_score": validation_metrics["raw_score"],
        "validation_period": validation_metrics["period"],
        "validation_full_score": validation_metrics["full_score"],
        "validation_knockout_score": validation_metrics["knockout_score"],
        "validation_knockout_pass_rate": validation_metrics["knockout_pass_rate"],
        "validation_knockdown_score": validation_metrics["knockdown_score"],
        "validation_param_perturb_score": validation_metrics["param_perturb_score"],
        "generalization_gap": generalization_gap,
        "num_evals": num_evals,
        "topology": topology,
        "wall_time": wall,
    }


# ── Quick self-test ──────────────────────────────────────────

if __name__ == "__main__":
    from grammar import REPRESSILATOR, GOODWIN_LOOP, TOGGLE_SWITCH_AB

    try:
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
                max_evals=320,     # small budget for quick test
                n_workers=16,
                n_seeds=2,
                verbose=True,
            )
            print(
                f"  BEST RANK: {res['best_score']:.4f}  "
                f"(train={res['train_score']:.4f}  val={res['validation_score']:.4f})"
            )
    finally:
        shutdown()
