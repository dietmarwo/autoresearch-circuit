#!/usr/bin/env python3
"""
Main entry point — run the split-brain circuit topology search.

Usage:
    python run_search.py --experiment oscillator3 --strategy random --n 30
    python run_search.py --experiment oscillator3 --strategy evo --n 50
    python run_search.py --experiment oscillator3 --strategy agentic --n 20
    python run_search.py --experiment oscillator3 --strategy agentic --agentic-mode blind --n 20
    python run_search.py --experiment robust5 --strategy random --n 12
    python run_search.py --experiment robust5 --strategy agentic --agentic-mode blind --n 12
    python run_search.py --experiment robust5 --strategy agentic --model MiniMax-M2.7 --llm-backend minimax

The search evaluates candidate circuit topologies by:
  1. Outer loop proposes a topology (grammar-bounded)
  2. fcmaes optimises continuous kinetic parameters
  3. GillesPy2 SSA simulation scores oscillation quality
  4. Results are archived and fed back to the outer loop

Strategies:
  random  — uniform sampling from the valid topology space (baseline)
  evo     — (1+1)-ES mutation search over topology encodings
  agentic — LLM-guided proposal via selectable backend/model options
"""

import argparse
import sys
import time
from pathlib import Path

import config as cfg


def _preparse_experiment(argv: list[str]) -> str:
    """Resolve the experiment before importing experiment-dependent modules."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--experiment",
        choices=cfg.AVAILABLE_EXPERIMENTS,
        default=cfg.DEFAULT_EXPERIMENT,
    )
    args, _ = parser.parse_known_args(argv)
    cfg.set_experiment(args.experiment)
    return args.experiment


_preparse_experiment(sys.argv[1:])

from outer_loop import run_random_search, run_evolutionary_search
from agentic_loop import run_agentic_search, make_llm_call_fn
from inner_optimizer import shutdown as shutdown_inner_optimizer


def main():
    parser = argparse.ArgumentParser(
        description="Split-brain circuit topology search with fcmaes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--experiment",
        choices=cfg.AVAILABLE_EXPERIMENTS,
        default=cfg.EXPERIMENT,
        help="Experiment preset controlling topology size and phenotype objective",
    )
    parser.add_argument(
        "--strategy", choices=["random", "evo", "agentic"],
        default="random",
        help="Outer-loop search strategy",
    )
    parser.add_argument(
        "--n", type=int, default=None,
        help="Number of topologies to evaluate (default: per-strategy config)",
    )
    parser.add_argument(
        "--inner-evals", type=int, default=cfg.INNER_MAX_EVALS,
        help="Total inner objective budget per topology",
    )
    parser.add_argument(
        "--workers", type=int, default=cfg.INNER_NUM_WORKERS,
        help="fcmaes parallel workers per topology",
    )
    parser.add_argument(
        "--output", type=str, default="results",
        help="Output directory for results and plots",
    )
    parser.add_argument(
        "--plot", action="store_true", default=True,
        help="Generate plots after search",
    )
    parser.add_argument(
        "--no-plot", dest="plot", action="store_false",
        help="Skip plot generation",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--llm-backend",
        choices=["auto", "openai", "claude", "gemini", "minimax"],
        default=cfg.LLM_BACKEND,
        help="LLM backend for agentic search",
    )
    parser.add_argument(
        "--model", type=str, default=cfg.LLM_MODEL,
        help="Model id for agentic search",
    )
    parser.add_argument(
        "--base-url", type=str, default=cfg.LLM_BASE_URL,
        help="OpenAI-compatible API base URL; ignored by native Claude/Gemini/MiniMax backends",
    )
    parser.add_argument(
        "--temperature", type=float, default=cfg.LLM_TEMPERATURE,
        help="LLM temperature for agentic search",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=cfg.LLM_MAX_TOKENS,
        help="Max output tokens for agentic LLM calls",
    )
    parser.add_argument(
        "--thinking-effort", choices=["none", "high"],
        default=cfg.LLM_THINKING_EFFORT,
        help="Optional native reasoning effort for Claude/Gemini/MiniMax backends",
    )
    parser.add_argument(
        "--agentic-mode", choices=["blind", "guided"],
        default=cfg.AGENTIC_MODE,
        help="Blind benchmark mode or guided application mode for the agentic loop",
    )
    parser.add_argument(
        "--bootstrap-iters", type=int, default=cfg.AGENTIC_BOOTSTRAP_ITERS,
        help="Initial no-score bootstrap proposals before showing the current best topology",
    )
    parser.add_argument(
        "--explore-min-hamming", type=int, default=cfg.AGENTIC_EXPLORE_MIN_HAMMING,
        help="Minimum Hamming distance enforced during bootstrap/exploration phases",
    )
    args = parser.parse_args()

    # Resolve defaults
    if args.n is None:
        defaults = {"random": cfg.RANDOM_SEARCH_N,
                     "evo": cfg.EVO_SEARCH_N,
                     "agentic": cfg.AGENTIC_SEARCH_N}
        args.n = defaults[args.strategy]

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f"Circuit Topology Search")
    print(f"  Experiment:  {args.experiment}")
    print(f"  Genes:       {cfg.NUM_GENES} ({', '.join(cfg.GENES)})")
    print(f"  Edge slots:  {cfg.NUM_EDGE_SLOTS}")
    print(f"  Strategy:    {args.strategy}")
    print(f"  Candidates:  {args.n}")
    print(f"  Inner evals: {args.inner_evals} total  ({args.workers} workers)")
    print(f"  Train eval:  t_end={cfg.SIM_T_END:g}  seeds={cfg.INNER_N_SEEDS}")
    print(f"  Valid eval:  t_end={cfg.VALID_T_END:g}  seeds={cfg.VALID_N_SEEDS}")
    if cfg.VALID_KNOCKOUT_SAMPLES != 0:
        knockout_desc = "all" if cfg.VALID_KNOCKOUT_SAMPLES < 0 else str(cfg.VALID_KNOCKOUT_SAMPLES)
        print(f"  Knockouts:   train={cfg.TRAIN_KNOCKOUT_SAMPLES}  valid={knockout_desc}")
    if cfg.VALID_KNOCKDOWN_SAMPLES != 0:
        knockdown_desc = "all" if cfg.VALID_KNOCKDOWN_SAMPLES < 0 else str(cfg.VALID_KNOCKDOWN_SAMPLES)
        print(
            f"  Knockdowns:  train={cfg.TRAIN_KNOCKDOWN_SAMPLES}  "
            f"valid={knockdown_desc}  remain={cfg.KNOCKDOWN_FACTOR:.2f}"
        )
    if cfg.VALID_PARAM_PERTURB_SAMPLES > 0:
        print(
            f"  Param jitter: train={cfg.TRAIN_PARAM_PERTURB_SAMPLES}  "
            f"valid={cfg.VALID_PARAM_PERTURB_SAMPLES}  sigma={cfg.PARAM_PERTURB_SIGMA:.2f}"
        )
    if (
        cfg.VALID_KNOCKOUT_SAMPLES != 0
        or cfg.VALID_KNOCKDOWN_SAMPLES != 0
        or cfg.VALID_PARAM_PERTURB_SAMPLES > 0
    ):
        print(
            f"  Stress metric: q{int(round(100 * cfg.ROBUST_SCENARIO_AGGREGATION_QUANTILE)):02d} "
            f"across scenarios  pass>={cfg.ROBUST_SUCCESS_THRESHOLD:.2f}"
        )
    print(f"  Output:      {out_dir}")
    if args.strategy == "agentic":
        print(f"  Agentic mode: {args.agentic_mode}")
        print(f"  Bootstrap:    {args.bootstrap_iters}")
        print(f"  Explore minΔ: {args.explore_min_hamming}")
        print(f"  LLM backend: {args.llm_backend}")
        print(f"  LLM model:   {args.model}")
        print(f"  Base URL:    {args.base_url or '(provider default)'}")
        print(f"  Max tokens:  {args.max_tokens}")
        print(f"  Thinking:    {args.thinking_effort}")
    print(f"{'='*60}\n")

    t0 = time.perf_counter()

    try:
        # ── Run search ────────────────────────────────────
        if args.strategy == "random":
            archive = run_random_search(
                n_candidates=args.n,
                max_evals_inner=args.inner_evals,
                n_workers=args.workers,
                seed=args.seed,
            )

        elif args.strategy == "evo":
            archive = run_evolutionary_search(
                n_iterations=args.n,
                max_evals_inner=args.inner_evals,
                n_workers=args.workers,
                seed=args.seed,
            )

        elif args.strategy == "agentic":
            try:
                llm_fn, resolved_backend, resolved_model = make_llm_call_fn(
                    backend=args.llm_backend,
                    base_url=args.base_url,
                    model=args.model,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    thinking_effort=args.thinking_effort,
                )
            except Exception as e:
                print(f"ERROR: Could not initialise agentic LLM backend: {e}")
                print("Check your API key env vars and backend/model selection.")
                sys.exit(1)

            print(f"Resolved LLM backend: {resolved_backend}")
            print(f"Resolved model:       {resolved_model}\n")

            archive = run_agentic_search(
                llm_fn,
                n_iterations=args.n,
                max_evals_inner=args.inner_evals,
                n_workers=args.workers,
                agentic_mode=args.agentic_mode,
                bootstrap_iters=args.bootstrap_iters,
                explore_min_hamming=args.explore_min_hamming,
            )

        total_wall = time.perf_counter() - t0

        # ── Save results ──────────────────────────────────
        print(f"\n{'='*60}")
        print("FINAL RESULTS")
        print(f"{'='*60}")
        print(archive.summary())
        print(f"\nTotal wall time: {total_wall:.1f}s")

        archive.save_pickle(str(out_dir / "archive.pkl"))
        archive.save_json(str(out_dir / "archive.json"))
        print(f"\nArchive saved to {out_dir}/")

        # ── Generate plots ────────────────────────────────
        if args.plot:
            from viz import generate_all_plots
            generate_all_plots(archive, output_dir=str(out_dir))

        print("\nDone.")
    finally:
        shutdown_inner_optimizer()


if __name__ == "__main__":
    main()
