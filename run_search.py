#!/usr/bin/env python3
"""
Main entry point — run the split-brain circuit topology search.

Usage:
    python run_search.py --strategy random --n 30
    python run_search.py --strategy evo --n 50
    python run_search.py --strategy agentic --n 20
    python run_search.py --strategy agentic --model claude-sonnet-4-20250514
    python run_search.py --strategy agentic --model gemini-2.5-pro --llm-backend gemini
    python run_search.py --strategy agentic --model MiniMax-M2.7 --llm-backend minimax
    python run_search.py --strategy agentic --base-url http://127.0.0.1:8011/v1 --model qwen3-coder

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

from outer_loop import run_random_search, run_evolutionary_search
from agentic_loop import run_agentic_search, make_llm_call_fn
import config as cfg


def main():
    parser = argparse.ArgumentParser(
        description="Split-brain circuit topology search with fcmaes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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
        help="fcmaes evaluation budget per topology",
    )
    parser.add_argument(
        "--retries", type=int, default=cfg.INNER_NUM_RETRIES,
        help="fcmaes parallel retries per topology",
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
    print(f"  Strategy:    {args.strategy}")
    print(f"  Candidates:  {args.n}")
    print(f"  Inner evals: {args.inner_evals} × {args.retries} retries")
    print(f"  Train eval:  t_end={cfg.SIM_T_END:g}  seeds={cfg.INNER_N_SEEDS}")
    print(f"  Valid eval:  t_end={cfg.VALID_T_END:g}  seeds={cfg.VALID_N_SEEDS}")
    print(f"  Output:      {out_dir}")
    if args.strategy == "agentic":
        print(f"  LLM backend: {args.llm_backend}")
        print(f"  LLM model:   {args.model}")
        print(f"  Base URL:    {args.base_url or '(provider default)'}")
        print(f"  Max tokens:  {args.max_tokens}")
        print(f"  Thinking:    {args.thinking_effort}")
    print(f"{'='*60}\n")

    t0 = time.perf_counter()

    # ── Run search ────────────────────────────────────────
    if args.strategy == "random":
        archive = run_random_search(
            n_candidates=args.n,
            max_evals_inner=args.inner_evals,
            n_retries=args.retries,
            seed=args.seed,
        )

    elif args.strategy == "evo":
        archive = run_evolutionary_search(
            n_iterations=args.n,
            max_evals_inner=args.inner_evals,
            n_retries=args.retries,
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
            n_retries=args.retries,
        )

    total_wall = time.perf_counter() - t0

    # ── Save results ──────────────────────────────────────
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(archive.summary())
    print(f"\nTotal wall time: {total_wall:.1f}s")

    archive.save_pickle(str(out_dir / "archive.pkl"))
    archive.save_json(str(out_dir / "archive.json"))
    print(f"\nArchive saved to {out_dir}/")

    # ── Generate plots ────────────────────────────────────
    if args.plot:
        from viz import generate_all_plots
        generate_all_plots(archive, output_dir=str(out_dir))

    print("\nDone.")


if __name__ == "__main__":
    main()
