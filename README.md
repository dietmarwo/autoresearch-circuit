# Circuit Search — Split-Brain Automated Design of Stochastic Biochemical Circuits

A split-brain automated design loop for stochastic biochemical circuits,
showcasing `fcmaes` as the inner optimization engine.

```
Outer loop proposes topology → fcmaes optimizes parameters → GillesPy2 evaluates phenotype
```

## Architecture

```
┌───────────────────────────────────┐
│       OUTER LOOP (Agentic)        │   Proposes circuit topology
│  random / evolutionary / LLM      │   from bounded 3-gene grammar
└──────────────┬────────────────────┘
               │ topology T
               ▼
┌───────────────────────────────────┐
│       MODEL BUILDER               │   topology + params → GillesPy2 model
│  Hill-function propensities       │   Adapts parameter vector to edges
└──────────────┬────────────────────┘
               │
               ▼
┌───────────────────────────────────┐
│       INNER LOOP (fcmaes)         │   Optimizes continuous kinetic params
│  Bite_cpp with parallel retry     │   Handles noisy stochastic objectives
└──────────────┬────────────────────┘
               │ best params x*
               ▼
┌───────────────────────────────────┐
│       PHENOTYPE EVALUATOR         │   SSA simulation → oscillation score
│  Detrending, peak/trough analysis │   Robust across multiple seeds
└───────────────────────────────────┘
```

## Quick Start

```bash
pip install fcmaes gillespy2 numpy scipy matplotlib

# Random search (baseline) — 30 topologies
python run_search.py --strategy random --n 30

# Evolutionary (1+1)-ES — 50 iterations
python run_search.py --strategy evo --n 50

# LLM-guided agentic search (requires ANTHROPIC_API_KEY)
python run_search.py --strategy agentic --n 20

# Quick test (small budget, fast)
python run_search.py --strategy random --n 5 --inner-evals 200 --retries 2
```

## File Structure

| File | Lines | Purpose |
|------|-------|---------|
| `config.py` | 81 | All hyperparameters in one place |
| `grammar.py` | 160 | 3-gene topology grammar, encoding, mutation, canonical motifs |
| `model_builder.py` | 184 | Topology + params → GillesPy2 model with Hill-function propensities |
| `evaluator.py` | 228 | Oscillation quality scoring: detrending, peak/trough analysis, multi-seed |
| `inner_optimizer.py` | 119 | fcmaes wrapper: Bite_cpp with coordinated parallel retry |
| `outer_loop.py` | 163 | Random search + evolutionary (1+1)-ES strategies |
| `agentic_loop.py` | 266 | LLM-guided topology proposal with structured feedback |
| `archive.py` | 121 | Results storage, ranking, JSON/pickle serialisation |
| `viz.py` | 237 | Trace plots, network diagrams, score progression charts |
| `run_search.py` | 146 | CLI entry point |

## Topology Grammar

- **3 genes** (A, B, C), each with production + degradation
- **9 edge slots**: 3 self-regulation + 6 cross-regulation
- Each edge: absent (0) / activation (1) / inhibition (2)
- Constraints: 2–6 active edges, no isolated nodes
- **12,024 valid topologies** in the grammar

## Evaluator Design

The oscillation scorer avoids common false positives:

1. **Linear detrending** — rejects monotonic growth/decay
2. **Prominence-based peak detection** — rejects stochastic noise bumps
3. **Trough depth validation** — requires real valleys between peaks
4. **Amplitude-to-mean ratio** — rejects weak fluctuations on high baselines
5. **Multi-seed median** — resists stochastic outliers

## Known Canonical Motifs

| Motif | Edges | Expected |
|-------|-------|----------|
| Repressilator | A⊣B, B⊣C, C⊣A | Strong oscillator |
| Goodwin loop | A→B, B→C, C⊣A | Delayed negative feedback |
| Toggle switch | A⊣B, B⊣A | Bistable (not oscillatory) |

## Dependencies

- `fcmaes` — fast gradient-free optimization (C++/Eigen backend)
- `gillespy2` — stochastic simulation (SSA)
- `numpy`, `scipy` — numerics, peak detection
- `matplotlib` — plotting
- `anthropic` — (optional) for agentic loop

## Context

This project extends `fast-cma-es/examples/vilar.py` from parameter optimization
of one fixed stochastic reaction network to **outer-loop structural search** over
a bounded space of reaction-network topologies. It mirrors the split-brain
architecture of `autoresearch-trading`: the outer loop proposes structure,
`fcmaes` optimizes numbers.
