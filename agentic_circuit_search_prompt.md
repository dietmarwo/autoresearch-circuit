# Implementation Prompt: Agentic Split-Brain Circuit Search with fcmaes

## Role & Mission

You are an expert Python developer and scientific computing engineer. Your task is to build a **split-brain automated design loop** for stochastic biochemical circuits:

- An **outer agentic loop** searches over circuit **topologies** (discrete structure).
- An **inner loop** powered by **fcmaes** optimizes **continuous kinetic parameters** for each candidate topology.
- **GillesPy2** evaluates stochastic phenotype quality via SSA simulation.

The deliverable is a self-contained Python project that demonstrates `fcmaes` as the essential inner optimization engine for automated scientific discovery — mirroring the architecture of `autoresearch-trading` but applied to gene-circuit design.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                   OUTER LOOP (Agentic)                  │
│                                                         │
│  Proposes topology T from a bounded grammar             │
│  Receives score(T) = best phenotype after optimization  │
│  Maintains archive of (topology, score, params, trace)  │
│  Decides next topology to explore                       │
│                                                         │
│  Strategies (implement in order):                       │
│    1. Random sampling baseline                          │
│    2. Evolutionary mutation over topology encodings      │
│    3. LLM-guided proposal (agentic, last)               │
└────────────────────┬────────────────────────────────────┘
                     │ topology T
                     ▼
┌─────────────────────────────────────────────────────────┐
│              MODEL BUILDER                              │
│                                                         │
│  topology_to_model(T, params) → GillesPy2 Model        │
│  Maps edges (activation/inhibition) to reactions        │
│  Adapts parameter vector length to topology              │
│  Validates model before simulation                      │
└────────────────────┬────────────────────────────────────┘
                     │ model + param bounds
                     ▼
┌─────────────────────────────────────────────────────────┐
│              INNER LOOP (fcmaes)                        │
│                                                         │
│  Optimizes continuous params x for topology T           │
│  objective(x) = -phenotype_score(simulate(T, x))       │
│  Uses Bite_cpp or Cma_cpp with parallel retry           │
│  Returns best_x, best_score                             │
└────────────────────┬────────────────────────────────────┘
                     │ best params x*
                     ▼
┌─────────────────────────────────────────────────────────┐
│              PHENOTYPE EVALUATOR                        │
│                                                         │
│  Runs SSA simulation(s) with GillesPy2                  │
│  Computes oscillation quality metrics:                  │
│    - peak regularity (spacing CV)                       │
│    - amplitude consistency                              │
│    - persistence over time window                       │
│    - robustness across N stochastic seeds               │
│  Returns scalar score ∈ [0, 1]                          │
│  Penalties for: no oscillation, flat trace, crash       │
└─────────────────────────────────────────────────────────┘
```

---

## Step-by-Step Implementation Plan

### Step 1 — Project Scaffold

Create this file structure:

```
circuit_search/
├── grammar.py          # Topology grammar and encoding
├── model_builder.py    # Topology → GillesPy2 model
├── evaluator.py        # Phenotype scoring (oscillation metrics)
├── inner_optimizer.py  # fcmaes parameter optimization wrapper
├── outer_loop.py       # Outer search strategies
├── agentic_loop.py     # LLM-guided outer loop (Phase 2)
├── archive.py          # Results storage and analysis
├── run_search.py       # Main entry point
├── config.py           # All hyperparameters in one place
└── viz.py              # Plotting best topologies and traces
```

### Step 2 — Topology Grammar (`grammar.py`)

Define a bounded, explicit grammar for 3-node gene regulatory networks.

```python
"""
Topology Grammar for 3-Node Gene Regulatory Networks

Components: 3 transcription factors (genes) named A, B, C.
Each gene has:
  - a constitutive production reaction (mRNA/protein simplified to one species)
  - a degradation reaction

Pairwise regulatory edges:
  - For each ordered pair (X, Y) where X != Y, the edge is one of:
      0 = absent
      1 = activation  (X promotes production of Y)
      2 = inhibition   (X represses production of Y)

Self-regulation:
  - For each gene X, self-edge is one of: 0, 1, 2

Encoding:
  A topology is a tuple of 9 integers (3 self-edges + 6 cross-edges):
    (A→A, B→B, C→C, A→B, A→C, B→A, B→C, C→A, C→B)
  Each value ∈ {0, 1, 2}

Total raw topology space: 3^9 = 19683
After filtering (require ≥ 2 edges, ≤ 6 edges, no isolated nodes): ~2000-5000

Continuous parameters per topology (adapted to edge count):
  - Per gene: basal_production_rate, degradation_rate  (6 params always)
  - Per active edge: regulatory_strength, hill_coefficient  (2 params × num_edges)
  - Total: 6 + 2 * num_active_edges
"""

from itertools import product
from dataclasses import dataclass
from typing import Tuple, List

EDGE_TYPES = {0: "absent", 1: "activation", 2: "inhibition"}
GENES = ["A", "B", "C"]

# Edge index semantics
EDGE_NAMES = [
    "A→A", "B→B", "C→C",  # self-regulation
    "A→B", "A→C",          # A regulates others
    "B→A", "B→C",          # B regulates others
    "C→A", "C→B",          # C regulates others
]

@dataclass(frozen=True)
class Topology:
    edges: Tuple[int, ...]   # length-9 tuple, values in {0,1,2}

    @property
    def num_active_edges(self) -> int:
        return sum(1 for e in self.edges if e != 0)

    @property
    def num_params(self) -> int:
        return 6 + 2 * self.num_active_edges

    @property
    def has_isolated_node(self) -> bool:
        """Check if any gene has zero incoming AND zero outgoing edges."""
        for i, gene in enumerate(GENES):
            incoming = outgoing = False
            for j in range(9):
                src, tgt = _edge_source_target(j)
                if self.edges[j] != 0:
                    if tgt == i:
                        incoming = True
                    if src == i:
                        outgoing = True
            if not incoming and not outgoing:
                return True
        return False

    def is_valid(self, min_edges=2, max_edges=6) -> bool:
        n = self.num_active_edges
        return (min_edges <= n <= max_edges) and not self.has_isolated_node

    def to_label(self) -> str:
        """Human-readable motif label."""
        parts = []
        for idx, val in enumerate(self.edges):
            if val != 0:
                parts.append(f"{EDGE_NAMES[idx]}({'act' if val==1 else 'inh'})")
        return " | ".join(parts)


def _edge_source_target(edge_idx: int):
    """Return (source_gene_idx, target_gene_idx) for a given edge index."""
    mapping = [
        (0,0),(1,1),(2,2),  # self
        (0,1),(0,2),        # A→
        (1,0),(1,2),        # B→
        (2,0),(2,1),        # C→
    ]
    return mapping[edge_idx]


def enumerate_valid_topologies(min_edges=2, max_edges=6) -> List[Topology]:
    """Generate all valid topologies within bounds."""
    valid = []
    for combo in product(range(3), repeat=9):
        t = Topology(edges=combo)
        if t.is_valid(min_edges, max_edges):
            valid.append(t)
    return valid


def mutate_topology(t: Topology, rng) -> Topology:
    """Single-edge mutation: flip one random edge to a different value."""
    edges = list(t.edges)
    idx = rng.integers(0, 9)
    choices = [v for v in range(3) if v != edges[idx]]
    edges[idx] = rng.choice(choices)
    return Topology(edges=tuple(edges))
```

### Step 3 — Model Builder (`model_builder.py`)

Convert a topology + parameter vector into a runnable GillesPy2 model.

```python
"""
Model Builder: Topology × Parameters → GillesPy2 Stochastic Model

Reaction template per gene X:
  ∅ → X   (production, rate = basal + Σ regulatory_contributions)
  X → ∅   (degradation, rate = degradation_rate * X)

Regulatory contributions use Hill-like propensity functions:
  Activation by R on X:  strength * R^n / (K^n + R^n)
  Inhibition by R on X:  strength * K^n / (K^n + R^n)

Where strength, K (implicit in parameter bounds), and n are optimized per edge.

Parameter vector layout for a given topology:
  [basal_A, deg_A, basal_B, deg_B, basal_C, deg_C,
   strength_edge0, hill_edge0, strength_edge1, hill_edge1, ...]
  (only active edges included, in edge-index order)

Bounds:
  basal_production:  [0.1, 50.0]
  degradation_rate:  [0.001, 1.0]
  reg_strength:      [0.1, 100.0]
  hill_coefficient:  [1.0, 5.0]
"""

import gillespy2
import numpy as np
from grammar import Topology, GENES, _edge_source_target


def build_param_bounds(topology: Topology):
    """Return (lower_bounds, upper_bounds) arrays for the parameter vector."""
    lower, upper = [], []
    # Per-gene basal + degradation
    for _ in GENES:
        lower.extend([0.1, 0.001])
        upper.extend([50.0, 1.0])
    # Per-active-edge: strength, hill
    for idx, val in enumerate(topology.edges):
        if val != 0:
            lower.extend([0.1, 1.0])
            upper.extend([100.0, 5.0])
    return np.array(lower), np.array(upper)


def build_model(topology: Topology, params: np.ndarray,
                t_end=200.0, n_steps=1000) -> gillespy2.Model:
    """
    Construct a GillesPy2 model from topology and continuous parameter vector.

    Returns a Model ready for stochastic simulation.
    Raises ValueError if the model is structurally invalid.
    """
    model = gillespy2.Model(name="CircuitModel")

    # --- Species ---
    species = {}
    for gene in GENES:
        sp = gillespy2.Species(name=gene, initial_value=10)
        model.add_species(sp)
        species[gene] = sp

    # --- Parse parameter vector ---
    gene_params = {}
    ptr = 0
    for gene in GENES:
        gene_params[gene] = {
            "basal": params[ptr],
            "degradation": params[ptr + 1],
        }
        ptr += 2

    edge_params = {}
    for idx, val in enumerate(topology.edges):
        if val != 0:
            edge_params[idx] = {
                "strength": params[ptr],
                "hill": params[ptr + 1],
            }
            ptr += 2

    # --- Parameters (GillesPy2) ---
    gp2_params = {}
    for gene in GENES:
        for pname, pval in gene_params[gene].items():
            key = f"{gene}_{pname}"
            gp2_params[key] = gillespy2.Parameter(name=key, expression=str(float(pval)))
            model.add_parameter(gp2_params[key])
    for idx, ep in edge_params.items():
        for pname, pval in ep.items():
            key = f"edge{idx}_{pname}"
            gp2_params[key] = gillespy2.Parameter(name=key, expression=str(float(pval)))
            model.add_parameter(gp2_params[key])

    # --- Reactions ---
    for gi, gene in enumerate(GENES):
        # Production: basal rate + regulatory Hill terms
        rate_terms = [f"{gene}_basal"]
        for idx, val in enumerate(topology.edges):
            if val == 0:
                continue
            src_idx, tgt_idx = _edge_source_target(idx)
            if tgt_idx != gi:
                continue
            src_gene = GENES[src_idx]
            s = f"edge{idx}_strength"
            h = f"edge{idx}_hill"
            K = "20.0"  # fixed half-max for simplicity
            if val == 1:  # activation
                rate_terms.append(
                    f"{s} * {src_gene}**{h} / ({K}**{h} + {src_gene}**{h})"
                )
            else:  # inhibition
                rate_terms.append(
                    f"{s} * {K}**{h} / ({K}**{h} + {src_gene}**{h})"
                )

        prod_rate = " + ".join(rate_terms)
        model.add_reaction(gillespy2.Reaction(
            name=f"produce_{gene}",
            reactants={},
            products={species[gene]: 1},
            propensity_function=prod_rate,
        ))

        # Degradation
        model.add_reaction(gillespy2.Reaction(
            name=f"degrade_{gene}",
            reactants={species[gene]: 1},
            products={},
            rate=gp2_params[f"{gene}_degradation"],
        ))

    model.timespan(np.linspace(0, t_end, n_steps))
    return model
```

### Step 4 — Phenotype Evaluator (`evaluator.py`)

Score oscillation quality from stochastic simulation traces.

```python
"""
Phenotype Evaluator — Oscillation Quality Scoring

Given a GillesPy2 simulation result, compute a scalar score ∈ [0, 1]
measuring the quality of oscillatory behavior.

Score components (weighted sum, then clipped to [0, 1]):
  1. peak_count_score:      ≥ 5 peaks in observation window → full credit
  2. spacing_regularity:    1 - CV(peak_spacings), clipped
  3. amplitude_regularity:  1 - CV(peak_amplitudes), clipped
  4. persistence_score:     fraction of time window with active oscillation
  5. multi_gene_bonus:      bonus if ≥ 2 genes oscillate independently

Penalties:
  - No peaks detected → score = 0
  - Simulation failure/NaN → score = 0
  - Flat trace (std < 1) → score = 0
"""

import numpy as np
from scipy.signal import find_peaks


def score_trace(time: np.ndarray, concentrations: dict,
                min_peaks: int = 3) -> float:
    """
    Score a single simulation trace for oscillation quality.

    Args:
        time: 1D array of time points
        concentrations: dict mapping gene_name → 1D array of values
    Returns:
        float score in [0, 1]
    """
    gene_scores = []
    for gene, vals in concentrations.items():
        vals = np.asarray(vals, dtype=float)
        if np.any(np.isnan(vals)) or np.std(vals) < 1.0:
            gene_scores.append(0.0)
            continue

        # Detect peaks
        height_thresh = np.mean(vals) + 0.3 * np.std(vals)
        peaks, props = find_peaks(vals, height=height_thresh,
                                  distance=len(vals) // 50)
        if len(peaks) < min_peaks:
            gene_scores.append(0.0)
            continue

        # Peak spacing regularity
        spacings = np.diff(time[peaks])
        spacing_cv = np.std(spacings) / (np.mean(spacings) + 1e-9)
        spacing_score = max(0.0, 1.0 - spacing_cv)

        # Amplitude regularity
        amplitudes = vals[peaks]
        amp_cv = np.std(amplitudes) / (np.mean(amplitudes) + 1e-9)
        amp_score = max(0.0, 1.0 - amp_cv)

        # Peak count score (saturates at 8+ peaks)
        count_score = min(1.0, len(peaks) / 8.0)

        # Persistence: fraction of time window covered by oscillation
        if len(peaks) >= 2:
            osc_span = time[peaks[-1]] - time[peaks[0]]
            total_span = time[-1] - time[0]
            persistence = osc_span / (total_span + 1e-9)
        else:
            persistence = 0.0

        gene_score = (
            0.30 * spacing_score +
            0.25 * amp_score +
            0.25 * count_score +
            0.20 * persistence
        )
        gene_scores.append(gene_score)

    if not gene_scores or max(gene_scores) == 0:
        return 0.0

    # Best single-gene score + small bonus for multi-gene oscillation
    best = max(gene_scores)
    num_oscillating = sum(1 for s in gene_scores if s > 0.3)
    multi_bonus = 0.1 * min(1.0, (num_oscillating - 1) / 2.0) if num_oscillating > 1 else 0.0

    return float(np.clip(best + multi_bonus, 0.0, 1.0))


def evaluate_topology_score(model_builder_fn, topology, params,
                            n_seeds: int = 3, t_end: float = 200.0) -> float:
    """
    Run multiple stochastic simulations and return a robust aggregate score.

    Uses the median score across seeds to resist stochastic outliers.
    Returns 0.0 on any simulation failure.
    """
    scores = []
    for seed in range(n_seeds):
        try:
            model = model_builder_fn(topology, params, t_end=t_end)
            result = model.run(algorithm="SSA", seed=seed + 42)
            time = result["time"]
            conc = {g: result[g] for g in ["A", "B", "C"]}
            scores.append(score_trace(time, conc))
        except Exception:
            scores.append(0.0)

    return float(np.median(scores))
```

### Step 5 — Inner Optimizer (`inner_optimizer.py`)

Wrap fcmaes to optimize continuous parameters for a single topology.

```python
"""
Inner Optimizer — fcmaes parameter tuning for one topology.

For each topology T, find:
    x* = argmin_x  -phenotype_score(simulate(T, x))

Uses fcmaes.optimizer with Bite_cpp or Cma_cpp,
parallel retry for robustness against multimodality.

Key fcmaes features exploited:
  - Low per-evaluation overhead (C++ backend)
  - Coordinated parallel retry (many restarts, keep best)
  - Handles noisy objectives gracefully
"""

import numpy as np
from fcmaes import retry, advretry
from fcmaes.optimizer import Bite_cpp, Cma_cpp, de_cma, wrapper

from grammar import Topology
from model_builder import build_model, build_param_bounds
from evaluator import evaluate_topology_score


def make_objective(topology: Topology, n_seeds_inner: int = 2,
                   t_end: float = 200.0):
    """
    Return a callable objective(x) -> float for fcmaes.

    The objective is NEGATED because fcmaes minimizes.
    Returns a large positive penalty on failure.
    """
    lower, upper = build_param_bounds(topology)

    def objective(x: np.ndarray) -> float:
        try:
            score = evaluate_topology_score(
                build_model, topology, x,
                n_seeds=n_seeds_inner, t_end=t_end,
            )
            return -score  # fcmaes minimizes
        except Exception:
            return 1e6  # penalty

    return objective, lower, upper


def optimize_topology(topology: Topology,
                      max_evals: int = 2000,
                      n_retries: int = 8,
                      n_seeds_inner: int = 2) -> dict:
    """
    Run fcmaes inner optimization for a single topology.

    Returns dict with keys:
      - "best_score": float (positive, higher = better)
      - "best_params": np.ndarray
      - "num_evals": int
      - "topology": Topology
    """
    objective, lower, upper = make_objective(topology, n_seeds_inner)

    # Use coordinated parallel retry with Bite_cpp
    # This is the key fcmaes strength: many restarts, low overhead
    result = retry.minimize(
        objective,
        bounds=retry.Bounds(lower, upper),
        num_retries=n_retries,
        max_evaluations=max_evals,
        optimizer=Bite_cpp(max_evals),
    )

    return {
        "best_score": -result.fun,   # un-negate
        "best_params": result.x,
        "num_evals": max_evals * n_retries,
        "topology": topology,
    }
```

### Step 6 — Outer Loop Strategies (`outer_loop.py`)

Implement three outer-loop strategies in order of complexity.

```python
"""
Outer Loop — Topology Search Strategies

Strategy 1: Random sampling (baseline)
Strategy 2: Evolutionary mutation (1+1 style)
Strategy 3: LLM-guided proposal (agentic, see agentic_loop.py)

All strategies use the same interface:
  - propose_topology() → Topology
  - report_result(topology, score, params)
"""

import json
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

from grammar import (
    Topology, enumerate_valid_topologies, mutate_topology
)
from inner_optimizer import optimize_topology


@dataclass
class SearchResult:
    topology: Topology
    score: float
    params: Optional[np.ndarray] = None
    iteration: int = 0


class Archive:
    """Stores all evaluated topologies and their scores."""

    def __init__(self):
        self.results: List[SearchResult] = []

    def add(self, result: SearchResult):
        self.results.append(result)

    @property
    def best(self) -> Optional[SearchResult]:
        if not self.results:
            return None
        return max(self.results, key=lambda r: r.score)

    def top_k(self, k: int = 5) -> List[SearchResult]:
        return sorted(self.results, key=lambda r: r.score, reverse=True)[:k]

    def summary(self) -> str:
        if not self.results:
            return "No results yet."
        lines = [f"Evaluated {len(self.results)} topologies."]
        for i, r in enumerate(self.top_k(5)):
            lines.append(
                f"  #{i+1}: score={r.score:.4f}  "
                f"edges={r.topology.num_active_edges}  "
                f"motif=[{r.topology.to_label()}]"
            )
        return "\n".join(lines)


# ──────────────────────────────────────────
# Strategy 1: Random Search
# ──────────────────────────────────────────

def run_random_search(n_candidates: int = 50,
                      max_evals_inner: int = 2000,
                      seed: int = 0) -> Archive:
    """Evaluate n random valid topologies."""
    rng = np.random.default_rng(seed)
    all_valid = enumerate_valid_topologies()
    rng.shuffle(all_valid)
    candidates = all_valid[:n_candidates]

    archive = Archive()
    for i, topo in enumerate(candidates):
        print(f"[Random {i+1}/{n_candidates}] {topo.to_label()}")
        result = optimize_topology(topo, max_evals=max_evals_inner)
        archive.add(SearchResult(
            topology=topo,
            score=result["best_score"],
            params=result["best_params"],
            iteration=i,
        ))
        print(f"  → score = {result['best_score']:.4f}")
    return archive


# ──────────────────────────────────────────
# Strategy 2: Evolutionary (1+1) Mutation
# ──────────────────────────────────────────

def run_evolutionary_search(n_iterations: int = 80,
                            max_evals_inner: int = 2000,
                            seed: int = 0) -> Archive:
    """
    Simple (1+1)-ES over topologies:
      - Start from a random valid topology
      - Each iteration: mutate, optimize, keep if better
      - Also keep a global archive of all evaluations
    """
    rng = np.random.default_rng(seed)
    all_valid = enumerate_valid_topologies()

    # Start from random valid topology
    current = all_valid[rng.integers(len(all_valid))]
    current_result = optimize_topology(current, max_evals=max_evals_inner)
    current_score = current_result["best_score"]

    archive = Archive()
    archive.add(SearchResult(current, current_score, current_result["best_params"], 0))

    for i in range(1, n_iterations):
        # Mutate until valid
        candidate = current
        for _ in range(20):
            candidate = mutate_topology(current, rng)
            if candidate.is_valid():
                break

        if not candidate.is_valid():
            continue

        print(f"[Evo {i}/{n_iterations}] {candidate.to_label()}")
        result = optimize_topology(candidate, max_evals=max_evals_inner)
        score = result["best_score"]
        archive.add(SearchResult(candidate, score, result["best_params"], i))
        print(f"  → score = {score:.4f} (current best = {current_score:.4f})")

        if score > current_score:
            current = candidate
            current_score = score
            print(f"  ★ New best topology!")

    return archive
```

### Step 7 — LLM-Guided Agentic Loop (`agentic_loop.py`)

The agentic outer loop that uses an LLM to propose topologies based on accumulated evidence. **Implement this ONLY after Strategies 1 and 2 are working.**

```python
"""
Agentic Outer Loop — LLM-Guided Topology Proposal

The LLM receives:
  - The topology grammar specification
  - The archive of previously evaluated topologies and scores
  - Phenotype descriptions of the best candidates so far

The LLM proposes:
  - A new topology encoding (9-integer tuple)
  - A brief rationale for why this topology might oscillate

The system:
  - Validates the proposal against the grammar
  - Runs fcmaes inner optimization
  - Reports the result back to the LLM
  - Repeats for N iterations

This mirrors the autoresearch-trading pattern:
  LLM proposes STRUCTURE, fcmaes optimizes NUMBERS.

Integration options:
  - OpenAI-compatible API (local or remote)
  - Anthropic API
  - Any LLM with structured output support
"""

import json
import numpy as np
from typing import Optional

from grammar import Topology, EDGE_NAMES, EDGE_TYPES
from inner_optimizer import optimize_topology
from outer_loop import Archive, SearchResult


SYSTEM_PROMPT = """\
You are a synthetic biology circuit designer. You search for 3-gene
regulatory network topologies that produce robust stochastic oscillations.

## Topology Encoding

A topology is a tuple of 9 integers, each ∈ {0, 1, 2}:
  0 = no interaction
  1 = activation
  2 = inhibition

Edge order: [A→A, B→B, C→C, A→B, A→C, B→A, B→C, C→A, C→B]

Constraints:
  - Between 2 and 6 active edges (non-zero values)
  - No isolated nodes (every gene must have ≥1 incoming or outgoing edge)

## Your Task

Given the history of previously tested topologies and their oscillation
scores (0 = no oscillation, 1 = perfect oscillation), propose the NEXT
topology to test.

Think about which motifs tend to oscillate (e.g., negative feedback loops,
repressilator-like structures, delayed negative feedback with positive
amplification) and use the observed scores to guide your proposals.

Respond with ONLY a JSON object:
{
  "topology": [int, int, int, int, int, int, int, int, int],
  "rationale": "Brief explanation of why this topology might oscillate."
}
"""


def format_history_for_llm(archive: Archive, max_entries: int = 20) -> str:
    """Format the evaluation archive as context for the LLM."""
    if not archive.results:
        return "No topologies evaluated yet. Start with a classic motif."

    lines = ["Previously tested topologies (sorted by score, descending):"]
    for r in archive.top_k(max_entries):
        edge_desc = r.topology.to_label()
        lines.append(f"  edges=[{r.topology.edges}] score={r.score:.4f}  ({edge_desc})")

    stats = {
        "total_evaluated": len(archive.results),
        "best_score": archive.best.score if archive.best else 0,
        "mean_score": np.mean([r.score for r in archive.results]),
    }
    lines.append(f"\nStats: {json.dumps(stats, indent=2)}")
    return "\n".join(lines)


def parse_llm_response(response_text: str) -> Optional[Topology]:
    """Parse the LLM's JSON response into a Topology."""
    try:
        # Handle markdown code fences
        text = response_text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0]

        data = json.loads(text)
        edges = tuple(data["topology"])

        if len(edges) != 9 or not all(e in (0, 1, 2) for e in edges):
            print(f"  [LLM] Invalid edge values: {edges}")
            return None

        topo = Topology(edges=edges)
        if not topo.is_valid():
            print(f"  [LLM] Topology fails validity check: {topo.to_label()}")
            return None

        print(f"  [LLM] Rationale: {data.get('rationale', 'none given')}")
        return topo

    except (json.JSONDecodeError, KeyError, TypeError) as e:
        print(f"  [LLM] Failed to parse response: {e}")
        return None


def run_agentic_search(
    llm_call_fn,  # callable(system_prompt, user_prompt) -> str
    n_iterations: int = 30,
    max_evals_inner: int = 2000,
    seed_archive: Optional[Archive] = None,
) -> Archive:
    """
    Run the LLM-guided agentic topology search.

    Args:
        llm_call_fn: Function that takes (system_prompt, user_prompt)
                     and returns the LLM's text response.
                     This is intentionally generic so any LLM backend works.
        n_iterations: Number of LLM proposals to evaluate.
        max_evals_inner: Budget for fcmaes inner optimization per topology.
        seed_archive: Optional archive from a prior search to warm-start.

    Returns:
        Archive with all evaluation results.
    """
    archive = seed_archive or Archive()
    consecutive_failures = 0

    for i in range(n_iterations):
        print(f"\n{'='*60}")
        print(f"[Agentic iteration {i+1}/{n_iterations}]")

        # Build prompt with history
        history = format_history_for_llm(archive)
        user_prompt = (
            f"Iteration {i+1}/{n_iterations}.\n\n"
            f"{history}\n\n"
            "Propose the next topology to evaluate. "
            "Respond with ONLY the JSON object."
        )

        # Query LLM
        try:
            response = llm_call_fn(SYSTEM_PROMPT, user_prompt)
        except Exception as e:
            print(f"  [LLM] API error: {e}")
            consecutive_failures += 1
            if consecutive_failures > 3:
                print("  Too many consecutive LLM failures. Stopping.")
                break
            continue

        # Parse proposal
        topology = parse_llm_response(response)
        if topology is None:
            consecutive_failures += 1
            if consecutive_failures > 5:
                print("  Too many invalid proposals. Stopping.")
                break
            continue

        consecutive_failures = 0
        print(f"  Topology: {topology.to_label()}")

        # Check if already evaluated
        already_seen = any(
            r.topology.edges == topology.edges for r in archive.results
        )
        if already_seen:
            print("  Already evaluated this topology — skipping.")
            continue

        # Run fcmaes inner optimization
        result = optimize_topology(topology, max_evals=max_evals_inner)
        archive.add(SearchResult(
            topology=topology,
            score=result["best_score"],
            params=result["best_params"],
            iteration=i,
        ))
        print(f"  → score = {result['best_score']:.4f}")
        print(f"\n{archive.summary()}")

    return archive
```

### Step 8 — Main Entry Point (`run_search.py`)

```python
"""
Main entry point — run the full pipeline.

Usage:
    python run_search.py --strategy random --n 30
    python run_search.py --strategy evo --n 50
    python run_search.py --strategy agentic --n 20
"""

import argparse
import json
import pickle
from pathlib import Path

from outer_loop import run_random_search, run_evolutionary_search
from agentic_loop import run_agentic_search


def make_anthropic_llm_fn():
    """Create an LLM call function using the Anthropic API."""
    import anthropic
    client = anthropic.Anthropic()

    def call(system_prompt: str, user_prompt: str) -> str:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return response.content[0].text

    return call


def main():
    parser = argparse.ArgumentParser(
        description="Split-brain circuit topology search with fcmaes"
    )
    parser.add_argument("--strategy", choices=["random", "evo", "agentic"],
                        default="random")
    parser.add_argument("--n", type=int, default=30,
                        help="Number of topologies to evaluate")
    parser.add_argument("--inner-evals", type=int, default=2000,
                        help="fcmaes evaluation budget per topology")
    parser.add_argument("--output", type=str, default="results",
                        help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(exist_ok=True)

    if args.strategy == "random":
        archive = run_random_search(n_candidates=args.n,
                                    max_evals_inner=args.inner_evals)
    elif args.strategy == "evo":
        archive = run_evolutionary_search(n_iterations=args.n,
                                          max_evals_inner=args.inner_evals)
    elif args.strategy == "agentic":
        llm_fn = make_anthropic_llm_fn()
        # Optionally warm-start from a prior random search
        archive = run_agentic_search(llm_fn, n_iterations=args.n,
                                     max_evals_inner=args.inner_evals)

    # Save results
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(archive.summary())

    with open(out_dir / "archive.pkl", "wb") as f:
        pickle.dump(archive, f)

    # Save human-readable summary
    summary = {
        "strategy": args.strategy,
        "n_evaluated": len(archive.results),
        "best_score": archive.best.score if archive.best else 0,
        "best_topology": list(archive.best.topology.edges) if archive.best else [],
        "best_motif": archive.best.topology.to_label() if archive.best else "",
        "top_5": [
            {"edges": list(r.topology.edges), "score": r.score, "motif": r.topology.to_label()}
            for r in archive.top_k(5)
        ],
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {out_dir}/")


if __name__ == "__main__":
    main()
```

---

## Key Design Principles

1. **Start without the LLM.** Strategies 1 (random) and 2 (evolutionary) must produce credible results before the agentic loop is added. The benchmark and evaluator must be trustworthy first.

2. **fcmaes is the inner engine, not an afterthought.** Every topology is scored by its *best achievable performance after continuous optimization*. This makes fcmaes essential — without it, the outer loop cannot fairly compare topologies.

3. **Bounded grammar, not arbitrary chemistry.** The 3-node / 9-edge grammar yields ~3000–5000 valid topologies. This is large enough to show that structure matters, small enough to search meaningfully.

4. **Stochastic robustness built in.** Every evaluation uses multiple SSA seeds. The median score resists lucky outliers. The inner optimizer sees noise but fcmaes handles it via parallel retry.

5. **The agentic loop is constrained.** The LLM proposes topologies *within the grammar*, not arbitrary Python code. It receives structured feedback (scores, motif labels), not raw simulation logs. This keeps the loop safe and reproducible.

6. **Clean separation of concerns.** Each module has one job. The model builder knows nothing about optimization. The evaluator knows nothing about topologies. The outer loop knows nothing about simulation internals.

---

## Dependencies

```
pip install fcmaes gillespy2 numpy scipy anthropic
```

For the agentic loop, set `ANTHROPIC_API_KEY` in environment. Any OpenAI-compatible endpoint can be substituted by changing `make_anthropic_llm_fn()`.

---

## Expected Outcomes

When the pipeline works correctly, you should observe:

- **Random search** finds a few decent oscillators among ~30–50 candidates, with most topologies scoring near zero.
- **Evolutionary search** converges faster, finding repressilator-like motifs (A→B inhibition, B→C inhibition, C→A inhibition) or delayed negative feedback loops.
- **Agentic search** should match or beat evolutionary search in fewer evaluations by leveraging biological knowledge to propose promising motifs early.

The key demonstration: **the same topology scores very differently depending on parameter quality**, proving that the fcmaes inner loop is essential, not optional.

---

## Validation Checklist

Before considering the implementation complete:

- [ ] `grammar.py`: `enumerate_valid_topologies()` returns 2000–5000 topologies
- [ ] `model_builder.py`: `build_model()` produces a runnable GillesPy2 model for at least 5 different topologies
- [ ] `evaluator.py`: Known oscillator (repressilator-like) scores > 0.5; known non-oscillator scores < 0.1
- [ ] `inner_optimizer.py`: fcmaes finds parameters that improve a repressilator's score from random initialization
- [ ] `outer_loop.py`: Random search of 30 topologies completes without crashes
- [ ] `outer_loop.py`: Evolutionary search finds a topology scoring > 0.5 within 50 iterations
- [ ] `agentic_loop.py`: LLM proposes valid topologies and the loop runs for 10+ iterations
- [ ] End-to-end: `run_search.py --strategy evo --n 50` produces a `summary.json` with meaningful results
