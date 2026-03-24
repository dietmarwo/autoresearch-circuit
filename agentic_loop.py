"""
Agentic Outer Loop — LLM-Guided Topology Proposal

The LLM receives:
  - The topology grammar specification
  - The archive of previously evaluated topologies and scores
  - Phenotype descriptions of the best candidates so far

The LLM proposes:
  - A new topology encoding (9-integer tuple)
  - A brief rationale

The system validates, optimises with fcmaes, and reports back.

This mirrors the autoresearch-trading pattern:
  LLM proposes STRUCTURE, fcmaes optimises NUMBERS.

Integration: uses any callable(system_prompt, user_prompt) → str.
Default implementation uses the Anthropic API.
"""

import json
import time
from typing import Optional, Callable

import numpy as np

from grammar import Topology
from inner_optimizer import optimize_topology
from archive import Archive, SearchResult
import config as cfg


# ── System prompt ─────────────────────────────────────────

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

## Known Oscillator Motifs

Good oscillators often feature:
  - Negative feedback loops (e.g. repressilator: A⊣B, B⊣C, C⊣A)
  - Delayed negative feedback with positive amplification
  - Combinations of activation and inhibition forming limit cycles
  - Self-inhibition can sometimes stabilise oscillation frequency

Bad topologies:
  - Pure positive feedback (usually leads to runaway or bistability)
  - No feedback loops at all
  - Very few edges (< 3 usually too simple)

## Your Task

Given the history of previously tested topologies and their oscillation
scores (0 = no oscillation, 1 = perfect oscillation), propose the NEXT
topology to test. Use the scores to learn which structural motifs work.

Respond with ONLY a JSON object:
{
  "topology": [int, int, int, int, int, int, int, int, int],
  "rationale": "Brief explanation of why this topology might oscillate."
}
"""


# ── Helper functions ──────────────────────────────────────

def format_history_for_llm(archive: Archive,
                           max_entries: int = cfg.LLM_HISTORY_TOP_K) -> str:
    """Format the evaluation archive as context for the LLM."""
    if not archive.results:
        return (
            "No topologies evaluated yet. Start with a classic oscillator motif "
            "like the repressilator (A⊣B, B⊣C, C⊣A) or a Goodwin-style loop."
        )

    lines = ["Previously tested topologies (sorted by score, descending):\n"]

    for rank, r in enumerate(archive.top_k(max_entries), 1):
        lines.append(
            f"  #{rank}: edges={list(r.topology.edges)}  "
            f"score={r.score:.4f}  ({r.topology.to_label()})"
        )

    stats = archive.score_stats()
    lines.append(f"\nSummary: {stats['n']} evaluated, "
                 f"best={stats['best']:.4f}, mean={stats['mean']:.4f}, "
                 f"nonzero={stats['nonzero']}")
    return "\n".join(lines)


def parse_llm_response(response_text: str) -> Optional[Topology]:
    """Parse the LLM's JSON response into a validated Topology."""
    try:
        text = response_text.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        data = json.loads(text)
        edges = tuple(int(e) for e in data["topology"])

        if len(edges) != 9:
            print(f"  [LLM] Expected 9 edges, got {len(edges)}")
            return None
        if not all(e in (0, 1, 2) for e in edges):
            print(f"  [LLM] Invalid edge values: {edges}")
            return None

        topo = Topology(edges=edges)
        if not topo.is_valid():
            print(f"  [LLM] Topology fails grammar constraints: {topo.to_label()}")
            return None

        rationale = data.get("rationale", "(no rationale)")
        print(f"  [LLM] Rationale: {rationale}")
        return topo

    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        print(f"  [LLM] Failed to parse response: {e}")
        return None


# ── Main agentic search ──────────────────────────────────

def run_agentic_search(
    llm_call_fn: Callable[[str, str], str],
    n_iterations: int = cfg.AGENTIC_SEARCH_N,
    max_evals_inner: int = cfg.INNER_MAX_EVALS,
    n_retries: int = cfg.INNER_NUM_RETRIES,
    seed_archive: Optional[Archive] = None,
) -> Archive:
    """
    Run the LLM-guided agentic topology search.

    Args:
        llm_call_fn:     callable(system_prompt, user_prompt) → str
        n_iterations:    Number of LLM proposals to evaluate.
        max_evals_inner: Budget for fcmaes inner optimisation per topology.
        n_retries:       Number of fcmaes restarts per topology.
        seed_archive:    Optional archive from a prior search to warm-start.

    Returns:
        Archive with all evaluation results.
    """
    archive = seed_archive if seed_archive is not None else Archive()
    consecutive_failures = 0
    total_t0 = time.perf_counter()

    for i in range(n_iterations):
        print(f"\n{'='*60}")
        print(f"[Agentic {i+1}/{n_iterations}]")

        # ── Build prompt with history ─────────────────────
        history = format_history_for_llm(archive)
        user_prompt = (
            f"Iteration {i+1} of {n_iterations}.\n\n"
            f"{history}\n\n"
            "Propose the next topology to evaluate.\n"
            "Respond with ONLY the JSON object."
        )

        # ── Query LLM ────────────────────────────────────
        try:
            response = llm_call_fn(SYSTEM_PROMPT, user_prompt)
        except Exception as e:
            print(f"  [LLM] API error: {e}")
            consecutive_failures += 1
            if consecutive_failures > cfg.LLM_MAX_CONSECUTIVE_FAILURES:
                print("  Too many consecutive LLM failures. Stopping.")
                break
            continue

        # ── Parse proposal ────────────────────────────────
        topology = parse_llm_response(response)
        if topology is None:
            consecutive_failures += 1
            if consecutive_failures > cfg.LLM_MAX_CONSECUTIVE_FAILURES:
                print("  Too many invalid proposals. Stopping.")
                break
            continue

        consecutive_failures = 0
        print(f"  Proposed: {topology.to_label()}")

        # ── Skip duplicates ───────────────────────────────
        if archive.already_evaluated(topology):
            print("  Already evaluated — skipping.")
            continue

        # ── Inner optimisation ────────────────────────────
        result = optimize_topology(
            topology, max_evals=max_evals_inner, n_retries=n_retries,
            verbose=True,
        )
        archive.add(SearchResult(
            topology=topology,
            score=result["best_score"],
            params=result["best_params"],
            iteration=i,
            wall_time=result["wall_time"],
            strategy="agentic",
        ))
        print(f"  → score = {result['best_score']:.4f}")
        print(f"\n{archive.summary()}")

    total_wall = time.perf_counter() - total_t0
    print(f"\nAgentic search done in {total_wall:.1f}s")
    print(archive.summary())
    return archive


# ── Default LLM backends ─────────────────────────────────

def make_anthropic_llm_fn() -> Callable[[str, str], str]:
    """Create an LLM call function using the Anthropic API."""
    import anthropic
    client = anthropic.Anthropic()

    def call(system_prompt: str, user_prompt: str) -> str:
        response = client.messages.create(
            model=cfg.LLM_MODEL,
            max_tokens=cfg.LLM_MAX_TOKENS,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return response.content[0].text

    return call


def make_openai_compatible_llm_fn(
    base_url: str = "http://localhost:8000/v1",
    model: str = "default",
    api_key: str = "not-needed",
) -> Callable[[str, str], str]:
    """Create an LLM call function for any OpenAI-compatible API."""
    from openai import OpenAI
    client = OpenAI(base_url=base_url, api_key=api_key)

    def call(system_prompt: str, user_prompt: str) -> str:
        response = client.chat.completions.create(
            model=model,
            max_tokens=cfg.LLM_MAX_TOKENS,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content

    return call
