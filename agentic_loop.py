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

Integration: uses any callable(messages) → str.
Built-in helpers support:
  - OpenAI-compatible APIs
  - Anthropic Claude (native SDK)
  - Google Gemini (native SDK)
  - MiniMax via Anthropic-compatible API
"""

import json
import os
import re
import time
from typing import Callable, Optional, Tuple

from grammar import Topology
from inner_optimizer import optimize_topology
from archive import Archive, SearchResult
import config as cfg


# ── System prompt ─────────────────────────────────────────

GUIDED_MOTIF_SECTION_3 = """\
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
"""

GUIDED_MOTIF_SECTION_5 = """\
## Robust 5-Gene Design Priors

Good robust oscillators often feature:
  - Multiple overlapping negative feedback loops
  - Backup sub-oscillators that can survive single-gene failure
  - Mixed sparse cores plus extra reinforcing loops
  - Alternative motif families beyond a single minimal 3-gene ring

Bad topologies:
  - Pure positive feedback with no delayed repression
  - One fragile minimal loop with no redundant support
  - Extremely dense topologies with no clear oscillatory backbone
"""

BLIND_MOTIF_SECTION = """\
## Search Discipline

This is a BLIND benchmark mode.
Do not assume any named canonical motif is optimal.
Use only the grammar constraints and the observed search history.

Bad topologies:
  - Pure positive feedback (usually leads to runaway or bistability)
  - No feedback loops at all
  - Very few edges (< 3 usually too simple)
"""


def _json_topology_template() -> str:
    """Return a compact zero-filled topology template for prompts."""
    return "[" + ",".join("0" for _ in range(cfg.NUM_EDGE_SLOTS)) + "]"


def _edge_order_text() -> str:
    """Return the current experiment's edge ordering for prompt display."""
    return "[" + ", ".join(cfg.EDGE_NAMES) + "]"


def _experiment_prompt_header() -> str:
    """Describe the active experiment objective to the LLM."""
    if cfg.EXPERIMENT == "robust5":
        return (
            f"You are a synthetic biology circuit designer. You search for {cfg.NUM_GENES}-gene\n"
            "regulatory network topologies that sustain stochastic oscillations and remain\n"
            "oscillatory after single-gene failures, partial knockdowns, and local\n"
            "parameter perturbations.\n"
        )
    return (
        f"You are a synthetic biology circuit designer. You search for {cfg.NUM_GENES}-gene\n"
        "regulatory network topologies that produce robust stochastic oscillations.\n"
    )


def _experiment_metrics_text() -> str:
    """Explain how experiment results are reported back to the model."""
    if cfg.EXPERIMENT == "robust5":
        return f"""\
Each experiment reports:
  - rank_score = validation_score - {cfg.GENERALIZATION_GAP_PENALTY:.1f} * |train_score - validation_score|
  - train_score / validation_score: weighted robust-oscillator score
  - full_score: intact-network oscillation quality
  - knockout_score: q{int(round(100 * cfg.ROBUST_SCENARIO_AGGREGATION_QUANTILE)):02d} score across single-gene knockouts
  - knockout_pass_rate: fraction of knockouts with score ≥ {cfg.ROBUST_SUCCESS_THRESHOLD:.2f}
  - knockdown_score: q{int(round(100 * cfg.ROBUST_SCENARIO_AGGREGATION_QUANTILE)):02d} score across partial single-gene knockdowns
  - perturb_score: q{int(round(100 * cfg.ROBUST_SCENARIO_AGGREGATION_QUANTILE)):02d} score across local parameter jitters around the fcmaes optimum
  - gap: |train_score - validation_score|
  - period: approximate intact-network oscillation period

Use validation_score, knockout_score, knockdown_score, perturb_score, and knockout_pass_rate as the main learning signals.
Do not chase train_score alone.
"""
    return f"""\
Each experiment reports:
  - rank_score = validation_score - {cfg.GENERALIZATION_GAP_PENALTY:.1f} * |train_score - validation_score|
  - train_score: short-horizon score used during parameter optimisation
  - validation_score: longer-horizon holdout score on different seeds
  - gap: |train_score - validation_score|
  - period: approximate validation oscillation period

Use validation_score and small gap as the main learning signal.
Do not chase train_score alone.
"""


def build_system_prompt(agentic_mode: str = cfg.AGENTIC_MODE) -> str:
    """Build the mode-specific system prompt for the LLM."""
    if agentic_mode == "guided":
        motif_section = GUIDED_MOTIF_SECTION_5 if cfg.EXPERIMENT == "robust5" else GUIDED_MOTIF_SECTION_3
    else:
        motif_section = BLIND_MOTIF_SECTION
    return f"""\
{_experiment_prompt_header()}

## Topology Encoding

A topology is a tuple of {cfg.NUM_EDGE_SLOTS} integers, each ∈ {{0, 1, 2}}:
  0 = no interaction
  1 = activation
  2 = inhibition

Edge order: {_edge_order_text()}

Constraints:
  - Between {cfg.MIN_ACTIVE_EDGES} and {cfg.MAX_ACTIVE_EDGES} active edges (non-zero values)
  - No isolated nodes (every gene must have ≥1 incoming or outgoing edge)

{motif_section}

## Your Task

Given the history of previously tested topologies and their measured
performance, propose the NEXT topology to test.

The archive preserves both:
  - best overall topologies
  - best representative per structural niche

Each structural niche summarizes:
  - active edge count
  - activation vs inhibition count
  - number of self-regulatory edges
  - coarse motif flags such as repressilator / toggle / mixed_cycle / other

{_experiment_metrics_text()}

Respond with ONLY a JSON object:
{{
  "topology": {_json_topology_template()},
  "rationale": "Brief explanation of why this topology might oscillate."
}}
"""


# ── Helper functions ──────────────────────────────────────

def _fmt_metric(value: Optional[float], digits: int = 4, default: str = "n/a") -> str:
    """Format optional float metrics for logs and prompts."""
    if value is None:
        return default
    return f"{value:.{digits}f}"


def hamming_distance(a: Topology, b: Topology) -> int:
    """Count the number of edge slots that differ between two topologies."""
    return sum(x != y for x, y in zip(a.edges, b.edges))


def select_agentic_phase(agentic_mode: str,
                         archive_size: int,
                         bootstrap_iters: int = cfg.AGENTIC_BOOTSTRAP_ITERS) -> str:
    """Choose the current outer-loop phase for prompting and diversity control."""
    if archive_size < bootstrap_iters:
        return "bootstrap"
    if agentic_mode == "blind":
        return "blind"
    return "explore" if (archive_size - bootstrap_iters) % 2 == 0 else "exploit"


def format_bootstrap_history(archive: Archive) -> str:
    """Summarize tried topologies during the no-score bootstrap phase."""
    lines = [f"Bootstrap evaluations completed: {len(archive)}"]
    if len(archive) == 0:
        lines.append("No topologies evaluated yet.")
        return "\n".join(lines)
    lines.append("Tried topologies so far (scores intentionally hidden):")
    for result in archive.results[-cfg.LLM_RECENT_K:]:
        lines.append(
            f"  iter={result.iteration}  "
            f"edges={list(result.topology.edges)}  "
            f"active={result.topology.num_active_edges}  "
            f"niche={result.niche_key}  "
            f"motif={result.topology.to_label()}"
        )
    return "\n".join(lines)


def build_current_best_block(archive: Archive) -> Optional[str]:
    """Format the current best result for exploit-style prompts."""
    best = archive.best
    if best is None:
        return None
    return (
        "\nCurrent best topology:\n"
        f"  rank={best.score:.4f}\n"
        f"  validation_score={_fmt_metric(best.validation_score)}\n"
        f"  train_score={_fmt_metric(best.train_score)}\n"
        f"  gap={_fmt_metric(best.generalization_gap)}\n"
        f"  period={_fmt_metric(best.validation_period, digits=1)}\n"
        f"  full_score={_fmt_metric(best.validation_full_score)}\n"
        f"  knockout_score={_fmt_metric(best.validation_knockout_score)}\n"
        f"  knockout_pass_rate={_fmt_metric(best.validation_knockout_pass_rate, digits=2)}\n"
        f"  knockdown_score={_fmt_metric(best.validation_knockdown_score)}\n"
        f"  perturb_score={_fmt_metric(best.validation_param_perturb_score)}\n"
        f"  niche={best.niche_key}\n"
        f"  edges={list(best.topology.edges)}\n"
        f"  motif={best.topology.to_label()}"
    )


def format_history_for_llm(archive: Archive,
                           top_k: int = cfg.LLM_TOP_K,
                           niche_k: int = cfg.LLM_NICHE_K,
                           recent_k: int = cfg.LLM_RECENT_K) -> str:
    """Build a compact regenerated history block for each new turn."""
    stats = archive.score_stats()
    lines = [
        f"Evaluations run: {stats.get('n', 0)}",
    ]
    if stats.get("n", 0) == 0:
        lines.append("No topologies evaluated yet.")
        return "\n".join(lines)

    lines.append(
        f"Best rank so far: {stats['best']:.4f}  "
        f"mean_rank={stats['mean']:.4f}  median_rank={stats['median']:.4f}  "
        f"nonzero={stats['nonzero']}"
    )
    lines.append(f"Structural niches discovered: {stats['niches']}")
    if "best_validation" in stats:
        lines.append(
            f"Best validation so far: {stats['best_validation']:.4f}  "
            f"mean_validation={stats['mean_validation']:.4f}"
        )
    if "mean_gap" in stats:
        lines.append(f"Mean train-validation gap: {stats['mean_gap']:.4f}")

    top_results = archive.top_k(top_k)
    if top_results:
        lines.append("\nBest overall topologies:")
        for rank, result in enumerate(top_results, 1):
            lines.append(
                f"  #{rank}: rank={result.score:.4f}  "
                f"val={_fmt_metric(result.validation_score)}  "
                f"train={_fmt_metric(result.train_score)}  "
                f"gap={_fmt_metric(result.generalization_gap)}  "
                f"period={_fmt_metric(result.validation_period, digits=1)}  "
                f"full={_fmt_metric(result.validation_full_score)}  "
                f"ko={_fmt_metric(result.validation_knockout_score)}  "
                f"pass={_fmt_metric(result.validation_knockout_pass_rate, digits=2)}  "
                f"kd={_fmt_metric(result.validation_knockdown_score)}  "
                f"pert={_fmt_metric(result.validation_param_perturb_score)}  "
                f"niche={result.niche_key}  "
                f"edges={list(result.topology.edges)}  "
                f"active={result.topology.num_active_edges}  "
                f"motif={result.topology.to_label()}"
            )

    niche_results = archive.niche_elites(niche_k)
    if niche_results:
        lines.append("\nBest structural niche elites:")
        for rank, result in enumerate(niche_results, 1):
            lines.append(
                f"  [N{rank}] niche={result.niche_key}  "
                f"rank={result.score:.4f}  "
                f"val={_fmt_metric(result.validation_score)}  "
                f"train={_fmt_metric(result.train_score)}  "
                f"gap={_fmt_metric(result.generalization_gap)}  "
                f"period={_fmt_metric(result.validation_period, digits=1)}  "
                f"full={_fmt_metric(result.validation_full_score)}  "
                f"ko={_fmt_metric(result.validation_knockout_score)}  "
                f"pass={_fmt_metric(result.validation_knockout_pass_rate, digits=2)}  "
                f"kd={_fmt_metric(result.validation_knockdown_score)}  "
                f"pert={_fmt_metric(result.validation_param_perturb_score)}  "
                f"edges={list(result.topology.edges)}  "
                f"motif={result.topology.to_label()}"
            )

    recent_results = archive.results[-recent_k:]
    if recent_results:
        lines.append("\nRecent evaluations:")
        for result in recent_results:
            lines.append(
                f"  iter={result.iteration}  rank={result.score:.4f}  "
                f"val={_fmt_metric(result.validation_score)}  "
                f"train={_fmt_metric(result.train_score)}  "
                f"gap={_fmt_metric(result.generalization_gap)}  "
                f"period={_fmt_metric(result.validation_period, digits=1)}  "
                f"full={_fmt_metric(result.validation_full_score)}  "
                f"ko={_fmt_metric(result.validation_knockout_score)}  "
                f"pass={_fmt_metric(result.validation_knockout_pass_rate, digits=2)}  "
                f"kd={_fmt_metric(result.validation_knockdown_score)}  "
                f"pert={_fmt_metric(result.validation_param_perturb_score)}  "
                f"niche={result.niche_key}  "
                f"edges={list(result.topology.edges)}  "
                f"motif={result.topology.to_label()}"
            )

    return "\n".join(lines)


def build_user_message(archive: Archive,
                       iteration: int,
                       n_iterations: int,
                       agentic_mode: str = cfg.AGENTIC_MODE,
                       phase: Optional[str] = None,
                       bootstrap_iters: int = cfg.AGENTIC_BOOTSTRAP_ITERS,
                       explore_min_hamming: int = cfg.AGENTIC_EXPLORE_MIN_HAMMING,
                       top_k: int = cfg.LLM_TOP_K,
                       niche_k: int = cfg.LLM_NICHE_K,
                       recent_k: int = cfg.LLM_RECENT_K) -> str:
    """Rebuild the full prompt for the current turn from archive state."""
    if phase is None:
        phase = select_agentic_phase(agentic_mode, len(archive), bootstrap_iters)
    parts = []

    if phase == "bootstrap":
        parts.append(format_bootstrap_history(archive))
        parts.append(
            "\nPrompt mode: BOOTSTRAP DISCOVERY\n"
            "- No scores are shown yet; use this phase to seed structurally different parts of the search space.\n"
            "- Do not start from a named canonical motif unless the observed archive already justifies it.\n"
            f"- Prefer a topology at least {explore_min_hamming} edge edits away from previously evaluated topologies.\n"
            "- Keep proposals simple, valid, and diverse."
        )
    else:
        parts.append(
            format_history_for_llm(
                archive,
                top_k=top_k,
                niche_k=niche_k,
                recent_k=recent_k,
            )
        )
        if phase == "exploit":
            best_block = build_current_best_block(archive)
            if best_block is not None:
                parts.append(best_block)
            parts.append(
                "\nPrompt mode: GUIDED EXPLOIT\n"
                "- Refine high-validation, low-gap families, but do not repeat the exact same topology.\n"
                "- Use both the global bests and the niche elites as anchors and look for small, plausible structural improvements.\n"
                "- Prefer motifs that should validate over longer horizons, not just spike on the short training run."
            )
        elif phase == "explore":
            parts.append(
                "\nPrompt mode: GUIDED EXPLORE\n"
                f"- Propose a topology at least {explore_min_hamming} edge edits away from the current top archive family.\n"
                "- Prioritize underrepresented niches, different sign patterns, and alternative feedback layouts.\n"
                "- Use the niche elite section to identify strong but structurally distinct families worth expanding.\n"
                "- Do not just mutate the current best by one small tweak."
            )
        else:
            best_block = build_current_best_block(archive)
            if best_block is not None:
                parts.append(best_block)
            parts.append(
                "\nPrompt mode: BLIND BENCHMARK\n"
                "- Use only the observed archive and the grammar constraints.\n"
                "- Do not rely on named canonical motifs or external literature priors.\n"
                "- Balance improvement with testing genuinely different structures and niche families."
            )

    parts.append(
        f"\nIteration {iteration + 1} of {n_iterations}.\n"
        "Propose the next topology to evaluate.\n"
        "Respond with ONLY the JSON object."
    )
    return "\n\n".join(parts)


def _dedupe_results(results: list[SearchResult]) -> list[SearchResult]:
    """Remove duplicate topology references while preserving order."""
    unique: list[SearchResult] = []
    seen = set()
    for result in results:
        if result.topology.edges in seen:
            continue
        seen.add(result.topology.edges)
        unique.append(result)
    return unique


def select_diversity_references(archive: Archive,
                                phase: str,
                                diversity_top_k: int = cfg.AGENTIC_DIVERSITY_TOP_K) -> list[SearchResult]:
    """Choose which archived results define the current diversity basin."""
    if len(archive) == 0:
        return []
    if phase == "bootstrap":
        return list(archive.results)
    if phase == "explore":
        refs = archive.niche_elites(diversity_top_k) + archive.results[-cfg.LLM_RECENT_K:]
        return _dedupe_results(refs)
    return []


def min_hamming_to_results(topology: Topology,
                           results: list[SearchResult]) -> Optional[int]:
    """Return the minimum Hamming distance to a set of archived topologies."""
    if not results:
        return None
    return min(hamming_distance(topology, result.topology) for result in results)


def build_diversity_retry_prompt(
    proposed: Topology,
    references: list[SearchResult],
    min_hamming: int,
    phase: str,
) -> str:
    """Ask the LLM for a more novel topology when the proposal is too close."""
    mode_label = "bootstrap" if phase == "bootstrap" else "exploration"
    lines = [
        f"Your previous topology was too close to the current {mode_label} reference set.",
        f"Reply with ONLY a JSON object for a topology that is at least {min_hamming} edge edits away from each reference below.",
        f"Rejected topology: {list(proposed.edges)}  [{proposed.to_label()}]",
        "Reference topologies:",
    ]
    for result in references[:cfg.AGENTIC_DIVERSITY_TOP_K]:
        lines.append(
            f"  - edges={list(result.topology.edges)}  "
            f"motif={result.topology.to_label()}"
        )
    lines.extend([
        "Required format:",
        f'{{"topology":{_json_topology_template()},"rationale":"brief text"}}',
    ])
    return "\n".join(lines)


def flatten_messages_for_native(messages: list[dict]) -> tuple[str, str]:
    """Convert OpenAI-style messages into a system prompt and transcript."""
    system_prompt = ""
    transcript = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system" and not system_prompt:
            system_prompt = str(content)
            continue
        label = "Assistant" if role == "assistant" else "User"
        transcript.append(f"{label}:\n{content}")
    return system_prompt, "\n\n".join(transcript).strip()


class Conversation:
    """Manage lightweight remembered exchanges between regenerated prompts."""

    def __init__(self,
                 system_prompt: str,
                 max_context_exchanges: int = cfg.LLM_MAX_CONTEXT_EXCHANGES):
        self.system = {"role": "system", "content": system_prompt}
        self.max_context_exchanges = max_context_exchanges
        self.exchanges: list[tuple[str, str]] = []

    def messages(self, user_msg: str) -> list[dict]:
        """Build the message list for the next LLM call."""
        msgs = [self.system]
        recent = self.exchanges[-self.max_context_exchanges:]
        for user_summary, assistant_summary in recent:
            msgs.append({"role": "user", "content": user_summary})
            msgs.append({"role": "assistant", "content": assistant_summary})
        msgs.append({"role": "user", "content": user_msg})
        return msgs

    def add_exchange(self, user_summary: str, assistant_msg: str) -> None:
        """Record a compact exchange summary."""
        if len(assistant_msg) > cfg.LLM_EXCHANGE_MAX_CHARS:
            assistant_msg = (
                assistant_msg[:cfg.LLM_EXCHANGE_MAX_CHARS] +
                "\n...[truncated]..."
            )
        self.exchanges.append((user_summary, assistant_msg))


def _strip_markdown_fences(text: str) -> str:
    """Strip enclosing markdown fences when present."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return stripped


def _extract_first_json_object(text: str) -> Optional[str]:
    """Extract the first balanced JSON object from free-form text."""
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escape = False
        for idx in range(start, len(text)):
            ch = text[idx]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == "\"":
                    in_string = False
                continue

            if ch == "\"":
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:idx + 1]
        start = text.find("{", start + 1)
    return None


def _topology_from_data(data: object) -> Optional[Topology]:
    """Convert parsed JSON-like data into a validated topology."""
    if isinstance(data, dict):
        edges_obj = data.get("topology")
        rationale = data.get("rationale", "(no rationale)")
    elif isinstance(data, (list, tuple)):
        edges_obj = data
        rationale = "(no rationale)"
    else:
        return None

    try:
        edges = tuple(int(e) for e in edges_obj)
    except Exception:
        return None

    if len(edges) != cfg.NUM_EDGE_SLOTS:
        print(f"  [LLM] Expected {cfg.NUM_EDGE_SLOTS} edges, got {len(edges)}")
        return None
    if not all(e in cfg.EDGE_VALUES for e in edges):
        print(f"  [LLM] Invalid edge values: {edges}")
        return None

    topo = Topology(edges=edges)
    if not topo.is_valid():
        print(f"  [LLM] Topology fails grammar constraints: {topo.to_label()}")
        return None

    print(f"  [LLM] Rationale: {rationale}")
    return topo


def _extract_topology_by_regex(text: str) -> Optional[Topology]:
    """Recover the topology list even if the surrounding JSON is malformed."""
    pattern = (
        r"\[\s*[012]\s*(?:,\s*[012]\s*){"
        + str(cfg.NUM_EDGE_SLOTS - 1)
        + r"}\]"
    )
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return None
    try:
        edges = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    topo = _topology_from_data(edges)
    if topo is not None:
        print("  [LLM] Recovered topology from non-JSON response.")
    return topo


def build_json_repair_prompt(raw_response: str) -> str:
    """Ask the model to re-emit a compact valid JSON object only."""
    compact = (raw_response or "").strip()
    if len(compact) > 1200:
        compact = compact[:1200] + "\n... (truncated)"
    return (
        "Your previous reply was not valid JSON for this task.\n"
        "Reply again with ONLY a compact JSON object, no markdown fences, no prose.\n"
        "Required format:\n"
        f'{{"topology":{_json_topology_template()},"rationale":"brief text"}}\n'
        f"The topology must contain exactly {cfg.NUM_EDGE_SLOTS} integers, each in {{0,1,2}}.\n\n"
        f"Previous reply:\n{compact}"
    )


def parse_llm_response(response_text: str) -> Optional[Topology]:
    """Parse the LLM's JSON response into a validated Topology."""
    text = _strip_markdown_fences((response_text or "").strip())
    if not text:
        print("  [LLM] Failed to parse response: empty response")
        return None

    candidates = [text]
    extracted = _extract_first_json_object(text)
    if extracted and extracted not in candidates:
        candidates.append(extracted)

    last_error = None
    for candidate in candidates:
        try:
            data = json.loads(candidate)
            topo = _topology_from_data(data)
            if topo is not None:
                return topo
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            last_error = exc

    topo = _extract_topology_by_regex(text)
    if topo is not None:
        return topo

    snippet = text[:240].replace("\n", "\\n")
    if last_error is not None:
        print(f"  [LLM] Failed to parse response: {last_error}")
    else:
        print("  [LLM] Failed to parse response.")
    print(f"  [LLM] Raw reply snippet: {snippet}")
    return None


# ── Main agentic search ──────────────────────────────────

def run_agentic_search(
    llm_call_fn: Callable[[list[dict]], str],
    n_iterations: int = cfg.AGENTIC_SEARCH_N,
    max_evals_inner: int = cfg.INNER_MAX_EVALS,
    n_workers: int = cfg.INNER_NUM_WORKERS,
    agentic_mode: str = cfg.AGENTIC_MODE,
    bootstrap_iters: int = cfg.AGENTIC_BOOTSTRAP_ITERS,
    explore_min_hamming: int = cfg.AGENTIC_EXPLORE_MIN_HAMMING,
    seed_archive: Optional[Archive] = None,
) -> Archive:
    """
    Run the LLM-guided agentic topology search.

    Args:
        llm_call_fn:     callable(messages) → response text
        n_iterations:    Number of LLM proposals to evaluate.
        max_evals_inner: Budget for fcmaes inner optimisation per topology.
        n_workers:       Number of fcmaes parallel workers.
        seed_archive:    Optional archive from a prior search to warm-start.

    Returns:
        Archive with all evaluation results.
    """
    archive = seed_archive if seed_archive is not None else Archive()
    conv = Conversation(build_system_prompt(agentic_mode))
    consecutive_failures = 0
    total_t0 = time.perf_counter()

    for i in range(n_iterations):
        print(f"\n{'='*60}")
        print(f"[Agentic {i+1}/{n_iterations}]")
        phase = select_agentic_phase(agentic_mode, len(archive), bootstrap_iters)
        print(f"  Phase: {phase}  mode={agentic_mode}")

        # ── Build prompt with history ─────────────────────
        user_prompt = build_user_message(
            archive,
            iteration=i,
            n_iterations=n_iterations,
            agentic_mode=agentic_mode,
            phase=phase,
            bootstrap_iters=bootstrap_iters,
            explore_min_hamming=explore_min_hamming,
            top_k=cfg.LLM_TOP_K,
            niche_k=cfg.LLM_NICHE_K,
            recent_k=cfg.LLM_RECENT_K,
        )
        messages = conv.messages(user_prompt)

        # ── Query LLM ────────────────────────────────────
        try:
            response = llm_call_fn(messages)
        except Exception as e:
            print(f"  [LLM] API error: {e}")
            conv.add_exchange(
                f"Iteration {i+1}: propose the next topology.",
                f"(API error: {e})",
            )
            consecutive_failures += 1
            if consecutive_failures > cfg.LLM_MAX_CONSECUTIVE_FAILURES:
                print("  Too many consecutive LLM failures. Stopping.")
                break
            continue

        # ── Parse proposal ────────────────────────────────
        topology = parse_llm_response(response)
        if topology is None:
            print("  [LLM] Requesting compact JSON reformat...")
            repair_prompt = build_json_repair_prompt(response)
            try:
                repair_messages = conv.messages(repair_prompt)
                repair_response = llm_call_fn(repair_messages)
                topology = parse_llm_response(repair_response)
            except Exception as e:
                print(f"  [LLM] Repair API error: {e}")

        if topology is None:
            conv.add_exchange(
                f"Iteration {i+1}: propose the next topology.",
                "(invalid or empty reply; output strict compact JSON only)",
            )
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
            conv.add_exchange(
                f"Tried topology {list(topology.edges)}.",
                f"(duplicate topology skipped: {topology.to_label()})",
            )
            continue

        diversity_refs = select_diversity_references(archive, phase)
        diversity_distance = min_hamming_to_results(topology, diversity_refs)
        if diversity_distance is not None:
            print(f"  Diversity distance: {diversity_distance}")
        if diversity_distance is not None and diversity_distance < explore_min_hamming:
            print(
                "  [LLM] Proposal too close to the current reference set "
                f"(distance={diversity_distance}, required>={explore_min_hamming})."
            )
            diversity_prompt = build_diversity_retry_prompt(
                topology,
                diversity_refs,
                explore_min_hamming,
                phase,
            )
            try:
                diversity_messages = conv.messages(diversity_prompt)
                diversity_response = llm_call_fn(diversity_messages)
                topology = parse_llm_response(diversity_response)
            except Exception as e:
                topology = None
                print(f"  [LLM] Diversity retry API error: {e}")

            if topology is None:
                conv.add_exchange(
                    f"Iteration {i+1}: propose the next topology.",
                    "(failed to satisfy diversity constraint; output a novel topology in strict compact JSON)",
                )
                consecutive_failures += 1
                if consecutive_failures > cfg.LLM_MAX_CONSECUTIVE_FAILURES:
                    print("  Too many invalid proposals. Stopping.")
                    break
                continue

            print(f"  Diversity retry proposed: {topology.to_label()}")
            if archive.already_evaluated(topology):
                print("  Already evaluated after diversity retry — skipping.")
                conv.add_exchange(
                    f"Tried topology {list(topology.edges)}.",
                    f"(duplicate topology skipped after diversity retry: {topology.to_label()})",
                )
                continue

            diversity_distance = min_hamming_to_results(topology, diversity_refs)
            if diversity_distance is not None:
                print(f"  Diversity distance: {diversity_distance}")
            if diversity_distance is not None and diversity_distance < explore_min_hamming:
                print("  [LLM] Diversity retry still too close — skipping.")
                conv.add_exchange(
                    f"Iteration {i+1}: propose the next topology.",
                    (
                        f"(proposal remained within {explore_min_hamming} edge edits "
                        "of the protected reference set; try a more novel structure)"
                    ),
                )
                consecutive_failures += 1
                if consecutive_failures > cfg.LLM_MAX_CONSECUTIVE_FAILURES:
                    print("  Too many invalid proposals. Stopping.")
                    break
                continue

        # ── Inner optimisation ────────────────────────────
        result = optimize_topology(
            topology, max_evals=max_evals_inner, n_workers=n_workers,
            verbose=True,
        )
        archive.add(SearchResult(
            topology=topology,
            score=result["best_score"],
            params=result["best_params"],
            iteration=i,
            wall_time=result["wall_time"],
            strategy="agentic",
            train_score=result["train_score"],
            train_raw_score=result["train_raw_score"],
            train_period=result["train_period"],
            train_full_score=result["train_full_score"],
            train_knockout_score=result["train_knockout_score"],
            train_knockout_pass_rate=result["train_knockout_pass_rate"],
            train_knockdown_score=result["train_knockdown_score"],
            train_param_perturb_score=result["train_param_perturb_score"],
            validation_score=result["validation_score"],
            validation_raw_score=result["validation_raw_score"],
            validation_period=result["validation_period"],
            validation_full_score=result["validation_full_score"],
            validation_knockout_score=result["validation_knockout_score"],
            validation_knockout_pass_rate=result["validation_knockout_pass_rate"],
            validation_knockdown_score=result["validation_knockdown_score"],
            validation_param_perturb_score=result["validation_param_perturb_score"],
            generalization_gap=result["generalization_gap"],
        ))
        detail = ""
        if result["validation_knockout_score"] is not None:
            detail = (
                f"  full={result['validation_full_score']:.4f}"
                f"  ko={result['validation_knockout_score']:.4f}"
                f"  pass={result['validation_knockout_pass_rate']:.2f}"
            )
        if result["validation_knockdown_score"] is not None:
            detail += f"  kd={result['validation_knockdown_score']:.4f}"
        if result["validation_param_perturb_score"] is not None:
            detail += f"  pert={result['validation_param_perturb_score']:.4f}"
        print(
            f"  → rank = {result['best_score']:.4f}  "
            f"(train={result['train_score']:.4f}  "
            f"val={result['validation_score']:.4f}  "
            f"gap={result['generalization_gap']:.4f}  "
            f"period={result['validation_period']:.1f}"
            f"{detail})"
        )
        print(f"\n{archive.summary()}")
        aux = ""
        if result["validation_knockout_score"] is not None:
            aux = (
                f" full={result['validation_full_score']:.4f}"
                f" ko={result['validation_knockout_score']:.4f}"
                f" pass={result['validation_knockout_pass_rate']:.2f}"
            )
        if result["validation_knockdown_score"] is not None:
            aux += f" kd={result['validation_knockdown_score']:.4f}"
        if result["validation_param_perturb_score"] is not None:
            aux += f" pert={result['validation_param_perturb_score']:.4f}"
        conv.add_exchange(
            f"Tried topology {list(topology.edges)}.",
            (
                f"Experiment #{i+1}: rank={result['best_score']:.4f} "
                f"val={result['validation_score']:.4f} "
                f"train={result['train_score']:.4f} "
                f"gap={result['generalization_gap']:.4f} "
                f"period={result['validation_period']:.1f} "
                f"{aux} "
                f"motif={topology.to_label()} active_edges={topology.num_active_edges}"
            ),
        )

    total_wall = time.perf_counter() - total_t0
    print(f"\nAgentic search done in {total_wall:.1f}s")
    print(archive.summary())
    return archive


# ── LLM backend helpers ──────────────────────────────────

LLMCallFn = Callable[[list[dict]], str]


def is_local_base_url(base_url: Optional[str]) -> bool:
    """Return True for local/default OpenAI-compatible endpoints."""
    if not base_url:
        return True
    base = base_url.lower()
    return "127.0.0.1" in base or "localhost" in base


def pick_llm_backend(model_name: Optional[str],
                     base_url: Optional[str],
                     requested_backend: str = "auto") -> str:
    """Choose between OpenAI-compatible and native provider SDKs."""
    if requested_backend != "auto":
        return requested_backend

    model = (model_name or "").lower()
    if "claude" in model and is_local_base_url(base_url):
        return "claude"
    if "gemini" in model and is_local_base_url(base_url):
        return "gemini"
    if "minimax" in model and is_local_base_url(base_url):
        return "minimax"
    return "openai"


def resolve_api_key(base_url: Optional[str]) -> str:
    """Resolve an API key for OpenAI-compatible endpoints."""
    if os.environ.get("OPENAI_API_KEY"):
        return os.environ["OPENAI_API_KEY"]
    if base_url and "anthropic.com" in base_url and os.environ.get("ANTHROPIC_API_KEY"):
        return os.environ["ANTHROPIC_API_KEY"]
    if base_url and "minimax.io" in base_url and os.environ.get("MINIMAX_API_KEY"):
        return os.environ["MINIMAX_API_KEY"]
    return "dummy"


def pick_model_id(client, requested_model: Optional[str] = None) -> str:
    """Resolve model id, preferring an explicit CLI override."""
    if requested_model:
        return requested_model
    try:
        models = client.models.list()
        return models.data[0].id if models.data else "default"
    except Exception as exc:
        print(f"  [WARN] Could not list models ({exc}); using provider default")
        return "default"


def _log_anthropic_usage(response) -> None:
    """Print token usage for Anthropic-compatible responses."""
    usage = getattr(response, "usage", None)
    if not usage:
        return
    input_tokens = getattr(usage, "input_tokens", 0)
    output_tokens = getattr(usage, "output_tokens", 0)
    total = input_tokens + output_tokens
    print(
        "  [USAGE] "
        f"Total: {total} | Prompt: {input_tokens} | Output: {output_tokens} "
        "(Thinking included in Output)"
    )


def _log_openai_usage(response) -> None:
    """Print token usage for OpenAI-compatible responses."""
    usage = getattr(response, "usage", None)
    if not usage:
        return
    total = getattr(usage, "total_tokens", 0)
    prompt = getattr(usage, "prompt_tokens", 0)
    completion = getattr(usage, "completion_tokens", 0)
    details = getattr(usage, "completion_tokens_details", None)
    reasoning = getattr(details, "reasoning_tokens", 0) if details else 0
    print(
        "  [USAGE] "
        f"Total: {total} | Prompt: {prompt} | Output: {completion} "
        f"(Thinking: {reasoning})"
    )


def _log_gemini_usage(response) -> None:
    """Print token usage for Gemini responses."""
    usage = getattr(response, "usage_metadata", None)
    if not usage:
        return
    total = getattr(usage, "total_token_count", 0)
    prompt = getattr(usage, "prompt_token_count", 0)
    output = getattr(usage, "candidates_token_count", 0)
    thinking = getattr(usage, "thoughts_token_count", 0)
    print(
        "  [USAGE] "
        f"Total: {total} | Prompt: {prompt} | Output: {output} "
        f"(Thinking: {thinking})"
    )


def extract_anthropic_text(response) -> str:
    """Extract plain text from an Anthropic-style response."""
    blocks = getattr(response, "content", None) or []
    text_parts = []
    for block in blocks:
        if getattr(block, "type", "") == "text" and getattr(block, "text", ""):
            text_parts.append(block.text)
    if text_parts:
        return "\n".join(text_parts).strip()
    for block in reversed(blocks):
        block_text = getattr(block, "text", "")
        if block_text:
            return block_text.strip()
    return ""


MINIMAX_ANTHROPIC_BASE_URL = "https://api.minimax.io/anthropic"
MINIMAX_OPENAI_BASE_URL = "https://api.minimax.io/v1"


def make_claude_llm_fn(
    model: Optional[str] = None,
    temperature: float = cfg.LLM_TEMPERATURE,
    max_tokens: int = cfg.LLM_MAX_TOKENS,
    thinking_effort: str = cfg.LLM_THINKING_EFFORT,
) -> Tuple[LLMCallFn, str]:
    """Create an Anthropic Claude call function."""
    try:
        import anthropic  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Claude support requires the optional 'anthropic' package."
        ) from exc

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set.")

    model_id = model or cfg.LLM_MODEL
    client = anthropic.Anthropic(api_key=api_key)

    def call(messages: list[dict]) -> str:
        system_prompt, user_prompt = flatten_messages_for_native(messages)
        request = dict(
            model=model_id,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=temperature,
        )
        try:
            try:
                if thinking_effort == "high":
                    response = client.messages.create(
                        **request,
                        output_config={"effort": "high"},
                    )
                else:
                    response = client.messages.create(**request)
            except Exception as exc:
                if any(token in str(exc).lower() for token in ("thinking", "adaptive", "effort")):
                    response = client.messages.create(**request)
                else:
                    raise
            _log_anthropic_usage(response)
            return extract_anthropic_text(response)
        except Exception as exc:
            print(f"  [LLM ERROR] {exc}")
            return ""

    return call, model_id


def make_minimax_llm_fn(
    model: Optional[str] = None,
    temperature: float = cfg.LLM_TEMPERATURE,
    max_tokens: int = cfg.LLM_MAX_TOKENS,
    thinking_effort: str = cfg.LLM_THINKING_EFFORT,
    base_url: Optional[str] = None,
) -> Tuple[LLMCallFn, str]:
    """Create a MiniMax call function via Anthropic or OpenAI compatibility."""
    api_key = os.environ.get("MINIMAX_API_KEY")
    if not api_key:
        raise RuntimeError("MINIMAX_API_KEY is not set.")
    if not model:
        raise RuntimeError("MiniMax backend requires an explicit model id via --model.")

    model_id = model
    resolved_base_url = base_url
    if not resolved_base_url:
        resolved_base_url = MINIMAX_ANTHROPIC_BASE_URL

    endpoint_kind = "openai" if "/v1" in resolved_base_url and "/anthropic" not in resolved_base_url else "anthropic"

    if endpoint_kind == "openai":
        from openai import OpenAI

        client = OpenAI(base_url=resolved_base_url, api_key=api_key)

        def call(messages: list[dict]) -> str:
            system_prompt, user_prompt = flatten_messages_for_native(messages)
            try:
                response = client.chat.completions.create(
                    model=model_id,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                _log_openai_usage(response)
                content = response.choices[0].message.content or ""
                if not content:
                    finish_reason = getattr(response.choices[0], "finish_reason", "")
                    print(f"  [LLM WARN] MiniMax OpenAI-compatible reply was empty (finish_reason={finish_reason}).")
                return content
            except Exception as exc:
                print(f"  [LLM ERROR] {exc}")
                return ""

        return call, model_id

    try:
        import anthropic  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "MiniMax Anthropic-compatible support requires the optional 'anthropic' package."
        ) from exc

    client = anthropic.Anthropic(
        api_key=api_key,
        base_url=resolved_base_url,
    )

    def call(messages: list[dict]) -> str:
        system_prompt, user_prompt = flatten_messages_for_native(messages)
        request = dict(
            model=model_id,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=temperature,
        )
        try:
            try:
                if thinking_effort == "high":
                    response = client.messages.create(
                        **request,
                        output_config={"effort": "high"},
                    )
                else:
                    response = client.messages.create(**request)
            except Exception as exc:
                if any(token in str(exc).lower() for token in ("thinking", "adaptive", "effort", "unrecognized")):
                    response = client.messages.create(**request)
                else:
                    raise
            _log_anthropic_usage(response)
            text = extract_anthropic_text(response)
            if not text:
                block_types = [getattr(block, "type", "?") for block in getattr(response, "content", []) or []]
                stop_reason = getattr(response, "stop_reason", "")
                print(
                    "  [LLM WARN] MiniMax Anthropic-compatible reply had no text blocks "
                    f"(stop_reason={stop_reason}, block_types={block_types})."
                )
            return text
        except Exception as exc:
            print(f"  [LLM ERROR] {exc}")
            return ""

    return call, model_id


def make_gemini_llm_fn(
    model: Optional[str] = None,
    temperature: float = cfg.LLM_TEMPERATURE,
    max_tokens: int = cfg.LLM_MAX_TOKENS,
    thinking_effort: str = cfg.LLM_THINKING_EFFORT,
) -> Tuple[LLMCallFn, str]:
    """Create a Google Gemini call function."""
    try:
        from google import genai  # type: ignore
        from google.genai import types  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Gemini support requires the optional 'google-genai' package."
        ) from exc

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY is not set.")
    if not model:
        raise RuntimeError("Gemini backend requires an explicit model id via --model.")

    model_id = model
    client = genai.Client(api_key=api_key)

    def call(messages: list[dict]) -> str:
        system_prompt, user_prompt = flatten_messages_for_native(messages)
        config_kwargs = dict(
            system_instruction=system_prompt,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        try:
            try:
                if thinking_effort == "high":
                    config = types.GenerateContentConfig(
                        **config_kwargs,
                        thinking_config=types.ThinkingConfig(thinking_level="HIGH"),
                    )
                else:
                    config = types.GenerateContentConfig(**config_kwargs)
            except Exception:
                config = types.GenerateContentConfig(**config_kwargs)
            response = client.models.generate_content(
                model=model_id,
                contents=user_prompt,
                config=config,
            )
            _log_gemini_usage(response)
            return (getattr(response, "text", "") or "").strip()
        except Exception as exc:
            print(f"  [LLM ERROR] {exc}")
            return ""

    return call, model_id


def make_openai_compatible_llm_fn(
    base_url: Optional[str] = cfg.LLM_BASE_URL,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = cfg.LLM_TEMPERATURE,
    max_tokens: int = cfg.LLM_MAX_TOKENS,
) -> Tuple[LLMCallFn, str]:
    """Create an LLM call function for any OpenAI-compatible API."""
    from openai import OpenAI

    client_kwargs = {"api_key": api_key or resolve_api_key(base_url)}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = OpenAI(**client_kwargs)
    model_id = pick_model_id(client, requested_model=model)

    def call(messages: list[dict]) -> str:
        try:
            response = client.chat.completions.create(
                model=model_id,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=messages,
            )
            _log_openai_usage(response)
            return response.choices[0].message.content or ""
        except Exception as exc:
            print(f"  [LLM ERROR] {exc}")
            return ""

    return call, model_id


def make_llm_call_fn(
    backend: str = cfg.LLM_BACKEND,
    base_url: Optional[str] = cfg.LLM_BASE_URL,
    model: Optional[str] = cfg.LLM_MODEL,
    temperature: float = cfg.LLM_TEMPERATURE,
    max_tokens: int = cfg.LLM_MAX_TOKENS,
    thinking_effort: str = cfg.LLM_THINKING_EFFORT,
    api_key: Optional[str] = None,
) -> Tuple[LLMCallFn, str, str]:
    """Create a configured LLM caller and report the resolved backend/model."""
    resolved_backend = pick_llm_backend(model, base_url, requested_backend=backend)

    if resolved_backend == "claude":
        call_fn, model_id = make_claude_llm_fn(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            thinking_effort=thinking_effort,
        )
        return call_fn, resolved_backend, model_id

    if resolved_backend == "gemini":
        call_fn, model_id = make_gemini_llm_fn(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            thinking_effort=thinking_effort,
        )
        return call_fn, resolved_backend, model_id

    if resolved_backend == "minimax":
        call_fn, model_id = make_minimax_llm_fn(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            thinking_effort=thinking_effort,
            base_url=base_url,
        )
        return call_fn, resolved_backend, model_id

    call_fn, model_id = make_openai_compatible_llm_fn(
        base_url=base_url,
        model=model,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return call_fn, resolved_backend, model_id


def make_anthropic_llm_fn() -> LLMCallFn:
    """Backward-compatible Anthropic wrapper."""
    call_fn, _ = make_claude_llm_fn(
        model=cfg.LLM_MODEL,
        temperature=cfg.LLM_TEMPERATURE,
        max_tokens=cfg.LLM_MAX_TOKENS,
        thinking_effort=cfg.LLM_THINKING_EFFORT,
    )
    return call_fn
