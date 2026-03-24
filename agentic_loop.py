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

SYSTEM_PROMPT = f"""\
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

Given the history of previously tested topologies and their measured
performance, propose the NEXT topology to test.

Each experiment reports:
  - rank_score = validation_score - {cfg.GENERALIZATION_GAP_PENALTY:.1f} * |train_score - validation_score|
  - train_score: short-horizon score used during parameter optimisation
  - validation_score: longer-horizon holdout score on different seeds
  - gap: |train_score - validation_score|
  - period: approximate validation oscillation period

Use validation_score and small gap as the main learning signal.
Do not chase train_score alone.

Respond with ONLY a JSON object:
{{
  "topology": [int, int, int, int, int, int, int, int, int],
  "rationale": "Brief explanation of why this topology might oscillate."
}}
"""


# ── Helper functions ──────────────────────────────────────

def _fmt_metric(value: Optional[float], digits: int = 4, default: str = "n/a") -> str:
    """Format optional float metrics for logs and prompts."""
    if value is None:
        return default
    return f"{value:.{digits}f}"


def format_history_for_llm(archive: Archive,
                           top_k: int = cfg.LLM_TOP_K,
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
                f"edges={list(result.topology.edges)}  "
                f"active={result.topology.num_active_edges}  "
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
                f"edges={list(result.topology.edges)}  "
                f"motif={result.topology.to_label()}"
            )

    return "\n".join(lines)


def build_user_message(archive: Archive,
                       iteration: int,
                       n_iterations: int,
                       top_k: int = cfg.LLM_TOP_K,
                       recent_k: int = cfg.LLM_RECENT_K) -> str:
    """Rebuild the full prompt for the current turn from archive state."""
    parts = []

    if len(archive) == 0:
        parts.append(
            "This is the FIRST experiment. Start with a strong oscillator motif "
            "such as the repressilator (A⊣B, B⊣C, C⊣A) or a Goodwin-style loop."
        )
    else:
        parts.append(format_history_for_llm(archive, top_k=top_k, recent_k=recent_k))
        best = archive.best
        if best is not None:
            parts.append(
                "\nCurrent best topology:\n"
                f"  rank={best.score:.4f}\n"
                f"  validation_score={_fmt_metric(best.validation_score)}\n"
                f"  train_score={_fmt_metric(best.train_score)}\n"
                f"  gap={_fmt_metric(best.generalization_gap)}\n"
                f"  period={_fmt_metric(best.validation_period, digits=1)}\n"
                f"  edges={list(best.topology.edges)}\n"
                f"  motif={best.topology.to_label()}"
            )
        parts.append(
            "\nPrompt mode: EXPLOIT STRUCTURE WITH DIVERSITY\n"
            "- Use high-validation, low-gap motifs as anchors, but do not keep proposing the exact same structure.\n"
            "- Prefer negative-feedback loops, delayed negative feedback, or mixed activation/inhibition motifs.\n"
            "- Prefer motifs that should validate over longer horizons, not just spike on the short training run.\n"
            "- Keep proposals inside the 3-gene grammar and respond with only the required JSON object."
        )

    parts.append(
        f"\nIteration {iteration + 1} of {n_iterations}.\n"
        "Propose the next topology to evaluate.\n"
        "Respond with ONLY the JSON object."
    )
    return "\n\n".join(parts)


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

    print(f"  [LLM] Rationale: {rationale}")
    return topo


def _extract_topology_by_regex(text: str) -> Optional[Topology]:
    """Recover the topology list even if the surrounding JSON is malformed."""
    match = re.search(r"\[\s*[012]\s*(?:,\s*[012]\s*){8}\]", text, re.DOTALL)
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
        '{"topology":[0,0,0,0,0,0,0,0,0],"rationale":"brief text"}\n'
        "The topology must contain exactly 9 integers, each in {0,1,2}.\n\n"
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
    n_retries: int = cfg.INNER_NUM_RETRIES,
    seed_archive: Optional[Archive] = None,
) -> Archive:
    """
    Run the LLM-guided agentic topology search.

    Args:
        llm_call_fn:     callable(messages) → response text
        n_iterations:    Number of LLM proposals to evaluate.
        max_evals_inner: Budget for fcmaes inner optimisation per topology.
        n_retries:       Number of fcmaes restarts per topology.
        seed_archive:    Optional archive from a prior search to warm-start.

    Returns:
        Archive with all evaluation results.
    """
    archive = seed_archive if seed_archive is not None else Archive()
    conv = Conversation(SYSTEM_PROMPT)
    consecutive_failures = 0
    total_t0 = time.perf_counter()

    for i in range(n_iterations):
        print(f"\n{'='*60}")
        print(f"[Agentic {i+1}/{n_iterations}]")

        # ── Build prompt with history ─────────────────────
        user_prompt = build_user_message(
            archive,
            iteration=i,
            n_iterations=n_iterations,
            top_k=cfg.LLM_TOP_K,
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
            train_score=result["train_score"],
            train_raw_score=result["train_raw_score"],
            train_period=result["train_period"],
            validation_score=result["validation_score"],
            validation_raw_score=result["validation_raw_score"],
            validation_period=result["validation_period"],
            generalization_gap=result["generalization_gap"],
        ))
        print(
            f"  → rank = {result['best_score']:.4f}  "
            f"(train={result['train_score']:.4f}  "
            f"val={result['validation_score']:.4f}  "
            f"gap={result['generalization_gap']:.4f}  "
            f"period={result['validation_period']:.1f})"
        )
        print(f"\n{archive.summary()}")
        conv.add_exchange(
            f"Tried topology {list(topology.edges)}.",
            (
                f"Experiment #{i+1}: rank={result['best_score']:.4f} "
                f"val={result['validation_score']:.4f} "
                f"train={result['train_score']:.4f} "
                f"gap={result['generalization_gap']:.4f} "
                f"period={result['validation_period']:.1f} "
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
