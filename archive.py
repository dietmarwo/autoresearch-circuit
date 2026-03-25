"""
Archive — persistent storage and analysis of search results.

Keeps all evaluated (topology, score, params) triples plus metadata.
Also preserves one best representative per structural niche so the
outer loop can reason about diverse high-performing families.
Supports serialisation to JSON and pickle.
"""

import json
import pickle
import time
from dataclasses import dataclass, field
from itertools import combinations, permutations
from typing import List, Optional

import numpy as np

from grammar import Topology
import config as cfg


def topology_core_flags(topology: Topology) -> tuple[str, ...]:
    """Detect coarse structural motif flags used for niche grouping."""
    edges = topology.edges
    flags = set()
    edge_lookup = {pair: idx for idx, pair in enumerate(cfg.EDGE_INDEX_MAP)}

    for genes in combinations(range(cfg.NUM_GENES), 3):
        for cycle in permutations(genes, 3):
            values = tuple(
                edges[edge_lookup[(src, tgt)]]
                for src, tgt in ((cycle[0], cycle[1]), (cycle[1], cycle[2]), (cycle[2], cycle[0]))
            )
            if values == (2, 2, 2):
                flags.add("repressilator")
            elif values == (1, 1, 2):
                flags.add("goodwin")
            elif values == (1, 1, 1):
                flags.add("positive_cycle")
            elif all(value != 0 for value in values):
                flags.add("mixed_cycle")

    for left in range(cfg.NUM_GENES):
        for right in range(left + 1, cfg.NUM_GENES):
            if (
                edges[edge_lookup[(left, right)]] == 2
                and edges[edge_lookup[(right, left)]] == 2
            ):
                flags.add("toggle")
                break
        if "toggle" in flags:
            break

    if not flags:
        flags.add("other")

    return tuple(sorted(flags))


def topology_niche_parts(topology: Topology) -> dict:
    """Return the structural descriptor used by the niche archive."""
    activating_edges = sum(edge == 1 for edge in topology.edges)
    inhibiting_edges = sum(edge == 2 for edge in topology.edges)
    self_edges = sum(topology.edges[idx] != 0 for idx in range(cfg.NUM_GENES))
    core_flags = topology_core_flags(topology)
    return {
        "active_edges": topology.num_active_edges,
        "activating_edges": activating_edges,
        "inhibiting_edges": inhibiting_edges,
        "self_edges": self_edges,
        "core_flags": list(core_flags),
    }


def topology_niche_key(topology: Topology) -> str:
    """Compact human-readable niche identifier."""
    parts = topology_niche_parts(topology)
    core = "+".join(parts["core_flags"])
    return (
        f"E{parts['active_edges']}-"
        f"A{parts['activating_edges']}-"
        f"I{parts['inhibiting_edges']}-"
        f"S{parts['self_edges']}-"
        f"C{core}"
    )


@dataclass
class SearchResult:
    """One evaluated topology, ranked by validation-aware score."""
    topology: Topology
    score: float
    params: Optional[np.ndarray] = None
    iteration: int = 0
    wall_time: float = 0.0
    strategy: str = ""
    train_score: Optional[float] = None
    train_raw_score: Optional[float] = None
    train_period: Optional[float] = None
    train_full_score: Optional[float] = None
    train_knockout_score: Optional[float] = None
    train_knockout_pass_rate: Optional[float] = None
    train_knockdown_score: Optional[float] = None
    train_param_perturb_score: Optional[float] = None
    validation_score: Optional[float] = None
    validation_raw_score: Optional[float] = None
    validation_period: Optional[float] = None
    validation_full_score: Optional[float] = None
    validation_knockout_score: Optional[float] = None
    validation_knockout_pass_rate: Optional[float] = None
    validation_knockdown_score: Optional[float] = None
    validation_param_perturb_score: Optional[float] = None
    generalization_gap: Optional[float] = None
    niche_key: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

    def __setstate__(self, state):
        """Backfill newly added fields when loading older pickles."""
        self.__dict__.update(state)
        for name in (
            "train_score",
            "train_raw_score",
            "train_period",
            "train_full_score",
            "train_knockout_score",
            "train_knockout_pass_rate",
            "train_knockdown_score",
            "train_param_perturb_score",
            "validation_score",
            "validation_raw_score",
            "validation_period",
            "validation_full_score",
            "validation_knockout_score",
            "validation_knockout_pass_rate",
            "validation_knockdown_score",
            "validation_param_perturb_score",
            "generalization_gap",
            "niche_key",
        ):
            self.__dict__.setdefault(name, None)


class Archive:
    """Stores all evaluated topologies and their scores."""

    def __init__(self):
        self.results: List[SearchResult] = []

    def _ensure_result_metadata(self, result: SearchResult) -> SearchResult:
        """Backfill archive-derived metadata for older results."""
        if result.niche_key is None:
            result.niche_key = topology_niche_key(result.topology)
        return result

    def _ensure_all_metadata(self) -> None:
        """Backfill metadata on all results in place."""
        for result in self.results:
            self._ensure_result_metadata(result)

    # ── Core ops ──────────────────────────────────────────

    def add(self, result: SearchResult):
        self.results.append(self._ensure_result_metadata(result))

    def __len__(self) -> int:
        return len(self.results)

    @property
    def best(self) -> Optional[SearchResult]:
        if not self.results:
            return None
        self._ensure_all_metadata()
        return max(self.results, key=lambda r: r.score)

    def top_k(self, k: int = 5) -> List[SearchResult]:
        self._ensure_all_metadata()
        return sorted(self.results, key=lambda r: r.score, reverse=True)[:k]

    def already_evaluated(self, topology: Topology) -> bool:
        return any(r.topology.edges == topology.edges for r in self.results)

    def niche_elite_map(self) -> dict[str, SearchResult]:
        """Return the best result currently stored for each niche."""
        self._ensure_all_metadata()
        elites: dict[str, SearchResult] = {}
        for result in self.results:
            current = elites.get(result.niche_key)
            if current is None or result.score > current.score:
                elites[result.niche_key] = result
        return elites

    def niche_elites(self, k: Optional[int] = None) -> List[SearchResult]:
        """Return the best representative from each structural niche."""
        elites = sorted(
            self.niche_elite_map().values(),
            key=lambda result: result.score,
            reverse=True,
        )
        return elites if k is None else elites[:k]

    def niche_counts(self) -> dict[str, int]:
        """Count how many archived results fall into each niche."""
        self._ensure_all_metadata()
        counts: dict[str, int] = {}
        for result in self.results:
            counts[result.niche_key] = counts.get(result.niche_key, 0) + 1
        return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))

    # ── Statistics ────────────────────────────────────────

    def score_stats(self) -> dict:
        if not self.results:
            return {"n": 0}
        self._ensure_all_metadata()
        scores = [r.score for r in self.results]
        stats = {
            "n": len(scores),
            "best": max(scores),
            "mean": float(np.mean(scores)),
            "median": float(np.median(scores)),
            "std": float(np.std(scores)),
            "nonzero": sum(1 for s in scores if s > 0.01),
            "niches": len(self.niche_elite_map()),
        }
        validation_scores = [r.validation_score for r in self.results
                             if r.validation_score is not None]
        if validation_scores:
            stats["best_validation"] = max(validation_scores)
            stats["mean_validation"] = float(np.mean(validation_scores))
        train_scores = [r.train_score for r in self.results
                        if r.train_score is not None]
        if train_scores:
            stats["best_train"] = max(train_scores)
            stats["mean_train"] = float(np.mean(train_scores))
        gaps = [r.generalization_gap for r in self.results
                if r.generalization_gap is not None]
        if gaps:
            stats["mean_gap"] = float(np.mean(gaps))
        return stats

    # ── Display ───────────────────────────────────────────

    def summary(self, top: int = 5, niche_top: int = 3) -> str:
        if not self.results:
            return "Archive empty."
        self._ensure_all_metadata()
        stats = self.score_stats()
        lines = [
            f"Evaluated {stats['n']} topologies  "
            f"(best_rank={stats['best']:.4f}  mean_rank={stats['mean']:.4f}  "
            f"nonzero={stats['nonzero']})",
        ]
        lines.append(
            f"Structural niches: {stats['niches']}  "
            "elite archive keeps one best representative per niche"
        )
        if "best_validation" in stats:
            lines.append(
                f"Validation: best={stats['best_validation']:.4f}  "
                f"mean={stats['mean_validation']:.4f}"
            )
        if "mean_gap" in stats:
            lines.append(f"Generalization gap: mean={stats['mean_gap']:.4f}")
        for i, r in enumerate(self.top_k(top)):
            parts = [f"  #{i+1}: rank={r.score:.4f}"]
            if r.validation_score is not None:
                parts.append(f"val={r.validation_score:.4f}")
            if r.train_score is not None:
                parts.append(f"train={r.train_score:.4f}")
            if r.generalization_gap is not None:
                parts.append(f"gap={r.generalization_gap:.4f}")
            if r.validation_period is not None and r.validation_period > 0.0:
                parts.append(f"period={r.validation_period:.1f}")
            if r.validation_knockout_score is not None:
                parts.append(f"full={r.validation_full_score:.4f}")
                parts.append(f"ko={r.validation_knockout_score:.4f}")
                parts.append(f"pass={r.validation_knockout_pass_rate:.2f}")
            if r.validation_knockdown_score is not None:
                parts.append(f"kd={r.validation_knockdown_score:.4f}")
            if r.validation_param_perturb_score is not None:
                parts.append(f"pert={r.validation_param_perturb_score:.4f}")
            parts.append(f"niche={r.niche_key}")
            parts.append(f"edges={r.topology.num_active_edges}")
            parts.append(f"[{r.topology.to_label()}]")
            lines.append("  ".join(parts))
        niche_results = self.niche_elites(niche_top)
        if niche_results:
            lines.append("\nBest niche elites:")
            for i, r in enumerate(niche_results):
                parts = [f"  [N{i+1}] niche={r.niche_key}"]
                parts.append(f"rank={r.score:.4f}")
                if r.validation_score is not None:
                    parts.append(f"val={r.validation_score:.4f}")
                if r.validation_knockout_score is not None:
                    parts.append(f"ko={r.validation_knockout_score:.4f}")
                    parts.append(f"pass={r.validation_knockout_pass_rate:.2f}")
                if r.validation_knockdown_score is not None:
                    parts.append(f"kd={r.validation_knockdown_score:.4f}")
                if r.validation_param_perturb_score is not None:
                    parts.append(f"pert={r.validation_param_perturb_score:.4f}")
                if r.generalization_gap is not None:
                    parts.append(f"gap={r.generalization_gap:.4f}")
                parts.append(f"[{r.topology.to_label()}]")
                lines.append("  ".join(parts))
        return "\n".join(lines)

    # ── Serialisation ─────────────────────────────────────

    def save_pickle(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load_pickle(path: str) -> "Archive":
        with open(path, "rb") as f:
            archive = pickle.load(f)
        archive._ensure_all_metadata()
        return archive

    def save_json(self, path: str):
        """Save a human-readable JSON summary (params stored as lists)."""
        self._ensure_all_metadata()
        records = []
        for r in self.results:
            records.append({
                "edges": list(r.topology.edges),
                "label": r.topology.to_label(),
                "score": r.score,
                "rank_score": r.score,
                "train_score": r.train_score,
                "train_raw_score": r.train_raw_score,
                "train_period": r.train_period,
                "train_full_score": r.train_full_score,
                "train_knockout_score": r.train_knockout_score,
                "train_knockout_pass_rate": r.train_knockout_pass_rate,
                "train_knockdown_score": r.train_knockdown_score,
                "train_param_perturb_score": r.train_param_perturb_score,
                "validation_score": r.validation_score,
                "validation_raw_score": r.validation_raw_score,
                "validation_period": r.validation_period,
                "validation_full_score": r.validation_full_score,
                "validation_knockout_score": r.validation_knockout_score,
                "validation_knockout_pass_rate": r.validation_knockout_pass_rate,
                "validation_knockdown_score": r.validation_knockdown_score,
                "validation_param_perturb_score": r.validation_param_perturb_score,
                "generalization_gap": r.generalization_gap,
                "niche_key": r.niche_key,
                "niche": topology_niche_parts(r.topology),
                "iteration": r.iteration,
                "strategy": r.strategy,
                "wall_time": r.wall_time,
                "params": r.params.tolist() if r.params is not None else None,
            })
        niche_elites = []
        for r in self.niche_elites():
            niche_elites.append({
                "niche_key": r.niche_key,
                "niche": topology_niche_parts(r.topology),
                "edges": list(r.topology.edges),
                "label": r.topology.to_label(),
                "score": r.score,
                "validation_score": r.validation_score,
                "validation_full_score": r.validation_full_score,
                "validation_knockout_score": r.validation_knockout_score,
                "validation_knockout_pass_rate": r.validation_knockout_pass_rate,
                "validation_knockdown_score": r.validation_knockdown_score,
                "validation_param_perturb_score": r.validation_param_perturb_score,
                "train_score": r.train_score,
                "generalization_gap": r.generalization_gap,
                "iteration": r.iteration,
            })
        payload = {
            "stats": self.score_stats(),
            "niche_counts": self.niche_counts(),
            "niche_elites": niche_elites,
            "results": records,
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
