"""
Archive — persistent storage and analysis of search results.

Keeps all evaluated (topology, score, params) triples plus metadata.
Supports serialisation to JSON and pickle.
"""

import json
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

from grammar import Topology


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
    validation_score: Optional[float] = None
    validation_raw_score: Optional[float] = None
    validation_period: Optional[float] = None
    generalization_gap: Optional[float] = None
    timestamp: float = field(default_factory=time.time)

    def __setstate__(self, state):
        """Backfill newly added fields when loading older pickles."""
        self.__dict__.update(state)
        for name in (
            "train_score",
            "train_raw_score",
            "train_period",
            "validation_score",
            "validation_raw_score",
            "validation_period",
            "generalization_gap",
        ):
            self.__dict__.setdefault(name, None)


class Archive:
    """Stores all evaluated topologies and their scores."""

    def __init__(self):
        self.results: List[SearchResult] = []

    # ── Core ops ──────────────────────────────────────────

    def add(self, result: SearchResult):
        self.results.append(result)

    def __len__(self) -> int:
        return len(self.results)

    @property
    def best(self) -> Optional[SearchResult]:
        if not self.results:
            return None
        return max(self.results, key=lambda r: r.score)

    def top_k(self, k: int = 5) -> List[SearchResult]:
        return sorted(self.results, key=lambda r: r.score, reverse=True)[:k]

    def already_evaluated(self, topology: Topology) -> bool:
        return any(r.topology.edges == topology.edges for r in self.results)

    # ── Statistics ────────────────────────────────────────

    def score_stats(self) -> dict:
        if not self.results:
            return {"n": 0}
        scores = [r.score for r in self.results]
        stats = {
            "n": len(scores),
            "best": max(scores),
            "mean": float(np.mean(scores)),
            "median": float(np.median(scores)),
            "std": float(np.std(scores)),
            "nonzero": sum(1 for s in scores if s > 0.01),
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

    def summary(self, top: int = 5) -> str:
        if not self.results:
            return "Archive empty."
        stats = self.score_stats()
        lines = [
            f"Evaluated {stats['n']} topologies  "
            f"(best_rank={stats['best']:.4f}  mean_rank={stats['mean']:.4f}  "
            f"nonzero={stats['nonzero']})",
        ]
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
            parts.append(f"edges={r.topology.num_active_edges}")
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
            return pickle.load(f)

    def save_json(self, path: str):
        """Save a human-readable JSON summary (params stored as lists)."""
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
                "validation_score": r.validation_score,
                "validation_raw_score": r.validation_raw_score,
                "validation_period": r.validation_period,
                "generalization_gap": r.generalization_gap,
                "iteration": r.iteration,
                "strategy": r.strategy,
                "wall_time": r.wall_time,
                "params": r.params.tolist() if r.params is not None else None,
            })
        payload = {
            "stats": self.score_stats(),
            "results": records,
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
