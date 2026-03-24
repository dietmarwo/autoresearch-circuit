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
    """One evaluated topology."""
    topology: Topology
    score: float
    params: Optional[np.ndarray] = None
    iteration: int = 0
    wall_time: float = 0.0
    strategy: str = ""
    timestamp: float = field(default_factory=time.time)


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
        return {
            "n": len(scores),
            "best": max(scores),
            "mean": float(np.mean(scores)),
            "median": float(np.median(scores)),
            "std": float(np.std(scores)),
            "nonzero": sum(1 for s in scores if s > 0.01),
        }

    # ── Display ───────────────────────────────────────────

    def summary(self, top: int = 5) -> str:
        if not self.results:
            return "Archive empty."
        stats = self.score_stats()
        lines = [
            f"Evaluated {stats['n']} topologies  "
            f"(best={stats['best']:.4f}  mean={stats['mean']:.4f}  "
            f"nonzero={stats['nonzero']})",
        ]
        for i, r in enumerate(self.top_k(top)):
            lines.append(
                f"  #{i+1}: score={r.score:.4f}  "
                f"edges={r.topology.num_active_edges}  "
                f"[{r.topology.to_label()}]"
            )
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
