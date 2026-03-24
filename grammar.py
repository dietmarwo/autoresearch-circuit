"""
Topology Grammar for 3-Node Gene Regulatory Networks

Components: 3 transcription factors (genes) named A, B, C.
Each gene has constitutive production and degradation.

Pairwise regulatory edges:
  For each ordered pair (X, Y), the edge is one of:
    0 = absent
    1 = activation  (X promotes production of Y)
    2 = inhibition  (X represses production of Y)

Self-regulation edges included (A->A, B->B, C->C).

Encoding: a topology is a tuple of 9 integers:
  (A->A, B->B, C->C, A->B, A->C, B->A, B->C, C->A, C->B)
  Each value in {0, 1, 2}

Total raw space: 3^9 = 19683
After filtering (min/max edges, no isolated nodes): several thousand.
"""

from itertools import product as iproduct
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np

import config as cfg


@dataclass(frozen=True)
class Topology:
    """Immutable representation of a 3-node regulatory network topology."""

    edges: Tuple[int, ...]  # length-9, values in {0, 1, 2}

    def __post_init__(self):
        if len(self.edges) != 9:
            raise ValueError(f"Topology needs 9 edge values, got {len(self.edges)}")
        if not all(e in cfg.EDGE_VALUES for e in self.edges):
            raise ValueError(f"Edge values must be in {cfg.EDGE_VALUES}, got {self.edges}")

    # -- Properties ----------------------------------------

    @property
    def num_active_edges(self) -> int:
        return sum(1 for e in self.edges if e != 0)

    @property
    def num_params(self) -> int:
        """Total continuous parameters: 2 per gene + 2 per active edge."""
        return 2 * cfg.NUM_GENES + 2 * self.num_active_edges

    @property
    def has_isolated_node(self) -> bool:
        """True if any gene has zero incoming AND zero outgoing active edges."""
        for gi in range(cfg.NUM_GENES):
            connected = False
            for idx in range(9):
                src, tgt = cfg.EDGE_INDEX_MAP[idx]
                if self.edges[idx] != 0 and (src == gi or tgt == gi):
                    connected = True
                    break
            if not connected:
                return True
        return False

    def is_valid(self,
                 min_edges: int = cfg.MIN_ACTIVE_EDGES,
                 max_edges: int = cfg.MAX_ACTIVE_EDGES) -> bool:
        n = self.num_active_edges
        return (min_edges <= n <= max_edges) and not self.has_isolated_node

    # -- Display -------------------------------------------

    def to_label(self) -> str:
        """Human-readable motif description."""
        parts = []
        for idx, val in enumerate(self.edges):
            if val != 0:
                kind = "act" if val == 1 else "inh"
                parts.append(f"{cfg.EDGE_NAMES[idx]}({kind})")
        return " | ".join(parts) if parts else "(empty)"

    def to_dict(self) -> dict:
        return {
            "edges": list(self.edges),
            "num_active": self.num_active_edges,
            "num_params": self.num_params,
            "label": self.to_label(),
        }

    def __repr__(self) -> str:
        return f"Topology({self.edges})  [{self.to_label()}]"


# -- Grammar operations -------------------------------------------


def enumerate_valid_topologies(
    min_edges: int = cfg.MIN_ACTIVE_EDGES,
    max_edges: int = cfg.MAX_ACTIVE_EDGES,
) -> List[Topology]:
    """Generate all valid topologies within the bounded grammar."""
    valid = []
    for combo in iproduct(range(3), repeat=9):
        t = Topology(edges=combo)
        if t.is_valid(min_edges, max_edges):
            valid.append(t)
    return valid


def mutate_topology(t: Topology, rng: np.random.Generator) -> Topology:
    """
    Single-edge mutation: change one random edge to a different value.
    Does NOT guarantee the result is valid -- caller must check.
    """
    edges = list(t.edges)
    idx = rng.integers(0, 9)
    choices = [v for v in range(3) if v != edges[idx]]
    edges[idx] = int(rng.choice(choices))
    return Topology(edges=tuple(edges))


def crossover_topologies(t1: Topology, t2: Topology,
                         rng: np.random.Generator) -> Topology:
    """Uniform crossover: each edge picked randomly from one parent."""
    edges = []
    for e1, e2 in zip(t1.edges, t2.edges):
        edges.append(e1 if rng.random() < 0.5 else e2)
    return Topology(edges=tuple(edges))


# -- Well-known motifs (for testing / seeding) ----------------------

REPRESSILATOR = Topology(edges=(0, 0, 0, 2, 0, 0, 2, 2, 0))
#  A-|B, B-|C, C-|A  (classic 3-node ring oscillator)

GOODWIN_LOOP = Topology(edges=(0, 0, 0, 1, 0, 0, 1, 2, 0))
#  A->B(act), B->C(act), C-|A  (Goodwin-style delayed negative feedback)

TOGGLE_SWITCH_AB = Topology(edges=(0, 0, 0, 2, 0, 2, 0, 0, 0))
#  A-|B, B-|A  (mutual inhibition, bistability motif)


# -- Quick self-test ------------------------------------------------

if __name__ == "__main__":
    all_valid = enumerate_valid_topologies()
    print(f"Total valid topologies: {len(all_valid)}")
    print(f"Repressilator valid: {REPRESSILATOR.is_valid()}")
    print(f"Repressilator:  {REPRESSILATOR}")
    print(f"  params: {REPRESSILATOR.num_params}")
    print(f"Goodwin loop:   {GOODWIN_LOOP}")
    print(f"Toggle switch:  {TOGGLE_SWITCH_AB}")

    rng = np.random.default_rng(42)
    mutant = mutate_topology(REPRESSILATOR, rng)
    print(f"Mutant of repressilator: {mutant}  valid={mutant.is_valid()}")
