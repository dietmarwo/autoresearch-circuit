"""
Topology Grammar for bounded gene regulatory networks.

The active experiment chooses the number of transcription factors
and the allowed topology size.

Pairwise regulatory edges:
  For each ordered pair (X, Y), the edge is one of:
    0 = absent
    1 = activation  (X promotes production of Y)
    2 = inhibition  (X represses production of Y)

Self-regulation edges are included.

Encoding: a topology is a tuple of `cfg.NUM_EDGE_SLOTS` integers.
The edge ordering is defined by `cfg.EDGE_INDEX_MAP`:
  - self-regulation edges first
  - then every ordered cross-regulation edge

The raw space scales as `len(cfg.EDGE_VALUES) ** cfg.NUM_EDGE_SLOTS`.
For large experiments we sample valid topologies instead of enumerating them.
"""

from itertools import product as iproduct
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np

import config as cfg


@dataclass(frozen=True)
class Topology:
    """Immutable representation of a bounded regulatory network topology."""

    edges: Tuple[int, ...]  # length = cfg.NUM_EDGE_SLOTS, values in {0, 1, 2}

    def __post_init__(self):
        if len(self.edges) != cfg.NUM_EDGE_SLOTS:
            raise ValueError(
                f"Topology needs {cfg.NUM_EDGE_SLOTS} edge values, got {len(self.edges)}"
            )
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
            for idx in range(cfg.NUM_EDGE_SLOTS):
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
    if not can_enumerate_topologies():
        raise RuntimeError(
            "Topology space too large to enumerate exactly for the current experiment."
        )
    valid = []
    for combo in iproduct(range(len(cfg.EDGE_VALUES)), repeat=cfg.NUM_EDGE_SLOTS):
        t = Topology(edges=combo)
        if t.is_valid(min_edges, max_edges):
            valid.append(t)
    return valid


def raw_topology_space_size() -> int:
    """Return the size of the unfiltered topology space."""
    return len(cfg.EDGE_VALUES) ** cfg.NUM_EDGE_SLOTS


def can_enumerate_topologies(max_raw_space: int = cfg.TOPOLOGY_ENUMERATION_MAX_RAW_SPACE) -> bool:
    """Return True if exact enumeration is feasible for the current experiment."""
    return raw_topology_space_size() <= max_raw_space


def random_valid_topology(
    rng: np.random.Generator,
    min_edges: int = cfg.MIN_ACTIVE_EDGES,
    max_edges: int = cfg.MAX_ACTIVE_EDGES,
    max_tries: int = cfg.MAX_RANDOM_TOPOLOGY_TRIES,
) -> Topology:
    """Sample one valid topology without enumerating the full space."""
    for _ in range(max_tries):
        n_active = int(rng.integers(min_edges, max_edges + 1))
        edges = np.zeros(cfg.NUM_EDGE_SLOTS, dtype=np.int64)
        active_slots = rng.choice(cfg.NUM_EDGE_SLOTS, size=n_active, replace=False)
        edges[active_slots] = rng.integers(1, len(cfg.EDGE_VALUES), size=n_active)
        topo = Topology(edges=tuple(int(e) for e in edges))
        if topo.is_valid(min_edges=min_edges, max_edges=max_edges):
            return topo

    # Fallback: keep mutating a sparse valid seed until one passes validation.
    edges = np.zeros(cfg.NUM_EDGE_SLOTS, dtype=np.int64)
    for gi in range(cfg.NUM_GENES):
        candidates = [idx for idx, (src, tgt) in enumerate(cfg.EDGE_INDEX_MAP)
                      if src == gi or tgt == gi]
        idx = int(rng.choice(candidates))
        edges[idx] = int(rng.integers(1, len(cfg.EDGE_VALUES)))
    topo = Topology(edges=tuple(int(e) for e in edges))
    while not topo.is_valid(min_edges=min_edges, max_edges=max_edges):
        topo = mutate_topology(topo, rng)
    return topo


def sample_valid_topologies(
    n: int,
    rng: np.random.Generator,
    min_edges: int = cfg.MIN_ACTIVE_EDGES,
    max_edges: int = cfg.MAX_ACTIVE_EDGES,
) -> List[Topology]:
    """Sample up to n unique valid topologies without full enumeration."""
    sample: List[Topology] = []
    seen = set()
    while len(sample) < n:
        topo = random_valid_topology(rng, min_edges=min_edges, max_edges=max_edges)
        if topo.edges in seen:
            continue
        seen.add(topo.edges)
        sample.append(topo)
    return sample


def mutate_topology(t: Topology, rng: np.random.Generator) -> Topology:
    """
    Single-edge mutation: change one random edge to a different value.
    Does NOT guarantee the result is valid -- caller must check.
    """
    edges = list(t.edges)
    idx = rng.integers(0, cfg.NUM_EDGE_SLOTS)
    choices = [v for v in cfg.EDGE_VALUES if v != edges[idx]]
    edges[idx] = int(rng.choice(choices))
    return Topology(edges=tuple(edges))


def crossover_topologies(t1: Topology, t2: Topology,
                         rng: np.random.Generator) -> Topology:
    """Uniform crossover: each edge picked randomly from one parent."""
    edges = []
    for e1, e2 in zip(t1.edges, t2.edges):
        edges.append(e1 if rng.random() < 0.5 else e2)
    return Topology(edges=tuple(edges))


def _make_3node_motif(edges: tuple[int, ...]) -> Topology | None:
    """Create a classic 3-node motif only when the active experiment matches it."""
    if cfg.NUM_EDGE_SLOTS != 9:
        return None
    return Topology(edges=edges)


# -- Well-known motifs (for testing / seeding in the 3-node experiment) ----

REPRESSILATOR = _make_3node_motif((0, 0, 0, 2, 0, 0, 2, 2, 0))
GOODWIN_LOOP = _make_3node_motif((0, 0, 0, 1, 0, 0, 1, 2, 0))
TOGGLE_SWITCH_AB = _make_3node_motif((0, 0, 0, 2, 0, 2, 0, 0, 0))


# -- Quick self-test ------------------------------------------------

if __name__ == "__main__":
    print(f"Experiment: {cfg.EXPERIMENT}")
    print(f"Genes: {cfg.GENES}")
    print(f"Edge slots: {cfg.NUM_EDGE_SLOTS}")
    print(f"Raw space size: {raw_topology_space_size()}")
    if can_enumerate_topologies():
        all_valid = enumerate_valid_topologies()
        print(f"Total valid topologies: {len(all_valid)}")
    else:
        print("Topology space too large for exact enumeration.")

    if REPRESSILATOR is not None:
        print(f"Repressilator valid: {REPRESSILATOR.is_valid()}")
        print(f"Repressilator:  {REPRESSILATOR}")
        print(f"  params: {REPRESSILATOR.num_params}")
        print(f"Goodwin loop:   {GOODWIN_LOOP}")
        print(f"Toggle switch:  {TOGGLE_SWITCH_AB}")

    rng = np.random.default_rng(42)
    seed = REPRESSILATOR if REPRESSILATOR is not None else random_valid_topology(rng)
    mutant = mutate_topology(seed, rng)
    print(f"Mutant: {mutant}  valid={mutant.is_valid()}")
