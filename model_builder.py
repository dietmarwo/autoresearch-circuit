"""
Model Builder: Topology × Parameters → GillesPy2 Stochastic Model

Reaction scheme per gene X:
  ∅ → X   production at rate = basal + Σ regulatory_contributions
  X → ∅   degradation at rate = degradation_rate * X

Regulatory contributions (Hill-like custom propensities):
  Activation by R on X:  strength * R^n / (K^n + R^n)
  Inhibition by R on X:  strength * K^n / (K^n + R^n)

Parameter vector layout for a given topology:
  [basal_A, deg_A, basal_B, deg_B, basal_C, deg_C,
   strength_edge_i, hill_edge_i, ...]
  where only active edges (edge value != 0) appear, in edge-index order.
"""

import numpy as np
import gillespy2

from grammar import Topology
import config as cfg


def build_param_bounds(topology: Topology):
    """
    Return (lower_bounds, upper_bounds) arrays matching the parameter vector
    layout for the given topology.
    """
    lower, upper = [], []
    # Per-gene: basal production, degradation rate
    for _ in cfg.GENES:
        lower.extend([cfg.BASAL_PRODUCTION_BOUNDS[0], cfg.DEGRADATION_RATE_BOUNDS[0]])
        upper.extend([cfg.BASAL_PRODUCTION_BOUNDS[1], cfg.DEGRADATION_RATE_BOUNDS[1]])
    # Per active edge: regulatory strength, Hill coefficient
    for val in topology.edges:
        if val != 0:
            lower.extend([cfg.REG_STRENGTH_BOUNDS[0], cfg.HILL_COEFF_BOUNDS[0]])
            upper.extend([cfg.REG_STRENGTH_BOUNDS[1], cfg.HILL_COEFF_BOUNDS[1]])
    return np.array(lower, dtype=np.float64), np.array(upper, dtype=np.float64)


def build_model(topology: Topology, params: np.ndarray,
                t_end: float = cfg.SIM_T_END,
                n_steps: int = cfg.SIM_N_STEPS) -> gillespy2.Model:
    """
    Construct a GillesPy2 Model from a topology and continuous parameter vector.

    Args:
        topology: A Topology instance defining the regulatory edges.
        params:   Continuous parameter vector (length = topology.num_params).
        t_end:    Simulation end time.
        n_steps:  Number of time-points in the output trajectory.

    Returns:
        A gillespy2.Model ready for stochastic simulation.

    Raises:
        ValueError: if params has wrong length.
    """
    expected = topology.num_params
    if len(params) != expected:
        raise ValueError(
            f"Expected {expected} params for topology with "
            f"{topology.num_active_edges} active edges, got {len(params)}"
        )

    model = gillespy2.Model(name="CircuitModel")

    # ── Species ───────────────────────────────────────────
    species = {}
    for gene in cfg.GENES:
        sp = gillespy2.Species(name=gene, initial_value=cfg.INITIAL_COPIES)
        model.add_species(sp)
        species[gene] = sp

    # ── Parse parameter vector ────────────────────────────
    ptr = 0
    gene_params = {}
    for gene in cfg.GENES:
        gene_params[gene] = {
            "basal": float(params[ptr]),
            "degradation": float(params[ptr + 1]),
        }
        ptr += 2

    edge_params = {}
    for idx, val in enumerate(topology.edges):
        if val != 0:
            edge_params[idx] = {
                "strength": float(params[ptr]),
                "hill": float(params[ptr + 1]),
            }
            ptr += 2

    # ── GillesPy2 Parameter objects ───────────────────────
    gp2_params = {}
    for gene in cfg.GENES:
        for pname, pval in gene_params[gene].items():
            key = f"{gene}_{pname}"
            gp2_params[key] = gillespy2.Parameter(name=key, expression=str(pval))
            model.add_parameter(gp2_params[key])

    for idx, ep in edge_params.items():
        for pname, pval in ep.items():
            key = f"edge{idx}_{pname}"
            gp2_params[key] = gillespy2.Parameter(name=key, expression=str(pval))
            model.add_parameter(gp2_params[key])

    # ── Reactions ─────────────────────────────────────────
    K = str(cfg.HILL_K)

    for gi, gene in enumerate(cfg.GENES):
        # ── Production ────────────────────────────────────
        # Basal term always present
        rate_terms = [f"{gene}_basal"]

        # Add Hill terms for each regulatory edge targeting this gene
        for idx, val in enumerate(topology.edges):
            if val == 0:
                continue
            src_idx, tgt_idx = cfg.EDGE_INDEX_MAP[idx]
            if tgt_idx != gi:
                continue

            src_gene = cfg.GENES[src_idx]
            s_key = f"edge{idx}_strength"
            h_key = f"edge{idx}_hill"

            if val == 1:  # activation
                term = f"{s_key} * {src_gene}**{h_key} / ({K}**{h_key} + {src_gene}**{h_key})"
            else:          # inhibition
                term = f"{s_key} * {K}**{h_key} / ({K}**{h_key} + {src_gene}**{h_key})"
            rate_terms.append(term)

        prod_rate = " + ".join(rate_terms)
        model.add_reaction(gillespy2.Reaction(
            name=f"produce_{gene}",
            reactants={},
            products={species[gene]: 1},
            propensity_function=prod_rate,
        ))

        # ── Degradation ──────────────────────────────────
        model.add_reaction(gillespy2.Reaction(
            name=f"degrade_{gene}",
            reactants={species[gene]: 1},
            products={},
            rate=gp2_params[f"{gene}_degradation"],
        ))

    # ── Timespan ──────────────────────────────────────────
    model.timespan(np.linspace(0, t_end, n_steps))

    return model


# ── Quick self-test ──────────────────────────────────────────

if __name__ == "__main__":
    from grammar import REPRESSILATOR

    topo = REPRESSILATOR
    lower, upper = build_param_bounds(topo)
    print(f"Topology: {topo}")
    print(f"Param count: {topo.num_params}")
    print(f"Lower bounds: {lower}")
    print(f"Upper bounds: {upper}")

    # Build with midpoint params
    mid = (lower + upper) / 2.0
    model = build_model(topo, mid, t_end=50.0, n_steps=200)
    print(f"\nModel species: {list(model.get_all_species().keys())}")
    print(f"Model reactions: {list(model.get_all_reactions().keys())}")
    print(f"Model parameters: {list(model.get_all_parameters().keys())}")

    # Quick sim test
    print("\nRunning a quick SSA simulation (t_end=50) ...")
    result = model.run(solver=cfg.SIM_SOLVER, seed=42)
    for gene in cfg.GENES:
        vals = np.array(result[gene])
        print(f"  {gene}: mean={vals.mean():.1f}  std={vals.std():.1f}  "
              f"min={vals.min():.0f}  max={vals.max():.0f}")
    print("Model builder OK.")
