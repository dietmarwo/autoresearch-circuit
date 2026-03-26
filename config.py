"""
Central configuration for the circuit search project.

All tunable hyperparameters live here so they can be adjusted
without touching module internals.
"""

from __future__ import annotations

import gillespy2 as _gp2


def _build_edge_index_map(genes: list[str]) -> list[tuple[int, int]]:
    """Build a deterministic edge ordering: self edges first, then all cross edges."""
    mapping: list[tuple[int, int]] = []
    for idx in range(len(genes)):
        mapping.append((idx, idx))
    for src in range(len(genes)):
        for tgt in range(len(genes)):
            if src != tgt:
                mapping.append((src, tgt))
    return mapping


def _set_topology_space(genes: list[str], min_edges: int, max_edges: int) -> None:
    """Update all topology-space globals for the active experiment."""
    global GENES, NUM_GENES, MIN_ACTIVE_EDGES, MAX_ACTIVE_EDGES
    global EDGE_INDEX_MAP, EDGE_NAMES, NUM_EDGE_SLOTS

    GENES = list(genes)
    NUM_GENES = len(GENES)
    MIN_ACTIVE_EDGES = int(min_edges)
    MAX_ACTIVE_EDGES = int(max_edges)
    EDGE_INDEX_MAP = _build_edge_index_map(GENES)
    EDGE_NAMES = [f"{GENES[src]}->{GENES[tgt]}" for src, tgt in EDGE_INDEX_MAP]
    NUM_EDGE_SLOTS = len(EDGE_INDEX_MAP)


# ── Experiment presets ───────────────────────────────────
AVAILABLE_EXPERIMENTS = ("oscillator3", "robust5")
DEFAULT_EXPERIMENT = "oscillator3"
EXPERIMENT = DEFAULT_EXPERIMENT

# ── Shared topology/scoring constants ────────────────────
EDGE_VALUES = (0, 1, 2)          # absent / activation / inhibition
SIM_ALGORITHM = "NumPySSA"
SIM_SOLVER = _gp2.NumPySSASolver
INITIAL_COPIES = 10
HILL_K = 20.0
TOPOLOGY_ENUMERATION_MAX_RAW_SPACE = 2_000_000
MAX_RANDOM_TOPOLOGY_TRIES = 5000

# ── Parameter Bounds ─────────────────────────────────────
BASAL_PRODUCTION_BOUNDS = (0.1, 50.0)
DEGRADATION_RATE_BOUNDS = (0.005, 1.0)
REG_STRENGTH_BOUNDS = (0.1, 100.0)
HILL_COEFF_BOUNDS = (1.0, 5.0)

# ── Per-gene oscillation detector ────────────────────────
MIN_PEAKS = 3
PEAK_COUNT_FULL_CREDIT = 8
PEAK_HEIGHT_FACTOR = 0.3
PEAK_MIN_DISTANCE_FRAC = 20
FLAT_STD_THRESHOLD = 1.0
MIN_OSC_AMPLITUDE = 8.0
MIN_REL_AMPLITUDE = 0.15
MIN_AMP_TO_MEAN = 0.20
W_SPACING_REGULARITY = 0.25
W_AMPLITUDE_REGULARITY = 0.20
W_PEAK_COUNT = 0.20
W_PERSISTENCE = 0.15
W_AUTOCORRELATION = 0.20

# ── Trace aggregation defaults (overwritten per experiment) ──
PHENOTYPE_MODE = "traveling_wave_3"
MULTI_GENE_THRESHOLD = 0.3
MULTI_GENE_BONUS_MAX = 0.1
COHERENCE_GENE_THRESHOLD = 0.55
COHERENCE_TARGET_GENES = 3
TRACE_TARGET_PARTICIPANTS = 3
TRACE_BEST_GENE_WEIGHT = 0.45
TRACE_PARTICIPATION_WEIGHT = 0.25
TRACE_PERIOD_COHERENCE_WEIGHT = 0.15
TRACE_PHASE_COHERENCE_WEIGHT = 0.15
TRACE_SUPPORT_WEIGHT = 0.0

# ── Robustness objective defaults (overwritten per experiment) ──
ROBUST_FULL_WEIGHT = 1.0
ROBUST_KNOCKOUT_WEIGHT = 0.0
ROBUST_KNOCKOUT_PASS_WEIGHT = 0.0
ROBUST_KNOCKDOWN_WEIGHT = 0.0
ROBUST_PARAM_PERTURB_WEIGHT = 0.0
TRAIN_KNOCKOUT_SAMPLES = 0       # 0 -> no knockout term during optimisation
VALID_KNOCKOUT_SAMPLES = 0       # -1 -> all single-gene knockouts
TRAIN_KNOCKDOWN_SAMPLES = 0      # 0 -> no partial knockdown term
VALID_KNOCKDOWN_SAMPLES = 0      # -1 -> all single-gene knockdowns
KNOCKDOWN_FACTOR = 0.35          # remaining production / initial copy fraction
TRAIN_PARAM_PERTURB_SAMPLES = 0
VALID_PARAM_PERTURB_SAMPLES = 0
PARAM_PERTURB_SIGMA = 0.20       # multiplicative log-normal jitter around x*
ROBUST_SCENARIO_AGGREGATION_QUANTILE = 0.50
ROBUST_SUCCESS_THRESHOLD = 0.55

# ── Validation-aware ranking ─────────────────────────────
GENERALIZATION_GAP_PENALTY = 0.25  # rank_score = val_score - penalty * |train - val|

# ── Agentic defaults that may change with experiment ────
AGENTIC_SEARCH_N = 30
AGENTIC_MODE = "guided"         # blind / guided
AGENTIC_BOOTSTRAP_ITERS = 4
AGENTIC_EXPLORE_MIN_HAMMING = 3
AGENTIC_DIVERSITY_TOP_K = 5
LLM_TOP_K = 10
LLM_NICHE_K = 6
LLM_RECENT_K = 10

# ── LLM backend defaults ─────────────────────────────────
LLM_BACKEND = "auto"            # auto / openai / claude / gemini / minimax
LLM_BASE_URL = None             # OpenAI-compatible endpoint; None -> provider default
LLM_MODEL = None
LLM_TEMPERATURE = 1.0
LLM_THINKING_EFFORT = "high"    # none / high
LLM_MAX_TOKENS = 8192
LLM_MAX_CONTEXT_EXCHANGES = 2
LLM_EXCHANGE_MAX_CHARS = 2000
LLM_MAX_CONSECUTIVE_FAILURES = 5


def set_experiment(name: str) -> str:
    """Switch all experiment-specific globals in place."""
    global EXPERIMENT, SIM_T_END, VALID_T_END, SIM_N_STEPS
    global INNER_N_SEEDS, VALID_N_SEEDS, TRAIN_SEED_OFFSET, VALID_SEED_OFFSET
    global RANDOM_SEARCH_N, EVO_SEARCH_N, MAX_MUTATION_TRIES
    global INNER_MAX_EVALS, INNER_NUM_WORKERS, PENALTY_VALUE
    global PHENOTYPE_MODE, COHERENCE_GENE_THRESHOLD, COHERENCE_TARGET_GENES
    global TRACE_TARGET_PARTICIPANTS, TRACE_BEST_GENE_WEIGHT
    global TRACE_PARTICIPATION_WEIGHT, TRACE_PERIOD_COHERENCE_WEIGHT
    global TRACE_PHASE_COHERENCE_WEIGHT, TRACE_SUPPORT_WEIGHT
    global ROBUST_FULL_WEIGHT, ROBUST_KNOCKOUT_WEIGHT
    global ROBUST_KNOCKOUT_PASS_WEIGHT, ROBUST_KNOCKDOWN_WEIGHT
    global ROBUST_PARAM_PERTURB_WEIGHT, TRAIN_KNOCKOUT_SAMPLES
    global VALID_KNOCKOUT_SAMPLES, TRAIN_KNOCKDOWN_SAMPLES
    global VALID_KNOCKDOWN_SAMPLES, KNOCKDOWN_FACTOR
    global TRAIN_PARAM_PERTURB_SAMPLES, VALID_PARAM_PERTURB_SAMPLES
    global PARAM_PERTURB_SIGMA, ROBUST_SCENARIO_AGGREGATION_QUANTILE
    global ROBUST_SUCCESS_THRESHOLD
    global AGENTIC_SEARCH_N, AGENTIC_BOOTSTRAP_ITERS
    global AGENTIC_EXPLORE_MIN_HAMMING, AGENTIC_DIVERSITY_TOP_K
    global LLM_TOP_K, LLM_NICHE_K, LLM_RECENT_K

    normalized = (name or DEFAULT_EXPERIMENT).strip().lower()
    if normalized not in AVAILABLE_EXPERIMENTS:
        raise ValueError(
            f"Unknown experiment '{name}'. Available: {', '.join(AVAILABLE_EXPERIMENTS)}"
        )

    EXPERIMENT = normalized
    MAX_MUTATION_TRIES = 20
    PENALTY_VALUE = 1e6

    if normalized == "oscillator3":
        _set_topology_space(["A", "B", "C"], min_edges=2, max_edges=6)
        SIM_T_END = 200.0
        VALID_T_END = 400.0
        SIM_N_STEPS = 1000
        INNER_N_SEEDS = 2
        VALID_N_SEEDS = 5
        TRAIN_SEED_OFFSET = 42
        VALID_SEED_OFFSET = 10_042
        RANDOM_SEARCH_N = 30
        EVO_SEARCH_N = 80
        INNER_MAX_EVALS = 480
        INNER_NUM_WORKERS = 16
        AGENTIC_SEARCH_N = 30
        AGENTIC_BOOTSTRAP_ITERS = 4
        AGENTIC_EXPLORE_MIN_HAMMING = 3
        AGENTIC_DIVERSITY_TOP_K = 5
        LLM_TOP_K = 10
        LLM_NICHE_K = 6
        LLM_RECENT_K = 10

        PHENOTYPE_MODE = "traveling_wave_3"
        COHERENCE_GENE_THRESHOLD = 0.55
        COHERENCE_TARGET_GENES = 3
        TRACE_TARGET_PARTICIPANTS = 3
        TRACE_BEST_GENE_WEIGHT = 0.45
        TRACE_PARTICIPATION_WEIGHT = 0.25
        TRACE_PERIOD_COHERENCE_WEIGHT = 0.15
        TRACE_PHASE_COHERENCE_WEIGHT = 0.15
        TRACE_SUPPORT_WEIGHT = 0.0

        ROBUST_FULL_WEIGHT = 1.0
        ROBUST_KNOCKOUT_WEIGHT = 0.0
        ROBUST_KNOCKOUT_PASS_WEIGHT = 0.0
        ROBUST_KNOCKDOWN_WEIGHT = 0.0
        ROBUST_PARAM_PERTURB_WEIGHT = 0.0
        TRAIN_KNOCKOUT_SAMPLES = 0
        VALID_KNOCKOUT_SAMPLES = 0
        TRAIN_KNOCKDOWN_SAMPLES = 0
        VALID_KNOCKDOWN_SAMPLES = 0
        KNOCKDOWN_FACTOR = 0.35
        TRAIN_PARAM_PERTURB_SAMPLES = 0
        VALID_PARAM_PERTURB_SAMPLES = 0
        PARAM_PERTURB_SIGMA = 0.20
        ROBUST_SCENARIO_AGGREGATION_QUANTILE = 0.50
        ROBUST_SUCCESS_THRESHOLD = 0.55

    else:  # robust5
        _set_topology_space(["A", "B", "C", "D", "E"], min_edges=5, max_edges=15)
        SIM_T_END = 250.0
        VALID_T_END = 500.0
        SIM_N_STEPS = 1200
        INNER_N_SEEDS = 1
        VALID_N_SEEDS = 5
        TRAIN_SEED_OFFSET = 142
        VALID_SEED_OFFSET = 20_142
        RANDOM_SEARCH_N = 20
        EVO_SEARCH_N = 40
        INNER_MAX_EVALS = 320
        INNER_NUM_WORKERS = 8
        AGENTIC_SEARCH_N = 20
        AGENTIC_BOOTSTRAP_ITERS = 6
        AGENTIC_EXPLORE_MIN_HAMMING = 5
        AGENTIC_DIVERSITY_TOP_K = 6
        LLM_TOP_K = 8
        LLM_NICHE_K = 8
        LLM_RECENT_K = 12

        PHENOTYPE_MODE = "robust_oscillator_5"
        COHERENCE_GENE_THRESHOLD = 0.50
        COHERENCE_TARGET_GENES = 3
        TRACE_TARGET_PARTICIPANTS = 3
        TRACE_BEST_GENE_WEIGHT = 0.20
        TRACE_PARTICIPATION_WEIGHT = 0.35
        TRACE_PERIOD_COHERENCE_WEIGHT = 0.25
        TRACE_PHASE_COHERENCE_WEIGHT = 0.0
        TRACE_SUPPORT_WEIGHT = 0.20

        ROBUST_FULL_WEIGHT = 0.20
        ROBUST_KNOCKOUT_WEIGHT = 0.25
        ROBUST_KNOCKOUT_PASS_WEIGHT = 0.15
        ROBUST_KNOCKDOWN_WEIGHT = 0.20
        ROBUST_PARAM_PERTURB_WEIGHT = 0.20
        TRAIN_KNOCKOUT_SAMPLES = 2
        VALID_KNOCKOUT_SAMPLES = -1
        TRAIN_KNOCKDOWN_SAMPLES = 0
        VALID_KNOCKDOWN_SAMPLES = -1
        KNOCKDOWN_FACTOR = 0.35
        TRAIN_PARAM_PERTURB_SAMPLES = 0
        VALID_PARAM_PERTURB_SAMPLES = 8
        PARAM_PERTURB_SIGMA = 0.25
        ROBUST_SCENARIO_AGGREGATION_QUANTILE = 0.25
        ROBUST_SUCCESS_THRESHOLD = 0.82

    return EXPERIMENT


# Initialize module globals for the default experiment.
set_experiment(DEFAULT_EXPERIMENT)
