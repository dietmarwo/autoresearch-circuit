"""
Central configuration for the circuit search project.

All tunable hyperparameters live here so they can be adjusted
without touching module internals.
"""

# ── Topology Grammar ──────────────────────────────────────
GENES = ["A", "B", "C"]
NUM_GENES = len(GENES)
EDGE_VALUES = (0, 1, 2)          # absent / activation / inhibition
MIN_ACTIVE_EDGES = 2
MAX_ACTIVE_EDGES = 6

# Edge index semantics (source, target)
EDGE_INDEX_MAP = [
    (0, 0), (1, 1), (2, 2),     # self-regulation
    (0, 1), (0, 2),             # A regulates others
    (1, 0), (1, 2),             # B regulates others
    (2, 0), (2, 1),             # C regulates others
]
EDGE_NAMES = [
    f"{GENES[s]}->{GENES[t]}" for s, t in EDGE_INDEX_MAP
]
NUM_EDGE_SLOTS = len(EDGE_INDEX_MAP)  # 9

# ── Simulation ────────────────────────────────────────────
SIM_T_END = 200.0                # training simulation duration
VALID_T_END = 400.0              # longer holdout validation duration
SIM_N_STEPS = 1000               # output time points
SIM_ALGORITHM = "NumPySSA"
# NOTE: for model.run() we pass the solver class, not the string.
# The string is kept for logging only.
import gillespy2 as _gp2
SIM_SOLVER = _gp2.NumPySSASolver
INITIAL_COPIES = 10              # initial molecule count per species
HILL_K = 20.0                    # fixed half-max in Hill functions

# ── Parameter Bounds ──────────────────────────────────────
BASAL_PRODUCTION_BOUNDS = (0.1, 50.0)
DEGRADATION_RATE_BOUNDS = (0.005, 1.0)
REG_STRENGTH_BOUNDS = (0.1, 100.0)
HILL_COEFF_BOUNDS = (1.0, 5.0)

# ── Evaluator ─────────────────────────────────────────────
INNER_N_SEEDS = 2                # SSA seeds during inner optimisation
VALID_N_SEEDS = 5                # SSA seeds for final validation
TRAIN_SEED_OFFSET = 42           # optimisation seeds
VALID_SEED_OFFSET = 10_042       # disjoint holdout seeds
MIN_PEAKS = 3                    # minimum peaks to count as oscillating
PEAK_COUNT_FULL_CREDIT = 8       # peak count where count_score saturates
PEAK_HEIGHT_FACTOR = 0.3         # fraction of std above mean for peak detection
PEAK_MIN_DISTANCE_FRAC = 20      # min distance = len(trace) / this
FLAT_STD_THRESHOLD = 1.0         # std below this → flat trace
MIN_OSC_AMPLITUDE = 8.0          # minimum absolute peak-to-trough amplitude (molecules)
MIN_REL_AMPLITUDE = 0.15         # minimum peak-to-trough / signal_range ratio
MIN_AMP_TO_MEAN = 0.20           # minimum oscillation amplitude / mean_signal ratio

# Score component weights (sum to 1.0)
W_SPACING_REGULARITY = 0.30
W_AMPLITUDE_REGULARITY = 0.25
W_PEAK_COUNT = 0.25
W_PERSISTENCE = 0.20

# Multi-gene bonus
MULTI_GENE_THRESHOLD = 0.3      # gene scores above this count as oscillating
MULTI_GENE_BONUS_MAX = 0.1      # maximum multi-gene bonus
GENERALIZATION_GAP_PENALTY = 0.5  # rank_score = val_score - penalty * |train - val|

# ── Inner Optimizer (fcmaes) ──────────────────────────────
INNER_MAX_EVALS = 100           # evaluations per retry
INNER_NUM_RETRIES = 16          # parallel retries (coordinated)
PENALTY_VALUE = 1e6              # returned on simulation failure

# ── Outer Loop ────────────────────────────────────────────
RANDOM_SEARCH_N = 30             # topologies to sample in random search
EVO_SEARCH_N = 80                # iterations for evolutionary search
MAX_MUTATION_TRIES = 20          # attempts to mutate into a valid topology

# ── Agentic Loop ──────────────────────────────────────────
AGENTIC_SEARCH_N = 30
LLM_BACKEND = "auto"            # auto / openai / claude / gemini / minimax
LLM_BASE_URL = None             # OpenAI-compatible endpoint; None -> provider default
LLM_MODEL = "claude-sonnet-4-20250514"
LLM_TEMPERATURE = 1.0
LLM_THINKING_EFFORT = "none"    # none / high
LLM_MAX_TOKENS = 8192
LLM_MAX_CONTEXT_EXCHANGES = 2   # lightweight prior user/assistant turns kept
LLM_TOP_K = 10                  # best results shown in the regenerated prompt
LLM_RECENT_K = 10               # recent results shown in the regenerated prompt
LLM_EXCHANGE_MAX_CHARS = 2000   # truncate remembered assistant summaries
LLM_MAX_CONSECUTIVE_FAILURES = 5
