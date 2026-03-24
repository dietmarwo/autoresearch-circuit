"""
Phenotype Evaluator — Oscillation Quality Scoring
===================================================

Given a GillesPy2 simulation result, compute:
  - a reported score in [0, 1] for human-facing summaries
  - a raw, non-saturating score for fcmaes optimisation

Key design decisions to avoid false positives:
  1. DETREND the signal before peak detection (rejects monotonic growth)
  2. Require significant TROUGH DEPTH between peaks (rejects noisy flat/growing traces)
  3. Check peak-to-trough amplitude vs signal range
  4. Standard regularity metrics on the detrended signal

Penalties -> score = 0.0:
  - No peaks detected after detrending
  - Insufficient trough depth (noise, not oscillation)
  - Simulation failure / NaN in trace
  - Flat trace (std < threshold)

The hot-path post-processing is numba-accelerated where possible:
  - linear detrending
  - trough and amplitude extraction
  - coefficient-of-variation calculations
  - final per-gene score computation

Peak detection still uses scipy's find_peaks to preserve the current
selection logic and thresholds.
"""

import numpy as np
from scipy.signal import find_peaks
try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover - fallback for environments without numba
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

from grammar import Topology
from model_builder import build_model
import config as cfg


@njit(cache=True)
def _has_nan(vals: np.ndarray) -> bool:
    """Return True if the array contains NaNs."""
    for i in range(len(vals)):
        if np.isnan(vals[i]):
            return True
    return False


@njit(cache=True)
def _mean_numba(vals: np.ndarray) -> float:
    """Compute the mean of a 1-D float array."""
    n = len(vals)
    if n == 0:
        return 0.0
    total = 0.0
    for i in range(n):
        total += vals[i]
    return total / n


@njit(cache=True)
def _std_numba(vals: np.ndarray) -> float:
    """Compute the population std of a 1-D float array."""
    n = len(vals)
    if n == 0:
        return 0.0
    mean = _mean_numba(vals)
    acc = 0.0
    for i in range(n):
        diff = vals[i] - mean
        acc += diff * diff
    return np.sqrt(acc / n)


@njit(cache=True)
def _median_numba(vals: np.ndarray) -> float:
    """Compute the median of a 1-D float array."""
    n = len(vals)
    if n == 0:
        return 0.0
    ordered = np.sort(vals.copy())
    mid = n // 2
    if n % 2 == 1:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


@njit(cache=True)
def _clip01(x: float) -> float:
    """Clip a float to [0, 1]."""
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


@njit(cache=True)
def _detrend_linear_numba(vals: np.ndarray) -> np.ndarray:
    """Remove a linear trend from a signal."""
    n = len(vals)
    if n == 0:
        return np.empty(0, dtype=np.float64)

    x_mean = 0.5 * (n - 1)
    v_mean = _mean_numba(vals)
    numer = 0.0
    denom = 0.0
    for i in range(n):
        dx = i - x_mean
        numer += dx * (vals[i] - v_mean)
        denom += dx * dx

    slope = numer / (denom + 1e-12)
    intercept = v_mean - slope * x_mean
    detrended = np.empty(n, dtype=np.float64)
    for i in range(n):
        detrended[i] = vals[i] - (slope * i + intercept)
    return detrended


def _detrend_linear(vals: np.ndarray) -> np.ndarray:
    """Public detrend wrapper using the cached numba implementation."""
    return _detrend_linear_numba(vals)


@njit(cache=True)
def _find_troughs_between_peaks_numba(vals: np.ndarray, peaks: np.ndarray) -> np.ndarray:
    """For each consecutive pair of peaks, find the minimum value between them."""
    n_troughs = len(peaks) - 1
    if n_troughs <= 0:
        return np.empty(0, dtype=np.float64)

    troughs = np.empty(n_troughs, dtype=np.float64)
    for i in range(n_troughs):
        start = peaks[i]
        end = peaks[i + 1]
        trough = vals[start]
        for j in range(start, end + 1):
            if vals[j] < trough:
                trough = vals[j]
        troughs[i] = trough
    return troughs


def _find_troughs_between_peaks(vals: np.ndarray, peaks: np.ndarray) -> np.ndarray:
    """Wrapper around the cached numba trough finder."""
    return _find_troughs_between_peaks_numba(vals, peaks)


@njit(cache=True)
def _score_from_peaks_numba(
    time: np.ndarray,
    raw_vals: np.ndarray,
    detrended: np.ndarray,
    peaks: np.ndarray,
    min_rel_amplitude: float,
    min_osc_amplitude: float,
    min_amp_to_mean: float,
    peak_count_full_credit: float,
    w_spacing: float,
    w_amplitude: float,
    w_count: float,
    w_persistence: float,
) -> tuple[float, float, float]:
    """Compute reported/raw per-gene scores plus a period estimate."""
    troughs = _find_troughs_between_peaks_numba(detrended, peaks)
    if len(troughs) == 0:
        return 0.0, 0.0, 0.0

    pt_amplitudes = np.empty(len(troughs), dtype=np.float64)
    for i in range(len(troughs)):
        pt_amplitudes[i] = detrended[peaks[i]] - troughs[i]

    signal_min = detrended[0]
    signal_max = detrended[0]
    for i in range(1, len(detrended)):
        if detrended[i] < signal_min:
            signal_min = detrended[i]
        if detrended[i] > signal_max:
            signal_max = detrended[i]
    signal_range = signal_max - signal_min
    if signal_range < 1e-6:
        return 0.0, 0.0, 0.0

    median_pt = _median_numba(pt_amplitudes)
    if median_pt / signal_range < min_rel_amplitude:
        return 0.0, 0.0, 0.0
    if median_pt < min_osc_amplitude:
        return 0.0, 0.0, 0.0

    mean_raw = _mean_numba(raw_vals)
    if mean_raw > 1.0 and median_pt / mean_raw < min_amp_to_mean:
        return 0.0, 0.0, 0.0

    spacing_score = 0.0
    persistence = 0.0
    period = 0.0
    if len(peaks) >= 2:
        spacings = np.empty(len(peaks) - 1, dtype=np.float64)
        for i in range(len(peaks) - 1):
            spacings[i] = time[peaks[i + 1]] - time[peaks[i]]
        spacing_cv = _std_numba(spacings) / (_mean_numba(spacings) + 1e-9)
        spacing_score = _clip01(1.0 - spacing_cv)
        period = _median_numba(spacings)
        total_span = time[-1] - time[0]
        persistence = (time[peaks[-1]] - time[peaks[0]]) / (total_span + 1e-9)

    amp_cv = _std_numba(pt_amplitudes) / (_mean_numba(pt_amplitudes) + 1e-9)
    amp_score = _clip01(1.0 - amp_cv)
    count_score = min(1.0, len(peaks) / peak_count_full_credit)
    count_score_raw = len(peaks) / peak_count_full_credit

    report_score = (
        w_spacing * spacing_score
        + w_amplitude * amp_score
        + w_count * count_score
        + w_persistence * persistence
    )
    raw_score = (
        w_spacing * spacing_score
        + w_amplitude * amp_score
        + w_count * count_score_raw
        + w_persistence * persistence
    )
    return report_score, raw_score, period


@njit(cache=True)
def _aggregate_trace_score_numba(gene_scores: np.ndarray,
                                 multi_gene_threshold: float,
                                 multi_gene_bonus_max: float) -> float:
    """Aggregate per-gene scores into the final trace score."""
    if len(gene_scores) == 0:
        return 0.0

    best = 0.0
    num_oscillating = 0
    for i in range(len(gene_scores)):
        score = gene_scores[i]
        if score > best:
            best = score
        if score > multi_gene_threshold:
            num_oscillating += 1

    if best == 0.0:
        return 0.0

    multi_bonus = 0.0
    if num_oscillating > 1:
        multi_bonus = multi_gene_bonus_max * min(1.0, (num_oscillating - 1) / 2.0)
    return _clip01(best + multi_bonus)


@njit(cache=True)
def _aggregate_trace_raw_score_numba(report_scores: np.ndarray,
                                     raw_scores: np.ndarray,
                                     multi_gene_threshold: float,
                                     multi_gene_bonus_max: float) -> float:
    """Aggregate per-gene raw scores without clipping away separation."""
    if len(raw_scores) == 0:
        return 0.0

    best = 0.0
    num_oscillating = 0
    for i in range(len(raw_scores)):
        raw_score = raw_scores[i]
        if raw_score > best:
            best = raw_score
        if report_scores[i] > multi_gene_threshold:
            num_oscillating += 1

    if best == 0.0:
        return 0.0

    multi_bonus = 0.0
    if num_oscillating > 1:
        multi_bonus = multi_gene_bonus_max * ((num_oscillating - 1) / 2.0)
    return best + multi_bonus


def score_single_gene_metrics(time: np.ndarray, raw_vals: np.ndarray) -> tuple[float, float, float]:
    """
    Score one gene's trajectory for oscillation quality.

    Returns:
      report_score in [0, 1]
      raw_score for optimisation (non-saturating in peak count)
      period estimate from median peak spacing
    """
    vals = np.asarray(raw_vals, dtype=np.float64)

    # ── Reject degenerate traces ──────────────────────────
    if _has_nan(vals) or _std_numba(vals) < cfg.FLAT_STD_THRESHOLD:
        return 0.0, 0.0, 0.0

    # ── Detrend to remove monotonic growth/decay ──────────
    detrended = _detrend_linear(vals)
    dt_std = _std_numba(detrended)
    if dt_std < cfg.FLAT_STD_THRESHOLD:
        return 0.0, 0.0, 0.0

    # ── Peak detection on detrended signal ────────────────
    height_thresh = _mean_numba(detrended) + cfg.PEAK_HEIGHT_FACTOR * dt_std
    min_dist = max(1, len(detrended) // cfg.PEAK_MIN_DISTANCE_FRAC)
    # Prominence: peak must rise at least 0.5*std above its nearest valleys
    min_prominence = 0.5 * dt_std
    peaks, _ = find_peaks(detrended, height=height_thresh,
                          distance=min_dist, prominence=min_prominence)
    peaks = np.asarray(peaks, dtype=np.int64)

    if len(peaks) < cfg.MIN_PEAKS:
        return 0.0, 0.0, 0.0

    return _score_from_peaks_numba(
        np.asarray(time, dtype=np.float64),
        vals,
        detrended,
        peaks,
        cfg.MIN_REL_AMPLITUDE,
        cfg.MIN_OSC_AMPLITUDE,
        cfg.MIN_AMP_TO_MEAN,
        float(cfg.PEAK_COUNT_FULL_CREDIT),
        cfg.W_SPACING_REGULARITY,
        cfg.W_AMPLITUDE_REGULARITY,
        cfg.W_PEAK_COUNT,
        cfg.W_PERSISTENCE,
    )


def score_single_gene(time: np.ndarray, raw_vals: np.ndarray) -> float:
    """Compatibility wrapper returning the reported score only."""
    report_score, _, _ = score_single_gene_metrics(time, raw_vals)
    return report_score


def score_trace_metrics(time: np.ndarray, concentrations: dict) -> dict:
    """
    Score a single SSA trace and return optimisation/reporting diagnostics.

    Returns:
      score: reported score in [0, 1]
      raw_score: non-saturating optimisation score
      period: period estimate from the strongest oscillating gene
    """
    report_scores = []
    raw_scores = []
    periods = []
    for gene, vals in concentrations.items():
        report_score, raw_score, period = score_single_gene_metrics(time, vals)
        report_scores.append(report_score)
        raw_scores.append(raw_score)
        periods.append(period)

    if not report_scores:
        return {"score": 0.0, "raw_score": 0.0, "period": 0.0}

    report_arr = np.asarray(report_scores, dtype=np.float64)
    raw_arr = np.asarray(raw_scores, dtype=np.float64)
    report_score = _aggregate_trace_score_numba(
        report_arr,
        cfg.MULTI_GENE_THRESHOLD,
        cfg.MULTI_GENE_BONUS_MAX,
    )
    raw_score = _aggregate_trace_raw_score_numba(
        report_arr,
        raw_arr,
        cfg.MULTI_GENE_THRESHOLD,
        cfg.MULTI_GENE_BONUS_MAX,
    )

    period = 0.0
    if report_score > 0.0:
        best_idx = int(np.argmax(report_arr))
        if report_arr[best_idx] > 0.0:
            period = float(periods[best_idx])

    return {
        "score": float(report_score),
        "raw_score": float(raw_score),
        "period": period,
    }


def score_trace(time: np.ndarray, concentrations: dict) -> float:
    """
    Score a single SSA simulation trace for oscillation quality.

    Args:
        time:           1-D array of time points.
        concentrations: dict mapping gene name -> 1-D array of copy numbers.

    Returns:
        Scalar score in [0, 1].
    """
    return score_trace_metrics(time, concentrations)["score"]


def score_trace_raw(time: np.ndarray, concentrations: dict) -> float:
    """Return the raw, non-saturating trace score used by fcmaes."""
    return score_trace_metrics(time, concentrations)["raw_score"]


def evaluate_topology_details(topology: Topology, params: np.ndarray,
                              n_seeds: int = cfg.INNER_N_SEEDS,
                              t_end: float = cfg.SIM_T_END,
                              seed_offset: int = cfg.TRAIN_SEED_OFFSET) -> dict:
    """
    Run multiple SSA simulations and return robust median diagnostics.
    """
    scores = []
    raw_scores = []
    periods = []
    for seed in range(n_seeds):
        try:
            model = build_model(topology, params, t_end=t_end)
            result = model.run(solver=cfg.SIM_SOLVER, seed=seed_offset + seed)
            time_arr = np.array(result["time"])
            conc = {g: np.array(result[g]) for g in cfg.GENES}
            metrics = score_trace_metrics(time_arr, conc)
            scores.append(metrics["score"])
            raw_scores.append(metrics["raw_score"])
            if metrics["period"] > 0.0:
                periods.append(metrics["period"])
        except Exception:
            scores.append(0.0)
            raw_scores.append(0.0)

    return {
        "score": float(np.median(scores)) if scores else 0.0,
        "raw_score": float(np.median(raw_scores)) if raw_scores else 0.0,
        "period": float(np.median(periods)) if periods else 0.0,
    }


def evaluate_topology(topology: Topology, params: np.ndarray,
                      n_seeds: int = cfg.INNER_N_SEEDS,
                      t_end: float = cfg.SIM_T_END,
                      seed_offset: int = cfg.TRAIN_SEED_OFFSET) -> float:
    """
    Compatibility wrapper returning the reported score only.
    """
    return evaluate_topology_details(
        topology,
        params,
        n_seeds=n_seeds,
        t_end=t_end,
        seed_offset=seed_offset,
    )["score"]


# ── Self-test ────────────────────────────────────────────────

if __name__ == "__main__":
    from grammar import REPRESSILATOR, TOGGLE_SWITCH_AB
    from model_builder import build_param_bounds

    rng = np.random.default_rng(7)

    print("=== Synthetic tests ===")
    print(f"Numba available: {NUMBA_AVAILABLE}")
    t = np.linspace(0, 100, 500)

    # Monotonic growth: MUST score 0
    mono = {"X": np.cumsum(rng.poisson(5, 500).astype(float))}
    print(f"  Monotonic growth: {score_trace(t, mono):.4f}  (expect 0)")

    # Clean sine oscillation: MUST score high
    osc = {"X": 50 + 30 * np.sin(2 * np.pi * t / 20) + rng.normal(0, 3, 500)}
    print(f"  Clean sine:       {score_trace(t, osc):.4f}  (expect >0.7)")

    # Flat line: MUST score 0
    flat = {"X": np.ones(500) * 50.0}
    print(f"  Flat line:        {score_trace(t, flat):.4f}  (expect 0)")

    # Noisy flat: MUST score 0
    noisy = {"X": 50 + rng.normal(0, 2, 500)}
    print(f"  Noisy flat:       {score_trace(t, noisy):.4f}  (expect 0)")

    print("\n=== Repressilator (GillesPy2 simulation) ===")
    topo = REPRESSILATOR
    lower, upper = build_param_bounds(topo)
    for trial in range(5):
        x = lower + rng.random(len(lower)) * (upper - lower)
        s = evaluate_topology(topo, x, n_seeds=2, t_end=200.0)
        print(f"  random params #{trial}: score = {s:.4f}")

    print("\n=== Toggle switch (should NOT oscillate) ===")
    topo2 = TOGGLE_SWITCH_AB
    lower2, upper2 = build_param_bounds(topo2)
    for trial in range(3):
        x = lower2 + rng.random(len(lower2)) * (upper2 - lower2)
        s = evaluate_topology(topo2, x, n_seeds=2, t_end=200.0)
        print(f"  random params #{trial}: score = {s:.4f}")
