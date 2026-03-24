"""
Phenotype Evaluator — Oscillation Quality Scoring
===================================================

Given a GillesPy2 simulation result, compute a scalar score in [0, 1]
measuring oscillatory behavior quality.

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
"""

import numpy as np
from scipy.signal import find_peaks

from grammar import Topology
from model_builder import build_model
import config as cfg


def _detrend_linear(vals: np.ndarray) -> np.ndarray:
    """Remove linear trend from a signal."""
    n = len(vals)
    x = np.arange(n, dtype=np.float64)
    x_mean = x.mean()
    v_mean = vals.mean()
    slope = np.sum((x - x_mean) * (vals - v_mean)) / (np.sum((x - x_mean) ** 2) + 1e-12)
    trend = slope * x + (v_mean - slope * x_mean)
    return vals - trend


def _find_troughs_between_peaks(vals: np.ndarray, peaks: np.ndarray) -> np.ndarray:
    """For each consecutive pair of peaks, find the minimum value between them."""
    troughs = []
    for i in range(len(peaks) - 1):
        segment = vals[peaks[i]:peaks[i + 1] + 1]
        troughs.append(np.min(segment))
    return np.array(troughs)


def score_single_gene(time: np.ndarray, raw_vals: np.ndarray) -> float:
    """
    Score one gene's trajectory for oscillation quality.
    Returns float in [0, 1]. Returns 0 for non-oscillatory traces.
    """
    vals = np.asarray(raw_vals, dtype=np.float64)

    # ── Reject degenerate traces ──────────────────────────
    if np.any(np.isnan(vals)) or np.std(vals) < cfg.FLAT_STD_THRESHOLD:
        return 0.0

    # ── Detrend to remove monotonic growth/decay ──────────
    detrended = _detrend_linear(vals)
    dt_std = np.std(detrended)
    if dt_std < cfg.FLAT_STD_THRESHOLD:
        return 0.0

    # ── Peak detection on detrended signal ────────────────
    height_thresh = np.mean(detrended) + cfg.PEAK_HEIGHT_FACTOR * dt_std
    min_dist = max(1, len(detrended) // cfg.PEAK_MIN_DISTANCE_FRAC)
    # Prominence: peak must rise at least 0.5*std above its nearest valleys
    min_prominence = 0.5 * dt_std
    peaks, _ = find_peaks(detrended, height=height_thresh,
                          distance=min_dist, prominence=min_prominence)

    if len(peaks) < cfg.MIN_PEAKS:
        return 0.0

    # ── Trough depth check ────────────────────────────────
    peak_vals = detrended[peaks]
    troughs = _find_troughs_between_peaks(detrended, peaks)

    if len(troughs) == 0:
        return 0.0

    # Peak-to-trough amplitudes
    pt_amplitudes = []
    for i in range(len(troughs)):
        pt_amplitudes.append(peak_vals[i] - troughs[i])
    pt_amplitudes = np.array(pt_amplitudes)

    # Require median peak-to-trough amplitude >= 15% of signal range
    signal_range = np.max(detrended) - np.min(detrended)
    if signal_range < 1e-6:
        return 0.0
    median_pt = np.median(pt_amplitudes)
    median_rel_amplitude = median_pt / signal_range
    if median_rel_amplitude < cfg.MIN_REL_AMPLITUDE:
        return 0.0

    # Require absolute amplitude above noise floor
    if median_pt < cfg.MIN_OSC_AMPLITUDE:
        return 0.0

    # Require amplitude is meaningful relative to mean signal level
    mean_raw = np.mean(raw_vals)
    if mean_raw > 1.0 and median_pt / mean_raw < cfg.MIN_AMP_TO_MEAN:
        return 0.0

    # ── Spacing regularity ────────────────────────────────
    spacings = np.diff(time[peaks])
    spacing_cv = np.std(spacings) / (np.mean(spacings) + 1e-9)
    spacing_score = float(np.clip(1.0 - spacing_cv, 0.0, 1.0))

    # ── Amplitude regularity (peak-to-trough) ────────────
    amp_cv = np.std(pt_amplitudes) / (np.mean(pt_amplitudes) + 1e-9)
    amp_score = float(np.clip(1.0 - amp_cv, 0.0, 1.0))

    # ── Peak count score (saturates) ─────────────────────
    count_score = min(1.0, len(peaks) / cfg.PEAK_COUNT_FULL_CREDIT)

    # ── Persistence: oscillation span / total span ───────
    if len(peaks) >= 2:
        osc_span = time[peaks[-1]] - time[peaks[0]]
        total_span = time[-1] - time[0]
        persistence = osc_span / (total_span + 1e-9)
    else:
        persistence = 0.0

    # ── Weighted score ───────────────────────────────────
    gene_score = (
        cfg.W_SPACING_REGULARITY * spacing_score
        + cfg.W_AMPLITUDE_REGULARITY * amp_score
        + cfg.W_PEAK_COUNT * count_score
        + cfg.W_PERSISTENCE * persistence
    )
    return gene_score


def score_trace(time: np.ndarray, concentrations: dict) -> float:
    """
    Score a single SSA simulation trace for oscillation quality.

    Args:
        time:           1-D array of time points.
        concentrations: dict mapping gene name -> 1-D array of copy numbers.

    Returns:
        Scalar score in [0, 1].
    """
    gene_scores = []
    for gene, vals in concentrations.items():
        gene_scores.append(score_single_gene(time, vals))

    if not gene_scores or max(gene_scores) == 0.0:
        return 0.0

    best = max(gene_scores)
    num_oscillating = sum(1 for s in gene_scores if s > cfg.MULTI_GENE_THRESHOLD)
    multi_bonus = (
        cfg.MULTI_GENE_BONUS_MAX * min(1.0, (num_oscillating - 1) / 2.0)
        if num_oscillating > 1
        else 0.0
    )
    return float(np.clip(best + multi_bonus, 0.0, 1.0))


def evaluate_topology(topology: Topology, params: np.ndarray,
                      n_seeds: int = cfg.INNER_N_SEEDS,
                      t_end: float = cfg.SIM_T_END) -> float:
    """
    Run multiple SSA simulations and return a robust aggregate score.
    Uses the median across seeds to resist stochastic outliers.
    """
    scores = []
    for seed in range(n_seeds):
        try:
            model = build_model(topology, params, t_end=t_end)
            result = model.run(solver=cfg.SIM_SOLVER, seed=seed + 42)
            time_arr = np.array(result["time"])
            conc = {g: np.array(result[g]) for g in cfg.GENES}
            scores.append(score_trace(time_arr, conc))
        except Exception:
            scores.append(0.0)

    return float(np.median(scores)) if scores else 0.0


# ── Self-test ────────────────────────────────────────────────

if __name__ == "__main__":
    from grammar import REPRESSILATOR, TOGGLE_SWITCH_AB
    from model_builder import build_param_bounds

    rng = np.random.default_rng(7)

    print("=== Synthetic tests ===")
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
