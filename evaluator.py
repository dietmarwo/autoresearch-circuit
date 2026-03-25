"""
Phenotype Evaluator — Oscillation Quality Scoring
===================================================

Given a GillesPy2 simulation result, compute:
  - a reported score in [0, 1] for human-facing summaries
  - a raw, non-saturating score for fcmaes optimisation

For the circuit-search task we now target a harder phenotype than
"any decent oscillator": a coherent 3-gene traveling-wave oscillator.
That means the final trace score rewards:
  - at least one strong oscillating gene
  - participation of all three genes
  - a shared oscillation period across genes
  - separated phases rather than synchronized or toggle-like motion

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

The active experiment chooses how whole-trace scores are aggregated:
  - `oscillator3`: coherent 3-gene traveling wave
  - `robust5`: 5-gene oscillator with knockout, knockdown, and
    local-parameter robustness

The hot-path post-processing is numba-accelerated where possible:
  - linear detrending
  - trough and amplitude extraction
  - coefficient-of-variation calculations
  - final per-gene score computation

Peak detection still uses scipy's find_peaks to preserve the current
selection logic and thresholds.
"""

from itertools import combinations

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
from model_builder import build_model, build_param_bounds
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
def _normalized_autocorr_at_lag_numba(vals: np.ndarray, lag: int) -> float:
    """Compute a normalized autocorrelation on the overlapping window."""
    n = len(vals)
    if lag <= 0 or lag >= n:
        return 0.0

    num = 0.0
    den_left = 0.0
    den_right = 0.0
    stop = n - lag
    for i in range(stop):
        left = vals[i]
        right = vals[i + lag]
        num += left * right
        den_left += left * left
        den_right += right * right

    denom = np.sqrt(den_left * den_right) + 1e-12
    return num / denom


@njit(cache=True)
def _autocorr_periodicity_score_numba(detrended: np.ndarray, peaks: np.ndarray) -> float:
    """
    Reward repeated self-similarity at the inferred oscillation lag.

    We estimate the period in sample steps from the median peak spacing, then
    score the normalized autocorrelation at one and two periods. Persistent,
    regular oscillators keep both lags high; noisy or transient traces do not.
    """
    if len(peaks) < 2:
        return 0.0

    idx_spacings = np.empty(len(peaks) - 1, dtype=np.float64)
    for i in range(len(peaks) - 1):
        idx_spacings[i] = peaks[i + 1] - peaks[i]
    lag_steps = int(np.round(_median_numba(idx_spacings)))
    if lag_steps <= 0:
        return 0.0

    acf1 = _clip01(_normalized_autocorr_at_lag_numba(detrended, lag_steps))
    if 2 * lag_steps < len(detrended):
        acf2 = _clip01(_normalized_autocorr_at_lag_numba(detrended, 2 * lag_steps))
    else:
        acf2 = acf1
    return 0.6 * acf1 + 0.4 * acf2


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
    w_autocorr: float,
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
    autocorr_score = _autocorr_periodicity_score_numba(detrended, peaks)

    report_score = (
        w_spacing * spacing_score
        + w_amplitude * amp_score
        + w_count * count_score
        + w_persistence * persistence
        + w_autocorr * autocorr_score
    )
    raw_score = (
        w_spacing * spacing_score
        + w_amplitude * amp_score
        + w_count * count_score_raw
        + w_persistence * persistence
        + w_autocorr * autocorr_score
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


@njit(cache=True)
def _harmonic_mean_numba(vals: np.ndarray) -> float:
    """Compute a harmonic mean, returning 0 if any entry is non-positive."""
    n = len(vals)
    if n == 0:
        return 0.0
    inv_sum = 0.0
    for i in range(n):
        if vals[i] <= 0.0:
            return 0.0
        inv_sum += 1.0 / vals[i]
    return n / inv_sum


@njit(cache=True)
def _coherence_period_score_numba(periods: np.ndarray) -> float:
    """Reward shared period across genes via low coefficient of variation."""
    if len(periods) == 0:
        return 0.0
    mean_period = _mean_numba(periods)
    if mean_period <= 1e-9:
        return 0.0
    period_cv = _std_numba(periods) / (mean_period + 1e-9)
    return _clip01(1.0 - period_cv)


def _estimate_phase_from_peaks(peak_times: np.ndarray, period: float) -> float:
    """Estimate the dominant phase in [0, 1) from repeated peak times."""
    if period <= 1e-9 or len(peak_times) == 0:
        return 0.0
    phases = np.mod(peak_times, period) / period
    angles = 2.0 * np.pi * phases
    mean_sin = float(np.mean(np.sin(angles)))
    mean_cos = float(np.mean(np.cos(angles)))
    phase = np.arctan2(mean_sin, mean_cos) / (2.0 * np.pi)
    return float(np.mod(phase, 1.0))


def _phase_triplet_score(phases: np.ndarray) -> float:
    """
    Reward three evenly separated phases, independent of absolute label order.

    Perfectly phased 3-gene oscillators have circular gaps of roughly 1/3 period.
    Synchronous or nearly synchronous genes receive a score near zero.
    """
    if len(phases) != 3:
        return 0.0
    ordered = np.sort(np.mod(phases, 1.0))
    gaps = np.array([
        ordered[1] - ordered[0],
        ordered[2] - ordered[1],
        1.0 + ordered[0] - ordered[2],
    ], dtype=np.float64)
    gap_error = float(np.mean(np.abs(gaps - (1.0 / 3.0))))
    return float(np.clip(1.0 - gap_error / (1.0 / 3.0), 0.0, 1.0))


def _topk_harmonic_mean(vals: np.ndarray, k: int) -> float:
    """Harmonic mean of the k strongest values, or 0 if fewer than k are positive."""
    if len(vals) < k or k <= 0:
        return 0.0
    ordered = np.sort(np.asarray(vals, dtype=np.float64))
    top = ordered[-k:]
    return float(_harmonic_mean_numba(top))


def _support_fraction(vals: np.ndarray, threshold: float) -> float:
    """Fraction of values above a threshold."""
    if len(vals) == 0:
        return 0.0
    return float(np.mean(np.asarray(vals) >= threshold))


def _best_coherent_subset(gene_details: list[dict], subset_size: int) -> tuple[list[dict], float, float]:
    """
    Select the most coherent oscillating subset of the requested size.

    The subset score prefers strong per-gene oscillators with a shared period.
    """
    eligible = [
        detail for detail in gene_details
        if detail["score"] >= cfg.COHERENCE_GENE_THRESHOLD
        and detail["period"] > 0.0
        and len(detail["peak_times"]) > 0
    ]
    if len(eligible) < subset_size or subset_size <= 0:
        return [], 0.0, 0.0

    best_subset: list[dict] = []
    best_score = -1.0
    best_period_score = 0.0
    best_period = 0.0
    for combo in combinations(eligible, subset_size):
        periods = np.asarray([detail["period"] for detail in combo], dtype=np.float64)
        period_score = float(_coherence_period_score_numba(periods))
        participation = float(_harmonic_mean_numba(
            np.asarray([detail["score"] for detail in combo], dtype=np.float64)
        ))
        combo_score = participation * period_score
        if combo_score > best_score:
            best_score = combo_score
            best_subset = list(combo)
            best_period_score = period_score
            best_period = float(np.median(periods))
    return best_subset, best_period_score, best_period


def _score_single_gene_detail(time: np.ndarray, raw_vals: np.ndarray) -> dict:
    """
    Score one gene's trajectory for oscillation quality and retain peak times.
    """
    time_arr = np.asarray(time, dtype=np.float64)
    vals = np.asarray(raw_vals, dtype=np.float64)

    # ── Reject degenerate traces ──────────────────────────
    if _has_nan(vals) or _std_numba(vals) < cfg.FLAT_STD_THRESHOLD:
        return {"score": 0.0, "raw_score": 0.0, "period": 0.0,
                "peak_times": np.empty(0, dtype=np.float64)}

    # ── Detrend to remove monotonic growth/decay ──────────
    detrended = _detrend_linear(vals)
    dt_std = _std_numba(detrended)
    if dt_std < cfg.FLAT_STD_THRESHOLD:
        return {"score": 0.0, "raw_score": 0.0, "period": 0.0,
                "peak_times": np.empty(0, dtype=np.float64)}

    # ── Peak detection on detrended signal ────────────────
    height_thresh = _mean_numba(detrended) + cfg.PEAK_HEIGHT_FACTOR * dt_std
    min_dist = max(1, len(detrended) // cfg.PEAK_MIN_DISTANCE_FRAC)
    # Prominence: peak must rise at least 0.5*std above its nearest valleys
    min_prominence = 0.5 * dt_std
    peaks, _ = find_peaks(detrended, height=height_thresh,
                          distance=min_dist, prominence=min_prominence)
    peaks = np.asarray(peaks, dtype=np.int64)

    if len(peaks) < cfg.MIN_PEAKS:
        return {"score": 0.0, "raw_score": 0.0, "period": 0.0,
                "peak_times": np.empty(0, dtype=np.float64)}

    report_score, raw_score, period = _score_from_peaks_numba(
        time_arr,
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
        cfg.W_AUTOCORRELATION,
    )
    peak_times = time_arr[peaks] if report_score > 0.0 else np.empty(0, dtype=np.float64)
    return {
        "score": float(report_score),
        "raw_score": float(raw_score),
        "period": float(period),
        "peak_times": peak_times,
    }


def score_single_gene_metrics(time: np.ndarray, raw_vals: np.ndarray) -> tuple[float, float, float]:
    """
    Score one gene's trajectory for oscillation quality.

    Returns:
      report_score in [0, 1]
      raw_score for optimisation (non-saturating in peak count)
      period estimate from median peak spacing
    """
    detail = _score_single_gene_detail(time, raw_vals)
    return detail["score"], detail["raw_score"], detail["period"]


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
    gene_details = []
    for gene in cfg.GENES:
        if gene in concentrations:
            gene_details.append(_score_single_gene_detail(time, concentrations[gene]))
    if not gene_details:
        for _, vals in concentrations.items():
            gene_details.append(_score_single_gene_detail(time, vals))

    if not gene_details:
        return {"score": 0.0, "raw_score": 0.0, "period": 0.0}

    report_scores = [detail["score"] for detail in gene_details]
    raw_scores = [detail["raw_score"] for detail in gene_details]
    report_arr = np.asarray(report_scores, dtype=np.float64)
    raw_arr = np.asarray(raw_scores, dtype=np.float64)

    target_participants = min(cfg.TRACE_TARGET_PARTICIPANTS, len(gene_details))

    # Preserve the simple single-trace behavior when too few genes remain,
    # e.g. in tiny self-tests or aggressive knockout experiments.
    if target_participants < 2:
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
                period = float(gene_details[best_idx]["period"])
        return {
            "score": float(report_score),
            "raw_score": float(raw_score),
            "period": period,
        }

    best_report = float(np.max(report_arr))
    best_raw = float(np.max(raw_arr))
    participation_report = _topk_harmonic_mean(report_arr, target_participants)
    participation_raw = _topk_harmonic_mean(raw_arr, target_participants)
    support_score = _support_fraction(report_arr, cfg.COHERENCE_GENE_THRESHOLD)

    coherent_subset, period_score, period = _best_coherent_subset(
        gene_details,
        min(cfg.COHERENCE_TARGET_GENES, len(gene_details)),
    )
    phase_score = 0.0
    if (
        cfg.PHENOTYPE_MODE == "traveling_wave_3"
        and len(coherent_subset) == 3
        and cfg.TRACE_PHASE_COHERENCE_WEIGHT > 0.0
    ):
        phases = np.asarray(
            [_estimate_phase_from_peaks(detail["peak_times"], period)
             for detail in coherent_subset],
            dtype=np.float64,
        )
        phase_score = _phase_triplet_score(phases)

    report_score = _clip01(
        cfg.TRACE_BEST_GENE_WEIGHT * best_report
        + cfg.TRACE_PARTICIPATION_WEIGHT * participation_report
        + cfg.TRACE_PERIOD_COHERENCE_WEIGHT * period_score
        + cfg.TRACE_PHASE_COHERENCE_WEIGHT * phase_score
        + cfg.TRACE_SUPPORT_WEIGHT * support_score
    )
    raw_score = (
        cfg.TRACE_BEST_GENE_WEIGHT * best_raw
        + cfg.TRACE_PARTICIPATION_WEIGHT * participation_raw
        + cfg.TRACE_PERIOD_COHERENCE_WEIGHT * period_score
        + cfg.TRACE_PHASE_COHERENCE_WEIGHT * phase_score
        + cfg.TRACE_SUPPORT_WEIGHT * support_score
    )

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


def select_single_gene_sets(
    scenario_samples: int,
    seed_offset: int,
) -> list[tuple[int, ...]]:
    """Select single-gene perturbation scenarios for the current evaluation."""
    if scenario_samples == 0:
        return []
    if scenario_samples < 0 or scenario_samples >= cfg.NUM_GENES:
        return [(gene_idx,) for gene_idx in range(cfg.NUM_GENES)]
    rng = np.random.default_rng(seed_offset + 17_171)
    chosen = rng.choice(cfg.NUM_GENES, size=scenario_samples, replace=False)
    return [(int(gene_idx),) for gene_idx in chosen]


def select_knockout_gene_sets(
    knockout_samples: int,
    seed_offset: int,
) -> list[tuple[int, ...]]:
    """Select single-gene knockout scenarios for the current evaluation."""
    return select_single_gene_sets(knockout_samples, seed_offset)


def select_knockdown_gene_sets(
    knockdown_samples: int,
    seed_offset: int,
) -> list[tuple[int, ...]]:
    """Select single-gene partial-knockdown scenarios for the current evaluation."""
    return select_single_gene_sets(knockdown_samples, seed_offset + 3_003)


def select_param_perturbations(
    topology: Topology,
    params: np.ndarray,
    perturb_samples: int,
    seed_offset: int,
) -> list[np.ndarray]:
    """Sample local multiplicative parameter jitters around the current optimum."""
    if perturb_samples <= 0:
        return []
    lower, upper = build_param_bounds(topology)
    rng = np.random.default_rng(seed_offset + 31_313)
    base = np.asarray(params, dtype=np.float64)
    perturbations: list[np.ndarray] = []
    for _ in range(perturb_samples):
        multipliers = np.exp(rng.normal(0.0, cfg.PARAM_PERTURB_SIGMA, size=base.shape[0]))
        perturbations.append(np.clip(base * multipliers, lower, upper))
    return perturbations


def aggregate_scenario_metrics(
    scores_by_scenario: dict[object, list[float]],
    raw_scores_by_scenario: dict[object, list[float]],
) -> tuple[float | None, float | None, float | None]:
    """Reduce multiple stress scenarios into one lower-quantile robustness metric."""
    if not scores_by_scenario:
        return None, None, None
    medians = [
        float(np.median(scores_by_scenario[scenario]))
        for scenario in scores_by_scenario
    ]
    raw_medians = [
        float(np.median(raw_scores_by_scenario[scenario]))
        for scenario in raw_scores_by_scenario
    ]
    quantile = cfg.ROBUST_SCENARIO_AGGREGATION_QUANTILE
    score = float(np.quantile(np.asarray(medians, dtype=np.float64), quantile))
    raw_score = float(np.quantile(np.asarray(raw_medians, dtype=np.float64), quantile))
    pass_rate = float(np.mean(
        np.asarray(medians, dtype=np.float64) >= cfg.ROBUST_SUCCESS_THRESHOLD
    ))
    return score, raw_score, pass_rate


def simulate_trace_metrics(
    topology: Topology,
    params: np.ndarray,
    seed: int,
    t_end: float,
    knockout_genes: tuple[int, ...] = (),
    knockdown_genes: tuple[int, ...] = (),
) -> dict:
    """Build, simulate, and score one stochastic trace."""
    model = build_model(
        topology,
        params,
        t_end=t_end,
        knockout_genes=knockout_genes,
        knockdown_genes=knockdown_genes,
    )
    result = model.run(solver=cfg.SIM_SOLVER, seed=seed)
    time_arr = np.array(result["time"])
    concentrations = {
        gene: np.array(result[gene])
        for gi, gene in enumerate(cfg.GENES)
        if gi not in knockout_genes
    }
    return score_trace_metrics(time_arr, concentrations)


def evaluate_topology_details(topology: Topology, params: np.ndarray,
                              n_seeds: int = cfg.INNER_N_SEEDS,
                              t_end: float = cfg.SIM_T_END,
                              seed_offset: int = cfg.TRAIN_SEED_OFFSET,
                              knockout_samples: int = cfg.TRAIN_KNOCKOUT_SAMPLES,
                              knockdown_samples: int = cfg.TRAIN_KNOCKDOWN_SAMPLES,
                              param_perturb_samples: int = cfg.TRAIN_PARAM_PERTURB_SAMPLES,
                              batch_workers: int | None = None) -> dict:
    """
    Run multiple SSA simulations and return robust median diagnostics.
    """
    full_scores = []
    full_raw_scores = []
    periods = []
    knockout_sets = select_knockout_gene_sets(knockout_samples, seed_offset)
    knockdown_sets = select_knockdown_gene_sets(knockdown_samples, seed_offset)
    param_perturbations = select_param_perturbations(
        topology,
        params,
        param_perturb_samples,
        seed_offset,
    )
    knockout_scores_by_scenario: dict[tuple[int, ...], list[float]] = {
        knockout: [] for knockout in knockout_sets
    }
    knockout_raw_scores_by_scenario: dict[tuple[int, ...], list[float]] = {
        knockout: [] for knockout in knockout_sets
    }
    knockdown_scores_by_scenario: dict[tuple[int, ...], list[float]] = {
        knockdown: [] for knockdown in knockdown_sets
    }
    knockdown_raw_scores_by_scenario: dict[tuple[int, ...], list[float]] = {
        knockdown: [] for knockdown in knockdown_sets
    }
    perturb_scores_by_scenario: dict[int, list[float]] = {
        idx: [] for idx in range(len(param_perturbations))
    }
    perturb_raw_scores_by_scenario: dict[int, list[float]] = {
        idx: [] for idx in range(len(param_perturbations))
    }
    use_batch_perturb = (
        batch_workers is not None
        and batch_workers >= 2
        and len(param_perturbations) > 0
    )

    for seed in range(n_seeds):
        try:
            metrics = simulate_trace_metrics(
                topology,
                params,
                seed=seed_offset + seed,
                t_end=t_end,
            )
            full_scores.append(metrics["score"])
            full_raw_scores.append(metrics["raw_score"])
            if metrics["period"] > 0.0:
                periods.append(metrics["period"])

            for offset, knockout_genes in enumerate(knockout_sets, 1):
                knockout_metrics = simulate_trace_metrics(
                    topology,
                    params,
                    seed=seed_offset + seed + 10_000 * offset,
                    t_end=t_end,
                    knockout_genes=knockout_genes,
                )
                knockout_scores_by_scenario[knockout_genes].append(knockout_metrics["score"])
                knockout_raw_scores_by_scenario[knockout_genes].append(knockout_metrics["raw_score"])

            for offset, knockdown_genes in enumerate(knockdown_sets, 1):
                knockdown_metrics = simulate_trace_metrics(
                    topology,
                    params,
                    seed=seed_offset + seed + 20_000 * offset,
                    t_end=t_end,
                    knockdown_genes=knockdown_genes,
                )
                knockdown_scores_by_scenario[knockdown_genes].append(knockdown_metrics["score"])
                knockdown_raw_scores_by_scenario[knockdown_genes].append(knockdown_metrics["raw_score"])

            if not use_batch_perturb:
                for offset, perturbed_params in enumerate(param_perturbations, 1):
                    perturb_metrics = simulate_trace_metrics(
                        topology,
                        perturbed_params,
                        seed=seed_offset + seed + 30_000 * offset,
                        t_end=t_end,
                    )
                    perturb_scores_by_scenario[offset - 1].append(perturb_metrics["score"])
                    perturb_raw_scores_by_scenario[offset - 1].append(perturb_metrics["raw_score"])
        except Exception:
            full_scores.append(0.0)
            full_raw_scores.append(0.0)
            for knockout_genes in knockout_sets:
                knockout_scores_by_scenario[knockout_genes].append(0.0)
                knockout_raw_scores_by_scenario[knockout_genes].append(0.0)
            for knockdown_genes in knockdown_sets:
                knockdown_scores_by_scenario[knockdown_genes].append(0.0)
                knockdown_raw_scores_by_scenario[knockdown_genes].append(0.0)
            if not use_batch_perturb:
                for perturb_idx in perturb_scores_by_scenario:
                    perturb_scores_by_scenario[perturb_idx].append(0.0)
                    perturb_raw_scores_by_scenario[perturb_idx].append(0.0)

    if use_batch_perturb:
        try:
            # Import lazily to avoid a module cycle at import time. We batch
            # perturbation scenarios across the reusable fcmaes worker pool.
            from inner_optimizer import evaluate_params_batch

            perturb_scores = evaluate_params_batch(
                topology,
                param_perturbations,
                metric_name="score",
                n_workers=batch_workers,
                n_seeds=n_seeds,
                t_end=t_end,
                seed_offset=seed_offset + 30_000,
                knockout_samples=0,
                knockdown_samples=0,
                param_perturb_samples=0,
            )
            perturb_raw_scores = evaluate_params_batch(
                topology,
                param_perturbations,
                metric_name="raw_score",
                n_workers=batch_workers,
                n_seeds=n_seeds,
                t_end=t_end,
                seed_offset=seed_offset + 30_000,
                knockout_samples=0,
                knockdown_samples=0,
                param_perturb_samples=0,
            )
            for idx, score in enumerate(perturb_scores):
                perturb_scores_by_scenario[idx].append(float(score))
            for idx, raw_score in enumerate(perturb_raw_scores):
                perturb_raw_scores_by_scenario[idx].append(float(raw_score))
        except Exception:
            for perturb_idx in perturb_scores_by_scenario:
                perturb_scores_by_scenario[perturb_idx].append(0.0)
                perturb_raw_scores_by_scenario[perturb_idx].append(0.0)

    full_score = float(np.median(full_scores)) if full_scores else 0.0
    full_raw_score = float(np.median(full_raw_scores)) if full_raw_scores else 0.0

    knockout_score, knockout_raw_score, knockout_pass_rate = aggregate_scenario_metrics(
        knockout_scores_by_scenario,
        knockout_raw_scores_by_scenario,
    )
    knockdown_score, knockdown_raw_score, knockdown_pass_rate = aggregate_scenario_metrics(
        knockdown_scores_by_scenario,
        knockdown_raw_scores_by_scenario,
    )
    param_perturb_score, param_perturb_raw_score, param_perturb_pass_rate = aggregate_scenario_metrics(
        perturb_scores_by_scenario,
        perturb_raw_scores_by_scenario,
    )

    score_terms = [(cfg.ROBUST_FULL_WEIGHT, full_score)]
    raw_terms = [(cfg.ROBUST_FULL_WEIGHT, full_raw_score)]
    if knockout_score is not None and knockout_raw_score is not None:
        score_terms.append((cfg.ROBUST_KNOCKOUT_WEIGHT, knockout_score))
        raw_terms.append((cfg.ROBUST_KNOCKOUT_WEIGHT, knockout_raw_score))
    if knockout_pass_rate is not None:
        score_terms.append((cfg.ROBUST_KNOCKOUT_PASS_WEIGHT, knockout_pass_rate))
        raw_terms.append((cfg.ROBUST_KNOCKOUT_PASS_WEIGHT, knockout_pass_rate))
    if knockdown_score is not None and knockdown_raw_score is not None:
        score_terms.append((cfg.ROBUST_KNOCKDOWN_WEIGHT, knockdown_score))
        raw_terms.append((cfg.ROBUST_KNOCKDOWN_WEIGHT, knockdown_raw_score))
    if param_perturb_score is not None and param_perturb_raw_score is not None:
        score_terms.append((cfg.ROBUST_PARAM_PERTURB_WEIGHT, param_perturb_score))
        raw_terms.append((cfg.ROBUST_PARAM_PERTURB_WEIGHT, param_perturb_raw_score))

    score_weight = sum(weight for weight, _ in score_terms if weight > 0.0)
    raw_weight = sum(weight for weight, _ in raw_terms if weight > 0.0)
    overall_score = full_score
    overall_raw_score = full_raw_score
    if score_weight > 0.0:
        overall_score = float(np.clip(
            sum(weight * value for weight, value in score_terms if weight > 0.0) / score_weight,
            0.0,
            1.0,
        ))
    if raw_weight > 0.0:
        overall_raw_score = (
            sum(weight * value for weight, value in raw_terms if weight > 0.0) / raw_weight
        )

    return {
        "score": overall_score,
        "raw_score": overall_raw_score,
        "period": float(np.median(periods)) if periods else 0.0,
        "full_score": full_score,
        "full_raw_score": full_raw_score,
        "knockout_score": knockout_score,
        "knockout_raw_score": knockout_raw_score,
        "knockout_pass_rate": knockout_pass_rate,
        "knockdown_score": knockdown_score,
        "knockdown_raw_score": knockdown_raw_score,
        "knockdown_pass_rate": knockdown_pass_rate,
        "param_perturb_score": param_perturb_score,
        "param_perturb_raw_score": param_perturb_raw_score,
        "param_perturb_pass_rate": param_perturb_pass_rate,
    }


def evaluate_topology(topology: Topology, params: np.ndarray,
                      n_seeds: int = cfg.INNER_N_SEEDS,
                      t_end: float = cfg.SIM_T_END,
                      seed_offset: int = cfg.TRAIN_SEED_OFFSET,
                      knockout_samples: int = cfg.TRAIN_KNOCKOUT_SAMPLES,
                      knockdown_samples: int = cfg.TRAIN_KNOCKDOWN_SAMPLES,
                      param_perturb_samples: int = cfg.TRAIN_PARAM_PERTURB_SAMPLES,
                      batch_workers: int | None = None) -> float:
    """
    Compatibility wrapper returning the reported score only.
    """
    return evaluate_topology_details(
        topology,
        params,
        n_seeds=n_seeds,
        t_end=t_end,
        seed_offset=seed_offset,
        knockout_samples=knockout_samples,
        knockdown_samples=knockdown_samples,
        param_perturb_samples=param_perturb_samples,
        batch_workers=batch_workers,
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
