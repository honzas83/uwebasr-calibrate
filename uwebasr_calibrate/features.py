import math
import numpy as np

# Canonical order of features in the feature vector
FEATURE_ORDER = [
    "ctc_blank_mean_run_fraction",
    "ctc_nonblank_error_geom_mean",
    "ctc_nonblank_to_blank_ratio",
    "ctc_blank_p040",
    "ctc_nonblank_mean_run_fraction",
    "ctc_nonblank_harmonic_mean",
    "ctc_blank_range",
    "ctc_nonblank_p001",
    "ctc_blank_run_len_cv",
    "ctc_blank_log_ratio",
    "ctc_nonblank_error_mean",
    "ctc_blank_p030",
    "ctc_nonblank_p070",
    "ctc_nonblank_geom_mean",
    "ctc_nonblank_frac_lt_50",
    "ctc_nonblank_short_run_fraction",
    "ctc_blank_p000",
    "ctc_blank_short_run_fraction",
    "ctc_blank_neglog_error_p50",
    "ctc_blank_max_run_fraction"
]

def nearest_rank_percentile(sorted_values, q):
    """
    Nearest-rank percentile implementation as specified:
    threshold = (q / 100) * len(sorted_values)
    index = ceil(threshold) - 1
    index = clamp(index, 0, len(sorted_values) - 1)
    """
    threshold = (q / 100.0) * len(sorted_values)
    index = math.ceil(threshold) - 1
    index = max(0, min(index, len(sorted_values) - 1))
    return sorted_values[index]

def compute_run_lengths(mask):
    """
    Computes lengths of maximal contiguous runs of True and False values in a boolean mask.
    """
    if len(mask) == 0:
        return [], []
        
    blank_runs = []
    nonblank_runs = []
    
    current_state = mask[0]
    current_len = 1
    
    for val in mask[1:]:
        if val == current_state:
            current_len += 1
        else:
            if current_state:
                blank_runs.append(current_len)
            else:
                nonblank_runs.append(current_len)
            current_state = val
            current_len = 1
            
    # Append the last run
    if current_state:
        blank_runs.append(current_len)
    else:
        nonblank_runs.append(current_len)
        
    return blank_runs, nonblank_runs

def extract_features(ctc_tokens, ctc_probs):
    """
    Extracts the top 20 CTC confidence features from the ctc_tokens and ctc_probs streams.
    Raises ValueError if inputs are invalid or empty, or if NaN/Inf features are produced.
    """
    eps = 1e-9
    
    if len(ctc_tokens) != len(ctc_probs):
        raise ValueError("ctc_tokens and ctc_probs must have the same length")
        
    n_frames = len(ctc_tokens)
    if n_frames == 0:
        raise ValueError("Empty CTC stream")
        
    # Clip probabilities
    probs_clipped = np.clip(ctc_probs, eps, 1.0 - eps)
    
    # Identify blank mask and separate values
    blank_mask = np.array([tok == "<blk>" for tok in ctc_tokens], dtype=bool)
    blank_values = probs_clipped[blank_mask]
    nonblank_values = probs_clipped[~blank_mask]
    
    if len(blank_values) == 0 or len(nonblank_values) == 0:
        raise ValueError("Missing blank or nonblank tokens in CTC slice")
        
    sorted_blank = np.sort(blank_values)
    sorted_nonblank = np.sort(nonblank_values)
    
    # 1. nearest-rank percentiles
    b_p000 = nearest_rank_percentile(sorted_blank, 0)
    b_p030 = nearest_rank_percentile(sorted_blank, 30)
    b_p040 = nearest_rank_percentile(sorted_blank, 40)
    b_p100 = nearest_rank_percentile(sorted_blank, 100)
    b_range = b_p100 - b_p000
    
    nb_p001 = nearest_rank_percentile(sorted_nonblank, 1)
    nb_p070 = nearest_rank_percentile(sorted_nonblank, 70)
    
    # 2. distribution stats
    nb_geom_mean = np.exp(np.mean(np.log(nonblank_values)))
    nb_error_mean = np.mean(1.0 - nonblank_values)
    
    # Harmonic mean
    nb_harmonic_mean = len(nonblank_values) / np.sum(1.0 / nonblank_values)
    
    # Nonblank error geom mean
    nonblank_errors = 1.0 - nonblank_values
    nb_error_geom_mean = np.exp(np.mean(np.log(np.maximum(eps, nonblank_errors))))
    
    nb_frac_lt_50 = np.mean(nonblank_values < 0.50)
    
    # Blank neglog error p50 (uses numpy linear quantile)
    blank_errors = np.maximum(eps, 1.0 - blank_values)
    blank_neglog_errors = -np.log(blank_errors)
    b_neglog_error_p50 = np.percentile(blank_neglog_errors, 50, method="linear")
    
    # 3. count-ratios
    blank_count = len(blank_values)
    nonblank_count = len(nonblank_values)
    nb_to_b_ratio = nonblank_count / max(1, blank_count)
    b_log_ratio = np.log((blank_count + 1) / (nonblank_count + 1))
    
    # 4. run-structure features
    blank_runs, nonblank_runs = compute_run_lengths(blank_mask)
    
    b_mean_run_frac = np.mean(blank_runs) / n_frames if blank_runs else 0.0
    nb_mean_run_frac = np.mean(nonblank_runs) / n_frames if nonblank_runs else 0.0
    b_max_run_frac = np.max(blank_runs) / n_frames if blank_runs else 0.0
    
    if blank_runs:
        b_mean_run = np.mean(blank_runs)
        b_run_len_cv = (np.std(blank_runs, ddof=0) / b_mean_run) if b_mean_run > 0 else 0.0
    else:
        b_run_len_cv = 0.0
        
    b_short_run_frac = np.mean(np.array(blank_runs) <= 2) if blank_runs else 0.0
    nb_short_run_frac = np.mean(np.array(nonblank_runs) <= 2) if nonblank_runs else 0.0
    
    features = {
        "ctc_blank_mean_run_fraction": b_mean_run_frac,
        "ctc_nonblank_error_geom_mean": nb_error_geom_mean,
        "ctc_nonblank_to_blank_ratio": nb_to_b_ratio,
        "ctc_blank_p040": b_p040,
        "ctc_nonblank_mean_run_fraction": nb_mean_run_frac,
        "ctc_nonblank_harmonic_mean": nb_harmonic_mean,
        "ctc_blank_range": b_range,
        "ctc_nonblank_p001": nb_p001,
        "ctc_blank_run_len_cv": b_run_len_cv,
        "ctc_blank_log_ratio": b_log_ratio,
        "ctc_nonblank_error_mean": nb_error_mean,
        "ctc_blank_p030": b_p030,
        "ctc_nonblank_p070": nb_p070,
        "ctc_nonblank_geom_mean": nb_geom_mean,
        "ctc_nonblank_frac_lt_50": nb_frac_lt_50,
        "ctc_nonblank_short_run_fraction": nb_short_run_frac,
        "ctc_blank_p000": b_p000,
        "ctc_blank_short_run_fraction": b_short_run_frac,
        "ctc_blank_neglog_error_p50": b_neglog_error_p50,
        "ctc_blank_max_run_fraction": b_max_run_frac
    }
    
    # Check for NaN or Inf
    for name, val in features.items():
        if not math.isfinite(val):
            raise ValueError(f"Feature '{name}' has non-finite value: {val}")
            
    return [features[name] for name in FEATURE_ORDER]
