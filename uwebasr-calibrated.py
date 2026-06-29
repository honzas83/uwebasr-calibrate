#!/usr/bin/env python3
# coding: utf-8

import os
import sys
import argparse
import logging
import subprocess
from queue import Queue
import threading
import time
import json
import urllib.request
import urllib.parse
import urllib.error
import http.cookiejar

# --- Sys Module Hack to allow loading CalibratedPredictor anywhere ---
import types
class CalibratedPredictor:
    def __init__(self, model, a, b, full_features=False):
        self.model = model
        self.a = a
        self.b = b
        self.full_features = full_features
        
    def predict(self, X):
        raw_pred = self.model.predict(X)
        calibrated_pred = self.a + self.b * raw_pred
        import numpy as np
        return np.clip(calibrated_pred, 0.0, 1.0)

mod = types.ModuleType("uwebasr_calibrate.model")
mod.CalibratedPredictor = CalibratedPredictor
sys.modules["uwebasr_calibrate.model"] = mod
sys.modules["uwebasr_calibrate"] = types.ModuleType("uwebasr_calibrate")
sys.modules["uwebasr_calibrate"].model = mod
# --------------------------------------------------------------------

UWEBASR_URL = "https://uwebasr.zcu.cz"
N_TRIES = 5

SPEECHCLOUD_JSON = "speechcloud_json"
FORMATS = {
    "json": SPEECHCLOUD_JSON, # special handling in format saving
    "txt": "plaintext",
    "s.txt": "plaintext&sp=0.3&pau=2.0",
    "vtt": "webvtt",
    "s.vtt": "sentvtt&sp=0.3&pau=2.0",
    "jsonl": "json",
}

parser = argparse.ArgumentParser(description='UWebASR client library with Accuracy Calibration')
parser.add_argument('model', metavar='MODEL', type=str, help='SpeechCloud app_id')
parser.add_argument('fns', metavar='FN', type=str, nargs="+", help='Input files')
parser.add_argument('--uwebasr-url', metavar='URL', type=str, default=UWEBASR_URL, help=f'UWEBASR_URL (default {UWEBASR_URL})')
parser.add_argument('--calibration-model', type=str, help="Path to the trained calibration model (model.joblib)")
parser.add_argument('--no-ffmpeg', action="store_true", help="Do not use ffmpeg, submit input files directly")
parser.add_argument('--no-cookies', action="store_true", help="Do not use cookies")
parser.add_argument('--overwrite', action="store_true", help="Allow overwrite of output files")
parser.add_argument('--output-dir', type=str, help="Optional output directory for saving output files")
parser.add_argument('--suffix', type=str, help="Optional suffix inserted after basename and before output file extension")
parser.add_argument('--n-workers', type=int, default=1, help="Number of parallel workers. Defaults to 1.")
parser.add_argument('--format', type=str, action="append", help="Generate only this format (can be used many times). Defaults to all formats.")

logger = logging.getLogger('uwebasr-calibrated')


# --- Standalone Feature Extraction Functions ---
def nearest_rank_percentile(sorted_values, q):
    import math
    threshold = (q / 100.0) * len(sorted_values)
    index = math.ceil(threshold) - 1
    index = max(0, min(index, len(sorted_values) - 1))
    return sorted_values[index]

def compute_run_lengths(mask):
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
    if current_state:
        blank_runs.append(current_len)
    else:
        nonblank_runs.append(current_len)
    return blank_runs, nonblank_runs

def extract_features(ctc_tokens, ctc_probs, ctc_frame_len=0.04, full_features=False):
    import math
    import numpy as np
    eps = 1e-9
    
    if len(ctc_tokens) != len(ctc_probs):
        raise ValueError("ctc_tokens and ctc_probs must have the same length")
    n_frames = len(ctc_tokens)
    if n_frames == 0:
        raise ValueError("Empty CTC stream")
        
    ctc_tokens = [str(tok) if tok is not None else "" for tok in ctc_tokens]
    probs_clipped = np.clip(ctc_probs, eps, 1.0 - eps)
    
    blank_mask = np.array([tok == "<blk>" for tok in ctc_tokens], dtype=bool)
    blank_values = probs_clipped[blank_mask]
    nonblank_values = probs_clipped[~blank_mask]
    
    if len(blank_values) == 0 or len(nonblank_values) == 0:
        raise ValueError("Missing blank or nonblank tokens in CTC slice")
        
    sorted_blank = np.sort(blank_values)
    sorted_nonblank = np.sort(nonblank_values)
    
    b_p000 = nearest_rank_percentile(sorted_blank, 0)
    b_p030 = nearest_rank_percentile(sorted_blank, 30)
    b_p040 = nearest_rank_percentile(sorted_blank, 40)
    b_p100 = nearest_rank_percentile(sorted_blank, 100)
    b_range = b_p100 - b_p000
    
    nb_p001 = nearest_rank_percentile(sorted_nonblank, 1)
    nb_p070 = nearest_rank_percentile(sorted_nonblank, 70)
    
    blank_deciles = {}
    for q in range(0, 110, 10):
        blank_deciles[f"ctc_blank_p{q:03d}"] = nearest_rank_percentile(sorted_blank, q)
        
    nonblank_deciles = {}
    for q in range(0, 110, 10):
        nonblank_deciles[f"ctc_nonblank_p{q:03d}"] = nearest_rank_percentile(sorted_nonblank, q)
    
    nb_geom_mean = np.exp(np.mean(np.log(nonblank_values)))
    nb_error_mean = np.mean(1.0 - nonblank_values)
    nb_harmonic_mean = len(nonblank_values) / np.sum(1.0 / nonblank_values)
    
    nonblank_errors = 1.0 - nonblank_values
    nb_error_geom_mean = np.exp(np.mean(np.log(np.maximum(eps, nonblank_errors))))
    nb_frac_lt_50 = np.mean(nonblank_values < 0.50)
    
    nonblank_thresholds = {}
    for th in range(10, 100, 10):
        nonblank_thresholds[f"ctc_nonblank_frac_lt_{th}"] = np.mean(nonblank_values < (th / 100.0))
    
    blank_errors = np.maximum(eps, 1.0 - blank_values)
    blank_neglog_errors = -np.log(blank_errors)
    b_neglog_error_p50 = np.percentile(blank_neglog_errors, 50, method="linear")
    
    blank_neglog_deciles = {}
    for q in range(0, 110, 10):
        blank_neglog_deciles[f"ctc_blank_neglog_error_p{q:03d}"] = np.percentile(blank_neglog_errors, q, method="linear")
    
    blank_count = len(blank_values)
    nonblank_count = len(nonblank_values)
    nb_to_b_ratio = nonblank_count / max(1, blank_count)
    b_log_ratio = np.log((blank_count + 1) / (nonblank_count + 1))
    
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
    
    blank_run_bounds = {}
    for L in range(1, 6):
        blank_run_bounds[f"ctc_blank_run_le_{L}"] = np.mean(np.array(blank_runs) <= L) if blank_runs else 0.0
        
    nonblank_run_bounds = {}
    for L in range(1, 6):
        nonblank_run_bounds[f"ctc_nonblank_run_le_{L}"] = np.mean(np.array(nonblank_runs) <= L) if nonblank_runs else 0.0
    
    collapsed_tokens = []
    collapsed_probs = []
    if len(ctc_tokens) > 0:
        curr_tok = ctc_tokens[0]
        curr_prob = probs_clipped[0]
        for tok, prob in zip(ctc_tokens[1:], probs_clipped[1:]):
            if tok == curr_tok:
                if prob > curr_prob:
                    curr_prob = prob
            else:
                collapsed_tokens.append(curr_tok)
                collapsed_probs.append(curr_prob)
                curr_tok = tok
                curr_prob = prob
        collapsed_tokens.append(curr_tok)
        collapsed_probs.append(curr_prob)

    nonblank_tokens = []
    nonblank_probs = []
    for tok, prob in zip(collapsed_tokens, collapsed_probs):
        if tok != "<blk>":
            nonblank_tokens.append(tok)
            nonblank_probs.append(prob)

    words_probs = []
    current_word_probs = []
    for tok, prob in zip(nonblank_tokens, nonblank_probs):
        if any(c in tok for c in ["▁", "\u2581", " "]):
            if current_word_probs:
                words_probs.append(current_word_probs)
                current_word_probs = []
        current_word_probs.append(prob)
    if current_word_probs:
        words_probs.append(current_word_probs)

    word_probabilities = [float(np.mean(w_probs)) for w_probs in words_probs]
    ctc_word_count = len(word_probabilities)
    ctc_word_prob_sum = sum(word_probabilities)

    nonblank_indices = [idx for idx, tok in enumerate(ctc_tokens) if tok != "<blk>"]
    if nonblank_indices:
        first_nonblank_index = nonblank_indices[0]
        last_nonblank_index = nonblank_indices[-1]
        speech_duration_seconds = (last_nonblank_index - first_nonblank_index) * ctc_frame_len
        speech_duration_seconds = max(ctc_frame_len, speech_duration_seconds)
    else:
        speech_duration_seconds = ctc_frame_len

    speech_duration_minutes = speech_duration_seconds / 60.0
    ctc_wpm = ctc_word_count / max(eps, speech_duration_minutes)

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
        "ctc_blank_max_run_fraction": b_max_run_frac,
        "ctc_token_count": float(n_frames),
        "ctc_prob_sum": float(np.sum(ctc_probs)),
        "ctc_word_count": float(ctc_word_count),
        "ctc_word_prob_sum": float(ctc_word_prob_sum),
        "ctc_wpm": float(ctc_wpm)
    }
    
    features.update(blank_deciles)
    features.update(nonblank_deciles)
    features.update(nonblank_thresholds)
    features.update(blank_run_bounds)
    features.update(nonblank_run_bounds)
    features.update(blank_neglog_deciles)
    
    for name, val in features.items():
        if not math.isfinite(val):
            raise ValueError(f"Feature '{name}' has non-finite value: {val}")
            
    FEATURE_ORDER_STANDARD = [
        "ctc_blank_mean_run_fraction", "ctc_nonblank_error_geom_mean", "ctc_nonblank_to_blank_ratio",
        "ctc_blank_p040", "ctc_nonblank_mean_run_fraction", "ctc_nonblank_harmonic_mean",
        "ctc_blank_range", "ctc_nonblank_p001", "ctc_blank_run_len_cv", "ctc_blank_log_ratio",
        "ctc_nonblank_error_mean", "ctc_blank_p030", "ctc_nonblank_p070", "ctc_nonblank_geom_mean",
        "ctc_nonblank_frac_lt_50", "ctc_nonblank_short_run_fraction", "ctc_blank_p000",
        "ctc_blank_short_run_fraction", "ctc_blank_neglog_error_p50", "ctc_blank_max_run_fraction",
        "ctc_token_count", "ctc_prob_sum", "ctc_word_count", "ctc_word_prob_sum", "ctc_wpm"
    ]

    FEATURE_ORDER_FULL = [
        "ctc_blank_mean_run_fraction", "ctc_nonblank_error_geom_mean", "ctc_nonblank_to_blank_ratio",
        "ctc_nonblank_mean_run_fraction", "ctc_nonblank_harmonic_mean", "ctc_blank_range",
        "ctc_blank_run_len_cv", "ctc_blank_log_ratio", "ctc_nonblank_error_mean", "ctc_nonblank_geom_mean",
        "ctc_blank_max_run_fraction", "ctc_token_count", "ctc_prob_sum", "ctc_word_count",
        "ctc_word_prob_sum", "ctc_wpm", "ctc_nonblank_p001", "ctc_blank_p000", "ctc_blank_p010",
        "ctc_blank_p020", "ctc_blank_p030", "ctc_blank_p040", "ctc_blank_p050", "ctc_blank_p060",
        "ctc_blank_p070", "ctc_blank_p080", "ctc_blank_p090", "ctc_blank_p100", "ctc_nonblank_p000",
        "ctc_nonblank_p010", "ctc_nonblank_p020", "ctc_nonblank_p030", "ctc_nonblank_p040",
        "ctc_nonblank_p050", "ctc_nonblank_p060", "ctc_nonblank_p070", "ctc_nonblank_p080",
        "ctc_nonblank_p090", "ctc_nonblank_p100", "ctc_nonblank_frac_lt_10", "ctc_nonblank_frac_lt_20",
        "ctc_nonblank_frac_lt_30", "ctc_nonblank_frac_lt_40", "ctc_nonblank_frac_lt_50",
        "ctc_nonblank_frac_lt_60", "ctc_nonblank_frac_lt_70", "ctc_nonblank_frac_lt_80",
        "ctc_nonblank_frac_lt_90", "ctc_blank_run_le_1", "ctc_blank_run_le_2", "ctc_blank_run_le_3",
        "ctc_blank_run_le_4", "ctc_blank_run_le_5", "ctc_nonblank_run_le_1", "ctc_nonblank_run_le_2",
        "ctc_nonblank_run_le_3", "ctc_nonblank_run_le_4", "ctc_nonblank_run_le_5",
        "ctc_blank_neglog_error_p000", "ctc_blank_neglog_error_p010", "ctc_blank_neglog_error_p020",
        "ctc_blank_neglog_error_p030", "ctc_blank_neglog_error_p040", "ctc_blank_neglog_error_p050",
        "ctc_blank_neglog_error_p060", "ctc_blank_neglog_error_p070", "ctc_blank_neglog_error_p080",
        "ctc_blank_neglog_error_p090", "ctc_blank_neglog_error_p100"
    ]
    
    order = FEATURE_ORDER_FULL if full_features else FEATURE_ORDER_STANDARD
    return [features[name] for name in order]
# ----------------------------------------------------


def get_model_url(model, uwebasr_url=None):
    if uwebasr_url is None:
        uwebasr_url = UWEBASR_URL
    return uwebasr_url+"/api/v2/"+model

def get_convert_url(uwebasr_url=None):
    if uwebasr_url is None:
        uwebasr_url = UWEBASR_URL
    return uwebasr_url+"/utils/v2/convert-speechcloud-json"

def recognize(model_url, fn, opener=None, no_ffmpeg=False):
    if no_ffmpeg:
        fr = open(fn, "rb")
        data = fr.read()
        fr.close()
    else:
        command = ["ffmpeg", "-xerror", "-hide_banner", "-loglevel", "error", "-i", fn, "-ar", "16000", "-ac", "1", "-vn", "-c:a", "libvorbis", "-q:a", "10", "-f", "ogg", "-"]
        ffmpeg = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=None)
        data = ffmpeg.stdout.read()
        ffmpeg.wait()

    url = model_url + "?format=speechcloud_json"
    req = urllib.request.Request(url, data=data, method='POST')
    
    if opener is None:
        opener = urllib.request.build_opener()

    try:
        with opener.open(req) as r:
            logger.info("Used SpeechCloud-SessionID: %s", r.headers.get("SpeechCloud-SessionID"))
            data_json = json.loads(r.read().decode('utf-8'))
            return data_json
    except urllib.error.HTTPError as e:
        logger.error("HTTP Error %s: %s", e.code, e.reason)
        raise

def convert(convert_url, data_json, format, opener=None):
    url = convert_url + "?format=" + format
    req = urllib.request.Request(url, data=json.dumps(data_json).encode('utf-8'), method='POST')
    req.add_header('Content-Type', 'application/json')

    if opener is None:
        opener = urllib.request.build_opener()

    with opener.open(req) as r:
        return r.read().decode('utf-8')

def _process_queue(model_url, convert_url, queue, predictor, cmdline_args):
    cj = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))

    while True:
        fn = queue.get()

        try:
            logger.info("Recognizing file: %s", fn)

            retry_count = 0
            while True:
                try:
                    if cmdline_args.no_cookies:
                        data_json = recognize(model_url, fn, no_ffmpeg=cmdline_args.no_ffmpeg)
                    else:
                        data_json = recognize(model_url, fn, opener=opener, no_ffmpeg=cmdline_args.no_ffmpeg)
                    break
                except urllib.error.HTTPError as e:
                    if e.code == 503:
                        retry_count += 1
                        if retry_count == N_TRIES:
                            raise
                        else:
                            time.sleep(1)
                            logger.info("    Trying again: %s", fn)
                            continue
                    else:
                        raise

            # Predict accuracy if predictor is specified
            accuracy_info = {
                "file_name": os.path.basename(fn),
                "estimated_accuracy": None,
                "has_ctc": False,
                "full_features": getattr(predictor, "full_features", False) if predictor else False,
                "features_count": None,
                "error": None
            }

            if predictor is not None:
                try:
                    res_list = None
                    if isinstance(data_json, dict) and "result" in data_json:
                        res_list = data_json["result"]
                    elif isinstance(data_json, list):
                        res_list = data_json
                    else:
                        raise ValueError("ASR output is not a JSON list or dict with 'result'")
                    
                    final_res = None
                    for res in res_list:
                        if isinstance(res, dict) and res.get("type") == "asr_result" and not res.get("partial_result", False):
                            final_res = res
                            break
                            
                    if final_res is None:
                        raise ValueError("No final asr_result found in UWebASR response")
                        
                    if "ctc_tokens" not in final_res or "ctc_probs" not in final_res:
                        raise ValueError("ASR output is missing ctc_tokens or ctc_probs streams")
                        
                    ctc_tokens = final_res["ctc_tokens"]
                    ctc_probs = final_res["ctc_probs"]
                    ctc_frame_len = final_res.get("ctc_frame_len", 0.04)
                    
                    # Extract local features
                    feats = extract_features(
                        ctc_tokens, ctc_probs, ctc_frame_len=ctc_frame_len,
                        full_features=predictor.full_features
                    )
                    
                    # Predict accuracy using the loaded HGBR model
                    import numpy as np
                    pred_acc = float(predictor.predict(np.array([feats]))[0])
                    
                    accuracy_info["estimated_accuracy"] = pred_acc
                    accuracy_info["has_ctc"] = True
                    accuracy_info["features_count"] = len(feats)
                    logger.info("--> Predicted accuracy for %s: %.4f", os.path.basename(fn), pred_acc)
                    
                    # Inject estimated accuracy into the JSON before output formatting
                    if isinstance(data_json, dict):
                        data_json["estimated_accuracy"] = pred_acc
                    final_res["estimated_accuracy"] = pred_acc
                except Exception as ex:
                    accuracy_info["error"] = str(ex)
                    logger.warning("--> Cannot estimate accuracy for %s: %s", os.path.basename(fn), ex)

            base_fn = os.path.splitext(fn)[0]

            for ext, format_val in FORMATS.items():
                if cmdline_args.format and ext not in cmdline_args.format:
                    continue

                if cmdline_args.output_dir:
                    base_fn = os.path.join(cmdline_args.output_dir, os.path.basename(base_fn))

                if not cmdline_args.suffix:
                    out_fn = base_fn+"."+ext
                else:
                    out_fn = base_fn+"."+cmdline_args.suffix+"."+ext

                if os.path.exists(out_fn) and not cmdline_args.overwrite:
                    logger.error("File already exists: %s, terminating... (use --overwrite to force file overwrite)", out_fn)
                    os._exit(-1)

                logger.info("Writing file %s (format %s)", out_fn, format_val)
                with open(out_fn, "w", encoding="utf-8") as fw:
                    if format_val == SPEECHCLOUD_JSON:
                        output = json.dumps(data_json, indent=4)
                    else:
                        output = convert(convert_url, data_json, format_val, opener=opener if not cmdline_args.no_cookies else None) 
                    fw.write(output)

            # Save separate calibrated metadata JSON file
            if predictor is not None:
                acc_fn = base_fn + ".accuracy.json"
                if cmdline_args.output_dir:
                    acc_fn = os.path.join(cmdline_args.output_dir, os.path.basename(acc_fn))
                if os.path.exists(acc_fn) and not cmdline_args.overwrite:
                    logger.error("Accuracy file already exists: %s, terminating...", acc_fn)
                    os._exit(-1)
                with open(acc_fn, "w", encoding="utf-8") as fw:
                    json.dump(accuracy_info, fw, indent=4)
                logger.info("Saved accuracy metadata to %s", acc_fn)

        except Exception as queue_err:
            logger.exception("Error while processing file: %s", fn)
        else:
            logger.info("Successfully recognized file: %s", fn)

        queue.task_done()


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s %(levelname)-10s %(message)s', level=logging.DEBUG)

    args = parser.parse_args()
    args.uwebasr_url = args.uwebasr_url.rstrip("/")

    predictor = None
    if args.calibration_model:
        try:
            import joblib
            predictor = joblib.load(args.calibration_model)
            logger.info("Loaded calibration model from %s. Expected features count: %s", 
                        args.calibration_model, 69 if getattr(predictor, "full_features", False) else 25)
        except Exception as e:
            logger.error("Failed to load calibration model: %s", e)
            sys.exit(1)

    model_url = get_model_url(args.model, args.uwebasr_url)
    convert_url = get_convert_url(args.uwebasr_url)

    logger.info("Using model: %s", model_url)

    file_queue = Queue()

    for idx in range(args.n_workers):
        threading.Thread(target=_process_queue, daemon=True, args=(model_url, convert_url, file_queue, predictor, args)).start()

    for fn in args.fns:
        file_queue.put(fn)

    logger.info("Waiting for processing of all files")
    file_queue.join()
    logger.info("All files processed")
