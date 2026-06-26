import os
import math
import sys
import json
import logging
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
import jiwer

from uwebasr_calibrate.normalizer import normalize_text, align_ref_and_hyp
from uwebasr_calibrate.asr import prepare_url, run_recognition_single, validate_asr_result
from uwebasr_calibrate.data import (
    load_manifest, split_dataset, run_segmentation,
    generate_ensemble_samples, get_test_real_windows,
    normalize_text as data_normalize_text
)
from uwebasr_calibrate.features import extract_features, FEATURE_ORDER
from uwebasr_calibrate.model import train_calibration_model, CalibratedPredictor

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("uwebasr-calibrate")

import requests
import threading

thread_local = threading.local()

def get_session():
    if not hasattr(thread_local, "session"):
        thread_local.session = requests.Session()
    return thread_local.session

def process_utterance(row, url, cache_dir, timeout, retries):
    """
    Checks cache for recognized result. If missing, requests endpoint and caches.
    Returns (utt_id, success, data_or_error_msg).
    """
    utt_id = row["utt_id"]
    cache_file = Path(cache_dir) / f"{utt_id}.json"
    
    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Basic validation of the cache file structure
            if "result" in data:
                validate_asr_result(data["result"])
                return utt_id, True, data
        except Exception as e:
            logger.warning(f"Cache file {cache_file} is invalid: {e}. Re-recognizing...")
            
    # Run recognition
    session = get_session()
    try:
        snr = row.get("snr")
        raw_result = run_recognition_single(
            session=session,
            url=url,
            audio_path=row["audio_path"],
            timeout_seconds=timeout,
            retries=retries,
            snr=snr,
            utt_id=row.get("original_utt_id") or utt_id
        )
        
        # Validate result
        validate_asr_result(raw_result)
        
        # Cache
        cache_data = {
            "utt_id": utt_id,
            "audio_path": row["audio_path"],
            "reference": row["reference"],
            "endpoint_url": url,
            "result_format": "speechcloud_json",
            "result": raw_result
        }
        if snr is not None:
            cache_data["snr"] = snr
            cache_data["original_utt_id"] = row.get("original_utt_id")
            
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
            
        return utt_id, True, cache_data
        
    except Exception as e:
        return utt_id, False, str(e)

def safe_pearsonr(x, y):
    if len(x) < 2:
        return None
    if np.var(x) == 0.0 or np.var(y) == 0.0:
        return None
    from scipy.stats import pearsonr
    corr, _ = pearsonr(x, y)
    return float(corr) if np.isfinite(corr) else None

def get_dataset_label(dataset_paths, index):
    path = Path(dataset_paths[index])
    name = path.parent.name
    if not name or name == ".":
        name = path.stem
    names = []
    for p in dataset_paths:
        p_path = Path(p)
        p_name = p_path.parent.name
        if not p_name or p_name == ".":
            p_name = p_path.stem
        names.append(p_name)
    if name and names.count(name) == 1:
        return name
    return f"dataset_{index}"

def evaluate_windowed_predictions(windows_by_key, predictor, split_name, split_window_name):
    window_records = []
    agg_records = []
    
    for key, windows in windows_by_key.items():
        agg_ref_words_total = 0
        agg_true_acc_weighted_sum = 0.0
        agg_est_acc_weighted_sum = 0.0
        n_windows = len(windows)
        
        for w_idx, win in enumerate(windows):
            win_tokens = []
            win_probs = []
            win_errors = 0
            win_ref_words = 0
            
            for chunk in win:
                win_tokens.extend(chunk["ctc_tokens"])
                win_probs.extend(chunk["ctc_probs"])
                win_errors += chunk["edit_errors"]
                win_ref_words += chunk["reference_words"]
                
            if win_ref_words == 0:
                continue
                
            win_true_acc = max(0.0, 1.0 - win_errors / win_ref_words)
            
            try:
                win_features = extract_features(win_tokens, win_probs)
                win_est_acc = float(predictor.predict([win_features])[0])
            except Exception as e:
                logger.warning(f"Failed to extract features or predict for window: {e}. Using fallback prediction 0.0")
                win_est_acc = 0.0
                
            window_records.append({
                "sample_id": f"{key}_w{w_idx}",
                "split": split_window_name,
                "accuracy": win_true_acc,
                "estimated_accuracy": win_est_acc,
                "residual": win_true_acc - win_est_acc,
                "ref_words": win_ref_words
            })
            
            agg_ref_words_total += win_ref_words
            agg_true_acc_weighted_sum += win_ref_words * win_true_acc
            agg_est_acc_weighted_sum += win_ref_words * win_est_acc
            
        if agg_ref_words_total > 0:
            agg_true_acc = agg_true_acc_weighted_sum / agg_ref_words_total
            agg_est_acc = agg_est_acc_weighted_sum / agg_ref_words_total
            
            agg_records.append({
                "sample_id": key,
                "split": split_name,
                "accuracy": agg_true_acc,
                "estimated_accuracy": agg_est_acc,
                "residual": agg_true_acc - agg_est_acc,
                "ref_words": agg_ref_words_total,
                "n_windows": n_windows
            })
            
    return pd.DataFrame(window_records), pd.DataFrame(agg_records)

def compute_overall_accuracy(rows, df_utt_metrics):
    utt_ids = {r["utt_id"] for r in rows}
    df_filtered = df_utt_metrics[df_utt_metrics["utt_id"].isin(utt_ids)]
    if df_filtered.empty:
        return 0.0, 0
    total_ref = int(df_filtered["reference_words"].sum())
    total_err = int(df_filtered["edit_errors"].sum())
    acc = max(0.0, 1.0 - total_err / total_ref) if total_ref > 0 else 0.0
    return acc, len(df_filtered)

def run_calibration_workflow(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cache_dir = output_dir / "asr_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    model_dir = output_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure logging file
    log_file = output_dir / "calibration.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(file_handler)
    
    try:
        logger.info(f"Loaded configuration. Output directory: {output_dir}")
        
        # 2. Load manifest files and map to URLs
        n_datasets = len(args.dataset)
        all_rows = []
        utt_to_url = {}
        
        for i in range(n_datasets):
            dataset_path = args.dataset[i]
            # Parse and prepare URL
            try:
                url = prepare_url(args.uwebasr_url[i])
                logger.info(f"Loading dataset {i+1}/{n_datasets}: {dataset_path} with endpoint {url}")
            except ValueError as e:
                logger.error(f"Invalid URL configuration at index {i}: {e}")
                raise e
                
            try:
                rows = load_manifest(dataset_path, skip_bad_rows=args.skip_bad_rows)
                logger.info(f"Loaded {len(rows)} utterances from dataset {i+1}.")
                for r in rows:
                    utt_id = r["utt_id"]
                    if utt_id in utt_to_url:
                        # De-duplicate utterance IDs across datasets if they collide
                        r["utt_id"] = f"{utt_id}_ds{i}"
                        utt_id = r["utt_id"]
                    utt_to_url[utt_id] = url
                    r["dataset_idx"] = i
                    all_rows.append(r)
            except Exception as e:
                logger.error(f"Failed to load manifest {dataset_path}: {e}")
                raise e
                
        if args.limit is not None:
            logger.info(f"Limiting execution to first {args.limit} utterances for debugging.")
            all_rows = all_rows[:args.limit]
            
        # Filter rows to only those with valid normalized reference words (before running ASR/splitting)
        filtered_rows = []
        for r in all_rows:
            ref_words = normalize_text(r["reference"])
            if len(ref_words) == 0:
                logger.warning(f"Utterance {r['utt_id']} has 0 normalized reference words. Skipping from calibration.")
            else:
                filtered_rows.append(r)
        
        # 5. Split rows by speaker/group
        from uwebasr_calibrate.data import get_speaker_id, get_group_id
        
        try:
            has_explicit_split = any("split" in r and r["split"] in ["train", "test"] for r in filtered_rows)
            if has_explicit_split:
                logger.info("Using pre-defined split from 'split' column in the manifest.")
                train_rows = [r for r in filtered_rows if r.get("split") == "train"]
                test_rows = [r for r in filtered_rows if r.get("split") == "test"]
                
                for r in train_rows:
                    r["speaker_id"] = get_speaker_id(r["utt_id"], r.get("speaker_id"))
                    r["group_id"] = get_group_id(r["utt_id"], r.get("group_id"))
                for r in test_rows:
                    r["speaker_id"] = get_speaker_id(r["utt_id"], r.get("speaker_id"))
                    r["group_id"] = get_group_id(r["utt_id"], r.get("group_id"))
                    
                train_speakers = list(set(r["speaker_id"] for r in train_rows if r["speaker_id"] is not None))
                test_speakers = list(set(r["speaker_id"] for r in test_rows if r["speaker_id"] is not None))
                
                logger.info(f"Split verified. Train: {len(train_rows)} utterances. Test: {len(test_rows)} utterances.")
            else:
                train_rows, test_rows, train_speakers, test_speakers = split_dataset(
                    filtered_rows, train_fraction=0.8, seed=args.seed, split_group=args.split_group
                )
                logger.info(f"Split completed. Train: {len(train_rows)} utterances. Test: {len(test_rows)} utterances.")
                if args.split_group == "speaker":
                    logger.info(f"Train speakers: {len(train_speakers)}. Test speakers: {len(test_speakers)}")
        except Exception as e:
            logger.error(f"Dataset split failed: {e}")
            raise e
            
        # Data Augmentation with Additive Noise
        augmented_train_rows = []
        augmented_test_rows = []
        if args.snr:
            logger.info(f"Augmenting training data with SNR levels: {args.snr}")
            for r in train_rows:
                for snr_val in args.snr:
                    aug_r = r.copy()
                    snr_str = f"{int(snr_val)}" if snr_val == int(snr_val) else f"{snr_val}"
                    aug_r["original_utt_id"] = r["utt_id"]
                    aug_r["utt_id"] = f"{r['utt_id']}_snr{snr_str}"
                    aug_r["snr"] = snr_val
                    if r.get("group_id"):
                        aug_r["group_id"] = f"{r['group_id']}_snr{snr_str}"
                    if r.get("speaker_id"):
                        aug_r["speaker_id"] = f"{r['speaker_id']}_snr{snr_str}"
                    augmented_train_rows.append(aug_r)
            logger.info(f"Generated {len(augmented_train_rows)} augmented training utterances.")
            
            logger.info(f"Augmenting test data with SNR levels: {args.snr}")
            for r in test_rows:
                for snr_val in args.snr:
                    aug_r = r.copy()
                    snr_str = f"{int(snr_val)}" if snr_val == int(snr_val) else f"{snr_val}"
                    aug_r["original_utt_id"] = r["utt_id"]
                    aug_r["utt_id"] = f"{r['utt_id']}_snr{snr_str}"
                    aug_r["snr"] = snr_val
                    if r.get("group_id"):
                        aug_r["group_id"] = f"{r['group_id']}_snr{snr_str}"
                    if r.get("speaker_id"):
                        aug_r["speaker_id"] = f"{r['speaker_id']}_snr{snr_str}"
                    augmented_test_rows.append(aug_r)
            logger.info(f"Generated {len(augmented_test_rows)} augmented test utterances.")
            
        all_to_process = train_rows + augmented_train_rows + test_rows + augmented_test_rows
        
        # Build utt_to_url mapping for all items to process
        for r in augmented_train_rows + augmented_test_rows:
            utt_to_url[r["utt_id"]] = utt_to_url[r["original_utt_id"]]
            
        # 3. Recognition (Resumable)
        asr_results = {}
        failed_recognition = []
        
        logger.info(f"Running ASR recognition with {args.jobs} jobs...")
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            futures = {
                executor.submit(process_utterance, row, utt_to_url[row["utt_id"]], cache_dir, 90, 7): row
                for row in all_to_process
            }
            total_utts = len(futures)
            completed_count = 0
            log_interval = 1 if total_utts <= 20 else (5 if total_utts <= 100 else 10)
            for fut in as_completed(futures):
                utt_id, success, data = fut.result()
                completed_count += 1
                if success:
                    asr_results[utt_id] = data
                    if completed_count % log_interval == 0 or completed_count == total_utts:
                        logger.info(f"ASR progress: {completed_count}/{total_utts} utterances processed.")
                else:
                    failed_recognition.append((utt_id, data))
                    logger.error(f"Recognition failed for {utt_id}: {data}")
                    
        if failed_recognition:
            logger.error(f"{len(failed_recognition)} utterances failed recognition. Stopping workflow.")
            raise RuntimeError(f"{len(failed_recognition)} utterances failed recognition.")
            
        logger.info("All utterances successfully recognized and verified.")
        
        # 4. Compute normalized reference/hypothesis metrics & save utterance_metrics.csv
        utt_metrics = []
        for row in all_to_process:
            utt_id = row["utt_id"]
            asr_data = asr_results[utt_id]
            asr_result = validate_asr_result(asr_data["result"])
            
            ref_words = normalize_text(row["reference"])
            hyp_words = normalize_text(" ".join(asr_result["word_array"]))
            
            # Word counts
            ref_words_count = len(ref_words)
            hyp_words_count = len(hyp_words)
            
            if ref_words_count == 0:
                logger.warning(f"Utterance {utt_id} has 0 normalized reference words. Skipping from calibration.")
                continue
                
            # Edit errors
            out = jiwer.process_words(" ".join(ref_words), " ".join(hyp_words))
            edit_errors = out.substitutions + out.insertions + out.deletions
            accuracy = max(0.0, 1.0 - edit_errors / ref_words_count)
            
            utt_metrics.append({
                "utt_id": utt_id,
                "speaker_id": row.get("speaker_id") or "unknown",
                "group_id": row.get("group_id") or utt_id,
                "audio_path": row["audio_path"],
                "reference_words": ref_words_count,
                "hypothesis_words": hyp_words_count,
                "edit_errors": edit_errors,
                "accuracy": accuracy
            })
            
        df_utt_metrics = pd.DataFrame(utt_metrics)
        df_utt_metrics.to_csv(output_dir / "utterance_metrics.csv", index=False)
        logger.info("Saved utterance_metrics.csv")
        
        # Ensure only successfully recognized rows are used
        valid_utt_ids = set(df_utt_metrics["utt_id"])
        
        train_rows_clean = [r for r in train_rows if r["utt_id"] in valid_utt_ids]
        augmented_train_rows = [r for r in augmented_train_rows if r["utt_id"] in valid_utt_ids]
        test_rows_clean = [r for r in test_rows if r["utt_id"] in valid_utt_ids]
        augmented_test_rows = [r for r in augmented_test_rows if r["utt_id"] in valid_utt_ids]
        
        # Logging ASR Accuracy Summary
        logger.info("--- ASR Accuracy Summary ---")
        
        # Clean train and test
        acc_tr, n_utts_tr = compute_overall_accuracy(train_rows_clean, df_utt_metrics)
        logger.info(f"Original (clean) - train: Acc={acc_tr:.5f} ({n_utts_tr} utts)")
        
        acc_te, n_utts_te = compute_overall_accuracy(test_rows_clean, df_utt_metrics)
        logger.info(f"Original (clean) - test: Acc={acc_te:.5f} ({n_utts_te} utts)")
        
        # Augmented train and test for each SNR
        if args.snr:
            for snr_val in args.snr:
                snr_train = [r for r in augmented_train_rows if r.get("snr") == snr_val]
                acc_snr_tr, n_utts_snr_tr = compute_overall_accuracy(snr_train, df_utt_metrics)
                logger.info(f"Augmented (SNR={snr_val}) - train: Acc={acc_snr_tr:.5f} ({n_utts_snr_tr} utts)")
                
                snr_test = [r for r in augmented_test_rows if r.get("snr") == snr_val]
                acc_snr_te, n_utts_snr_te = compute_overall_accuracy(snr_test, df_utt_metrics)
                logger.info(f"Augmented (SNR={snr_val}) - test: Acc={acc_snr_te:.5f} ({n_utts_snr_te} utts)")
                
        logger.info("----------------------------")
        
        # Combine train_rows and test_rows for training/testing
        train_rows = train_rows_clean + augmented_train_rows
        test_rows = test_rows_clean + augmented_test_rows
        filtered_rows = train_rows + test_rows
            
        # 6. Balanced Word-aligned Segmentation per dataset
        logger.info(f"Running balanced segmentation targeting {args.target_segments} segments per dataset/language...")
        train_segments = []
        test_segments = []
        dataset_variants = {}
        
        for i in range(n_datasets):
            ds_filtered_rows = [r for r in filtered_rows if r.get("dataset_idx") == i]
            ds_train_rows = [r for r in train_rows if r.get("dataset_idx") == i]
            ds_test_rows = [r for r in test_rows if r.get("dataset_idx") == i]
            
            if not ds_filtered_rows:
                logger.warning(f"Dataset {i+1} has no valid rows after filtering. Skipping segmentation.")
                continue
                
            # Pre-segmentation pass for this dataset
            ds_pre_segments = run_segmentation(ds_filtered_rows, asr_results, seed=args.seed, variant_index=0)
            ds_estimated_per_variant = len(ds_pre_segments)
            
            if ds_estimated_per_variant == 0:
                logger.warning(f"Dataset {i+1} pre-segmentation produced 0 segments. Skipping.")
                continue
                
            ds_variants = max(1, math.ceil(args.target_segments / ds_estimated_per_variant))
            dataset_variants[i] = ds_variants
            
            logger.info(
                f"Dataset {i+1} ({args.dataset[i]}): "
                f"Pre-segmentation produced {ds_estimated_per_variant} segments/variant. "
                f"Target is {args.target_segments} segments. Computed variant count: {ds_variants}"
            )
            
            # Generate actual segments for this dataset
            for v in range(ds_variants):
                train_segments.extend(run_segmentation(ds_train_rows, asr_results, seed=args.seed, variant_index=v))
                test_segments.extend(run_segmentation(ds_test_rows, asr_results, seed=args.seed, variant_index=v))
                
        logger.info(f"Total segment count: train_segments={len(train_segments)}, test_segments={len(test_segments)}")
        
        # Save segments.csv
        segments_records = []
        for s in train_segments:
            segments_records.append({
                "segment_id": f"{s['utt_id']}_t_{s['ref_start']}_{s['ref_end']}",
                "utt_id": s["utt_id"],
                "split": "train",
                "ref_start": s["ref_start"],
                "ref_end": s["ref_end"],
                "reference_words": s["reference_words"],
                "edit_errors": s["edit_errors"],
                "accuracy": s["accuracy"]
            })
        for s in test_segments:
            segments_records.append({
                "segment_id": f"{s['utt_id']}_e_{s['ref_start']}_{s['ref_end']}",
                "utt_id": s["utt_id"],
                "split": "test",
                "ref_start": s["ref_start"],
                "ref_end": s["ref_end"],
                "reference_words": s["reference_words"],
                "edit_errors": s["edit_errors"],
                "accuracy": s["accuracy"]
            })
        df_segments = pd.DataFrame(segments_records)
        df_segments.to_csv(output_dir / "segments.csv", index=False)
        logger.info("Saved segments.csv")
        
        # 7. Generate ensemble samples
        logger.info("Generating ensemble samples...")
        train_samples, train_deciles_count = generate_ensemble_samples(
            train_segments, args.target_ensemble, seed=args.seed, n_jobs=args.jobs,
            min_words=args.ensemble_min_words, max_words=args.ensemble_max_words, min_segments=args.ensemble_min_segments
        )
        test_samples, test_deciles_count = generate_ensemble_samples(
            test_segments, args.target_ensemble // 4, seed=args.seed, n_jobs=args.jobs,
            min_words=args.ensemble_min_words, max_words=args.ensemble_max_words, min_segments=args.ensemble_min_segments
        )
        
        logger.info("Extracted features for ensemble samples.")
        
        # 8. Train HGBR models & fit affine calibration
        logger.info("Training models...")
        predictor, best_params, best_val_score, val_pred_calib, y_val = train_calibration_model(
            train_samples, seed=args.seed, loss_metric=args.loss
        )
        
        # Save trained predictor
        joblib.dump(predictor, model_dir / "model.joblib")
        logger.info("Saved model.joblib")
        
        # 9. Evaluate validation
        val_mae = float(np.mean(np.abs(y_val - val_pred_calib)))
        val_mse = float(np.mean((y_val - val_pred_calib) ** 2))
        val_corr = safe_pearsonr(y_val, val_pred_calib)
        logger.info(f"Validation results: MAE={val_mae:.5f}, MSE={val_mse:.5f}, corr={val_corr}")
        
        # Save validation predictions
        val_records = []
        for idx, (act, est) in enumerate(zip(y_val, val_pred_calib)):
            val_records.append({
                "sample_id": f"val_{idx}",
                "split": "validation",
                "accuracy": act,
                "estimated_accuracy": est,
                "residual": act - est,
                "ref_words": 512 # Approx, since validation split comes from train ensemble
            })
        df_val_preds = pd.DataFrame(val_records)
        df_val_preds.to_csv(output_dir / "predictions.validation.csv", index=False)
        
        # 10. Evaluate test
        X_test = np.array([s["features"] for s in test_samples])
        y_test = np.array([s["accuracy"] for s in test_samples])
        test_pred_calib = predictor.predict(X_test)
        
        test_mae = float(np.mean(np.abs(y_test - test_pred_calib)))
        test_mse = float(np.mean((y_test - test_pred_calib) ** 2))
        test_corr = safe_pearsonr(y_test, test_pred_calib)
        logger.info(f"Test results: MAE={test_mae:.5f}, MSE={test_mse:.5f}, corr={test_corr}")
        
        test_records = []
        for idx, (act, est, s) in enumerate(zip(y_test, test_pred_calib, test_samples)):
            test_records.append({
                "sample_id": s["sample_id"],
                "split": "test",
                "accuracy": act,
                "estimated_accuracy": est,
                "residual": act - est,
                "ref_words": s["ref_words"]
            })
        df_test_preds = pd.DataFrame(test_records)
        df_test_preds.to_csv(output_dir / "predictions.test.csv", index=False)
        
        # 11. Evaluate test_real (Windowed prediction grouped by group_id, clean only)
        logger.info("Running test_real windowed evaluation (grouped by group_id, clean only)...")
        test_real_window_size = (args.ensemble_min_words + args.ensemble_max_words) // 2
        windows_by_group = get_test_real_windows(test_rows_clean, asr_results, window_size=test_real_window_size, group_key="group_id")
        
        df_test_real_window, df_test_real_group = evaluate_windowed_predictions(
            windows_by_group, predictor, "test_real", "test_real_window"
        )
        
        df_test_real_window.to_csv(output_dir / "predictions.test_real_window.csv", index=False)
        df_test_real_group.to_csv(output_dir / "predictions.test_real.csv", index=False)
        
        # Compute test_real metrics
        if not df_test_real_group.empty:
            test_real_mae = float(np.mean(np.abs(df_test_real_group["accuracy"] - df_test_real_group["estimated_accuracy"])))
            test_real_mse = float(np.mean((df_test_real_group["accuracy"] - df_test_real_group["estimated_accuracy"]) ** 2))
            test_real_corr = safe_pearsonr(df_test_real_group["accuracy"], df_test_real_group["estimated_accuracy"])
        else:
            test_real_mae = 0.0
            test_real_mse = 0.0
            test_real_corr = None
            
        logger.info(f"Test_real results: MAE={test_real_mae:.5f}, MSE={test_real_mse:.5f}, corr={test_real_corr}")
        
        # 11b. Evaluate test_real_part (Windowed prediction on individual utterances, ungrouped, clean only)
        logger.info("Running test_real_part windowed evaluation (ungrouped, clean only)...")
        windows_by_utt = get_test_real_windows(test_rows_clean, asr_results, window_size=test_real_window_size, group_key="utt_id")
        
        df_test_real_part_window, df_test_real_part_utt = evaluate_windowed_predictions(
            windows_by_utt, predictor, "test_real_part", "test_real_part_window"
        )
        
        df_test_real_part_window.to_csv(output_dir / "predictions.test_real_part_window.csv", index=False)
        df_test_real_part_utt.to_csv(output_dir / "predictions.test_real_part.csv", index=False)
        
        # Compute test_real_part metrics
        if not df_test_real_part_utt.empty:
            test_real_part_mae = float(np.mean(np.abs(df_test_real_part_utt["accuracy"] - df_test_real_part_utt["estimated_accuracy"])))
            test_real_part_mse = float(np.mean((df_test_real_part_utt["accuracy"] - df_test_real_part_utt["estimated_accuracy"]) ** 2))
            test_real_part_corr = safe_pearsonr(df_test_real_part_utt["accuracy"], df_test_real_part_utt["estimated_accuracy"])
        else:
            test_real_part_mae = 0.0
            test_real_part_mse = 0.0
            test_real_part_corr = None
            
        logger.info(f"Test_real_part results: MAE={test_real_part_mae:.5f}, MSE={test_real_part_mse:.5f}, corr={test_real_part_corr}")

        # 11c. Evaluate test_real_snr (Windowed prediction grouped by group_id, including SNR)
        logger.info("Running test_real_snr windowed evaluation (grouped by group_id, including SNR)...")
        windows_by_group_snr = get_test_real_windows(test_rows, asr_results, window_size=test_real_window_size, group_key="group_id")
        
        df_test_real_snr_window, df_test_real_snr_group = evaluate_windowed_predictions(
            windows_by_group_snr, predictor, "test_real_snr", "test_real_snr_window"
        )
        
        df_test_real_snr_window.to_csv(output_dir / "predictions.test_real_snr_window.csv", index=False)
        df_test_real_snr_group.to_csv(output_dir / "predictions.test_real_snr.csv", index=False)
        
        # Compute test_real_snr metrics
        if not df_test_real_snr_group.empty:
            test_real_snr_mae = float(np.mean(np.abs(df_test_real_snr_group["accuracy"] - df_test_real_snr_group["estimated_accuracy"])))
            test_real_snr_mse = float(np.mean((df_test_real_snr_group["accuracy"] - df_test_real_snr_group["estimated_accuracy"]) ** 2))
            test_real_snr_corr = safe_pearsonr(df_test_real_snr_group["accuracy"], df_test_real_snr_group["estimated_accuracy"])
        else:
            test_real_snr_mae = 0.0
            test_real_snr_mse = 0.0
            test_real_snr_corr = None
            
        logger.info(f"Test_real_snr results: MAE={test_real_snr_mae:.5f}, MSE={test_real_snr_mse:.5f}, corr={test_real_snr_corr}")
        
        # 11d. Evaluate test_real_part_snr (Windowed prediction on individual utterances, ungrouped, including SNR)
        logger.info("Running test_real_part_snr windowed evaluation (ungrouped, including SNR)...")
        windows_by_utt_snr = get_test_real_windows(test_rows, asr_results, window_size=test_real_window_size, group_key="utt_id")
        
        df_test_real_part_snr_window, df_test_real_part_snr_utt = evaluate_windowed_predictions(
            windows_by_utt_snr, predictor, "test_real_part_snr", "test_real_part_snr_window"
        )
        
        df_test_real_part_snr_window.to_csv(output_dir / "predictions.test_real_part_snr_window.csv", index=False)
        df_test_real_part_snr_utt.to_csv(output_dir / "predictions.test_real_part_snr.csv", index=False)
        
        # Compute test_real_part_snr metrics
        if not df_test_real_part_snr_utt.empty:
            test_real_part_snr_mae = float(np.mean(np.abs(df_test_real_part_snr_utt["accuracy"] - df_test_real_part_snr_utt["estimated_accuracy"])))
            test_real_part_snr_mse = float(np.mean((df_test_real_part_snr_utt["accuracy"] - df_test_real_part_snr_utt["estimated_accuracy"]) ** 2))
            test_real_part_snr_corr = safe_pearsonr(df_test_real_part_snr_utt["accuracy"], df_test_real_part_snr_utt["estimated_accuracy"])
        else:
            test_real_part_snr_mae = 0.0
            test_real_part_snr_mse = 0.0
            test_real_part_snr_corr = None
            
        logger.info(f"Test_real_part_snr results: MAE={test_real_part_snr_mae:.5f}, MSE={test_real_part_snr_mse:.5f}, corr={test_real_part_snr_corr}")
        
        # 11e. Evaluate test_real_snr_{S} and test_real_part_snr_{S} for each SNR
        df_test_real_snr_val_group_dict = {}
        df_test_real_snr_val_window_dict = {}
        df_test_real_part_snr_val_utt_dict = {}
        df_test_real_part_snr_val_window_dict = {}
        
        snr_evals = {}
        if args.snr:
            for snr_val in args.snr:
                snr_str = f"{int(snr_val)}" if snr_val == int(snr_val) else f"{snr_val}"
                logger.info(f"Running test_real_snr_{snr_str} windowed evaluation...")
                test_rows_s = test_rows_clean + [r for r in augmented_test_rows if r.get("snr") == snr_val]
                
                # group level
                windows_by_group_s = get_test_real_windows(test_rows_s, asr_results, window_size=test_real_window_size, group_key="group_id")
                df_window_s, df_group_s = evaluate_windowed_predictions(
                    windows_by_group_s, predictor, f"test_real_snr_{snr_str}", f"test_real_snr_{snr_str}_window"
                )
                df_window_s.to_csv(output_dir / f"predictions.test_real_snr_{snr_str}_window.csv", index=False)
                df_group_s.to_csv(output_dir / f"predictions.test_real_snr_{snr_str}.csv", index=False)
                
                df_test_real_snr_val_group_dict[snr_val] = df_group_s
                df_test_real_snr_val_window_dict[snr_val] = df_window_s
                
                if not df_group_s.empty:
                    mae_s = float(np.mean(np.abs(df_group_s["accuracy"] - df_group_s["estimated_accuracy"])))
                    mse_s = float(np.mean((df_group_s["accuracy"] - df_group_s["estimated_accuracy"]) ** 2))
                    corr_s = safe_pearsonr(df_group_s["accuracy"], df_group_s["estimated_accuracy"])
                else:
                    mae_s = 0.0
                    mse_s = 0.0
                    corr_s = None
                
                # part level
                logger.info(f"Running test_real_part_snr_{snr_str} windowed evaluation...")
                windows_by_utt_s = get_test_real_windows(test_rows_s, asr_results, window_size=test_real_window_size, group_key="utt_id")
                df_part_window_s, df_part_utt_s = evaluate_windowed_predictions(
                    windows_by_utt_s, predictor, f"test_real_part_snr_{snr_str}", f"test_real_part_snr_{snr_str}_window"
                )
                df_part_window_s.to_csv(output_dir / f"predictions.test_real_part_snr_{snr_str}_window.csv", index=False)
                df_part_utt_s.to_csv(output_dir / f"predictions.test_real_part_snr_{snr_str}.csv", index=False)
                
                df_test_real_part_snr_val_utt_dict[snr_val] = df_part_utt_s
                df_test_real_part_snr_val_window_dict[snr_val] = df_part_window_s
                
                if not df_part_utt_s.empty:
                    part_mae_s = float(np.mean(np.abs(df_part_utt_s["accuracy"] - df_part_utt_s["estimated_accuracy"])))
                    part_mse_s = float(np.mean((df_part_utt_s["accuracy"] - df_part_utt_s["estimated_accuracy"]) ** 2))
                    part_corr_s = safe_pearsonr(df_part_utt_s["accuracy"], df_part_utt_s["estimated_accuracy"])
                else:
                    part_mae_s = 0.0
                    part_mse_s = 0.0
                    part_corr_s = None
                
                snr_evals[snr_val] = {
                    "snr_str": snr_str,
                    "df_group": df_group_s,
                    "df_utt": df_part_utt_s,
                    "group_mae": mae_s,
                    "group_mse": mse_s,
                    "group_corr": corr_s,
                    "utt_mae": part_mae_s,
                    "utt_mse": part_mse_s,
                    "utt_corr": part_corr_s
                }

        # 12. Save features.csv (for inspection/reproducibility)
        # Build list of feature rows
        feat_rows = []
        # Train/validation ensemble samples
        for s in train_samples:
            feat_rows.append([s["sample_id"], "train", s["accuracy"]] + s["features"])
        # Test ensemble samples
        for s in test_samples:
            feat_rows.append([s["sample_id"], "test", s["accuracy"]] + s["features"])
        # Test_real windows
        # Actually, we can just save features for train and test ensemble samples to keep features.csv size manageable (~80k rows, 23 columns).
            
        # Actually, we can just save features for train and test ensemble samples to keep features.csv size manageable (~80k rows, 23 columns).
        df_features = pd.DataFrame(
            feat_rows,
            columns=["sample_id", "split", "accuracy"] + FEATURE_ORDER
        )
        df_features.to_csv(output_dir / "features.csv", index=False)
        logger.info("Saved features.csv")
        
        # 13. Create plots
        logger.info("Creating scatter plots...")
        
        # Validation scatter plot
        plt.figure(figsize=(6, 6))
        plt.scatter(df_val_preds["accuracy"], df_val_preds["estimated_accuracy"], alpha=0.3, color="blue")
        plt.plot([0, 1], [0, 1], color="red", linestyle="--")
        plt.xlabel("True Accuracy")
        plt.ylabel("Estimated Accuracy")
        plt.title(f"Validation Scatter (MAE={val_mae:.5f}, MSE={val_mse:.5f})")
        plt.xlim(-0.05, 1.05)
        plt.ylim(-0.05, 1.05)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plots_dir / "scatter_validation.png")
        plt.close()
        
        # Test scatter plot
        plt.figure(figsize=(6, 6))
        plt.scatter(df_test_preds["accuracy"], df_test_preds["estimated_accuracy"], alpha=0.3, color="green")
        plt.plot([0, 1], [0, 1], color="red", linestyle="--")
        plt.xlabel("True Accuracy")
        plt.ylabel("Estimated Accuracy")
        plt.title(f"Test Scatter (MAE={test_mae:.5f}, MSE={test_mse:.5f})")
        plt.xlim(-0.05, 1.05)
        plt.ylim(-0.05, 1.05)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plots_dir / "scatter_test.png")
        plt.close()
        
        # Test_real scatter plot
        plt.figure(figsize=(6, 6))
        if not df_test_real_group.empty:
            plt.scatter(df_test_real_group["accuracy"], df_test_real_group["estimated_accuracy"], alpha=0.7, color="purple")
        plt.plot([0, 1], [0, 1], color="red", linestyle="--")
        plt.xlabel("True Accuracy")
        plt.ylabel("Estimated Accuracy")
        plt.title(f"Test_real Scatter (MSE={test_real_mse:.5f})" if df_test_real_group.empty else f"Test_real Group Scatter (MAE={test_real_mae:.5f}, MSE={test_real_mse:.5f})")
        plt.xlim(-0.05, 1.05)
        plt.ylim(-0.05, 1.05)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plots_dir / "scatter_test_real.png")
        plt.close()
        
        # Test_real_part scatter plot
        plt.figure(figsize=(6, 6))
        if not df_test_real_part_utt.empty:
            plt.scatter(df_test_real_part_utt["accuracy"], df_test_real_part_utt["estimated_accuracy"], alpha=0.5, color="orange")
        plt.plot([0, 1], [0, 1], color="red", linestyle="--")
        plt.xlabel("True Accuracy")
        plt.ylabel("Estimated Accuracy")
        plt.title(f"Test_real_part Scatter (MSE={test_real_part_mse:.5f})" if df_test_real_part_utt.empty else f"Test_real_part Scatter (MAE={test_real_part_mae:.5f}, MSE={test_real_part_mse:.5f})")
        plt.xlim(-0.05, 1.05)
        plt.ylim(-0.05, 1.05)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plots_dir / "scatter_test_real_part.png")
        plt.close()
        
        # Test_real_snr scatter plot
        plt.figure(figsize=(6, 6))
        if not df_test_real_snr_group.empty:
            plt.scatter(df_test_real_snr_group["accuracy"], df_test_real_snr_group["estimated_accuracy"], alpha=0.7, color="teal")
        plt.plot([0, 1], [0, 1], color="red", linestyle="--")
        plt.xlabel("True Accuracy")
        plt.ylabel("Estimated Accuracy")
        plt.title(f"Test_real_snr Scatter (MSE={test_real_snr_mse:.5f})" if df_test_real_snr_group.empty else f"Test_real_snr Group Scatter (MAE={test_real_snr_mae:.5f}, MSE={test_real_snr_mse:.5f})")
        plt.xlim(-0.05, 1.05)
        plt.ylim(-0.05, 1.05)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plots_dir / "scatter_test_real_snr.png")
        plt.close()
        
        # Test_real_part_snr scatter plot
        plt.figure(figsize=(6, 6))
        if not df_test_real_part_snr_utt.empty:
            plt.scatter(df_test_real_part_snr_utt["accuracy"], df_test_real_part_snr_utt["estimated_accuracy"], alpha=0.5, color="brown")
        plt.plot([0, 1], [0, 1], color="red", linestyle="--")
        plt.xlabel("True Accuracy")
        plt.ylabel("Estimated Accuracy")
        plt.title(f"Test_real_part_snr Scatter (MSE={test_real_part_snr_mse:.5f})" if df_test_real_part_snr_utt.empty else f"Test_real_part_snr Scatter (MAE={test_real_part_snr_mae:.5f}, MSE={test_real_part_snr_mse:.5f})")
        plt.xlim(-0.05, 1.05)
        plt.ylim(-0.05, 1.05)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plots_dir / "scatter_test_real_part_snr.png")
        plt.close()
        
        # Test_real_snr_{S} and Test_real_part_snr_{S} scatter plots
        if args.snr:
            for snr_val, info in snr_evals.items():
                snr_str = info["snr_str"]
                df_group_s = info["df_group"]
                df_utt_s = info["df_utt"]
                
                plt.figure(figsize=(6, 6))
                if not df_group_s.empty:
                    plt.scatter(df_group_s["accuracy"], df_group_s["estimated_accuracy"], alpha=0.7, color="teal")
                plt.plot([0, 1], [0, 1], color="red", linestyle="--")
                plt.xlabel("True Accuracy")
                plt.ylabel("Estimated Accuracy")
                plt.title(f"Test_real_snr_{snr_str} Scatter (MSE={info['group_mse']:.5f})" if df_group_s.empty else f"Test_real_snr_{snr_str} Group Scatter (MAE={info['group_mae']:.5f}, MSE={info['group_mse']:.5f})")
                plt.xlim(-0.05, 1.05)
                plt.ylim(-0.05, 1.05)
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(plots_dir / f"scatter_test_real_snr_{snr_str}.png")
                plt.close()
                
                plt.figure(figsize=(6, 6))
                if not df_utt_s.empty:
                    plt.scatter(df_utt_s["accuracy"], df_utt_s["estimated_accuracy"], alpha=0.5, color="brown")
                plt.plot([0, 1], [0, 1], color="red", linestyle="--")
                plt.xlabel("True Accuracy")
                plt.ylabel("Estimated Accuracy")
                plt.title(f"Test_real_part_snr_{snr_str} Scatter (MSE={info['utt_mse']:.5f})" if df_utt_s.empty else f"Test_real_part_snr_{snr_str} Scatter (MAE={info['utt_mae']:.5f}, MSE={info['utt_mse']:.5f})")
                plt.xlim(-0.05, 1.05)
                plt.ylim(-0.05, 1.05)
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(plots_dir / f"scatter_test_real_part_snr_{snr_str}.png")
                plt.close()
        
        logger.info("Saved scatter plots.")
        
        # 14. Save metadata.json
        metadata = {
            "endpoint_url": args.uwebasr_url,
            "feature_list": FEATURE_ORDER,
            "feature_version": "1.0",
            "normalizer_version": "default_calibration_normalizer",
            "normalizer_regex": r"[^\W_]+(?:['’][^\W_]+)?",
            "jiwer_version": jiwer.__version__ if hasattr(jiwer, "__version__") else "unknown",
            "training_configuration": {
                "seed": args.seed,
                "target_segments": args.target_segments,
                "split_group": args.split_group,
                "loss": args.loss,
                "train_fraction": 0.8,
                "ensemble_train_size": 64000,
                "ensemble_test_size": 16000,
                "ensemble_min_words": args.ensemble_min_words,
                "ensemble_max_words": args.ensemble_max_words,
                "ensemble_min_segments": args.ensemble_min_segments,
                "segment_variants": dataset_variants
            },
            "selected_hyperparameters": best_params,
            "affine_calibration": {
                "a": predictor.a,
                "b": predictor.b
            },
            "software_versions": {
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scikit-learn": "1.3.0" # or package import version
            },
            "validation_score": val_mae
        }
        with open(model_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
            
        # 15. Save config.json
        config = {
            "dataset": args.dataset,
            "uwebasr_url": args.uwebasr_url,
            "output_dir": args.output_dir,
            "target_segments": args.target_segments,
            "target_ensemble": args.target_ensemble,
            "jobs": args.jobs,
            "seed": args.seed,
            "split_group": args.split_group,
            "loss": args.loss,
            "ensemble_min_words": args.ensemble_min_words,
            "ensemble_max_words": args.ensemble_max_words,
            "ensemble_min_segments": args.ensemble_min_segments,
            "skip_bad_rows": args.skip_bad_rows,
            "snr": args.snr,
            "train_speakers": train_speakers,
            "test_speakers": test_speakers,
            "normalizer_version": "default_calibration_normalizer",
            "feature_version": "1.0",
            "packages": {
                "jiwer": jiwer.__version__ if hasattr(jiwer, "__version__") else "unknown",
                "numpy": np.__version__,
                "pandas": pd.__version__
            }
        }
        with open(output_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
            
        # 16. Save metrics.json
        metrics = {
            "validation": {
                "MAE": val_mae,
                "MSE": val_mse,
                "pearson_correlation": val_corr,
                "n_points": len(y_val)
            },
            "test": {
                "MAE": test_mae,
                "MSE": test_mse,
                "pearson_correlation": test_corr,
                "n_points": len(y_test)
            },
            "test_real": {
                "MAE": test_real_mae,
                "MSE": test_real_mse,
                "pearson_correlation": test_real_corr,
                "n_points": len(df_test_real_group)
            },
            "test_real_part": {
                "MAE": test_real_part_mae,
                "MSE": test_real_part_mse,
                "pearson_correlation": test_real_part_corr,
                "n_points": len(df_test_real_part_utt)
            },
            "test_real_snr": {
                "MAE": test_real_snr_mae,
                "MSE": test_real_snr_mse,
                "pearson_correlation": test_real_snr_corr,
                "n_points": len(df_test_real_snr_group)
            },
            "test_real_part_snr": {
                "MAE": test_real_part_snr_mae,
                "MSE": test_real_part_snr_mse,
                "pearson_correlation": test_real_part_snr_corr,
                "n_points": len(df_test_real_part_snr_utt)
            }
        }
        if args.snr:
            for snr_val, info in snr_evals.items():
                snr_str = info["snr_str"]
                metrics[f"test_real_snr_{snr_str}"] = {
                    "MAE": info["group_mae"],
                    "MSE": info["group_mse"],
                    "pearson_correlation": info["group_corr"],
                    "n_points": len(info["df_group"])
                }
                metrics[f"test_real_part_snr_{snr_str}"] = {
                    "MAE": info["utt_mae"],
                    "MSE": info["utt_mse"],
                    "pearson_correlation": info["utt_corr"],
                    "n_points": len(info["df_utt"])
                }
        with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
            
        # 17. Dataset-specific reports for multi-dataset scenario
        if n_datasets > 1:
            logger.info("Generating dataset-specific reports for multi-dataset scenario...")
            # Map group_id and utt_id to dataset_idx
            group_to_dataset = {r["group_id"]: r["dataset_idx"] for r in filtered_rows if r.get("group_id") is not None}
            utt_to_dataset = {r["utt_id"]: r["dataset_idx"] for r in filtered_rows}
            
            for i in range(n_datasets):
                # Determine dataset directory name/label
                dataset_label = get_dataset_label(args.dataset, i)
                ds_output_dir = output_dir / dataset_label
                ds_output_dir.mkdir(parents=True, exist_ok=True)
                
                # Filter this dataset's test segments
                ds_test_rows = [r for r in test_rows if r.get("dataset_idx") == i]
                
                # Check how many variants were generated for this dataset
                ds_variants = dataset_variants.get(i, 1)
                
                # Generate test segments for this dataset specifically
                ds_test_segments = []
                for v in range(ds_variants):
                    ds_test_segments.extend(run_segmentation(ds_test_rows, asr_results, seed=args.seed, variant_index=v))
                
                if not ds_test_segments:
                    logger.warning(f"No test segments found for dataset {dataset_label}. Skipping test evaluation.")
                    ds_test_mae = 0.0
                    ds_test_mse = 0.0
                    ds_test_corr = None
                    ds_test_preds = pd.DataFrame(columns=["sample_id", "split", "accuracy", "estimated_accuracy", "residual", "ref_words"])
                    y_ds_test = []
                else:
                    # Generate test ensemble samples for this dataset
                    ds_test_samples, _ = generate_ensemble_samples(
                        ds_test_segments, args.target_ensemble // 4, seed=args.seed, n_jobs=args.jobs,
                        min_words=args.ensemble_min_words, max_words=args.ensemble_max_words, min_segments=args.ensemble_min_segments
                    )
                    
                    X_ds_test = np.array([s["features"] for s in ds_test_samples])
                    y_ds_test = np.array([s["accuracy"] for s in ds_test_samples])
                    ds_test_pred_calib = predictor.predict(X_ds_test)
                    
                    ds_test_mae = float(np.mean(np.abs(y_ds_test - ds_test_pred_calib)))
                    ds_test_mse = float(np.mean((y_ds_test - ds_test_pred_calib) ** 2))
                    ds_test_corr = safe_pearsonr(y_ds_test, ds_test_pred_calib)
                    
                    ds_test_records = []
                    for idx, (act, est, s) in enumerate(zip(y_ds_test, ds_test_pred_calib, ds_test_samples)):
                        ds_test_records.append({
                            "sample_id": s["sample_id"],
                            "split": "test",
                            "accuracy": act,
                            "estimated_accuracy": est,
                            "residual": act - est,
                            "ref_words": s["ref_words"]
                        })
                    ds_test_preds = pd.DataFrame(ds_test_records)
                    ds_test_preds.to_csv(ds_output_dir / "predictions.test.csv", index=False)
                
                # Define helper functions locally for filtering
                def get_original_group(x):
                    base = x.rsplit('_w', 1)[0]
                    if '_snr' in base:
                        base = base.rsplit('_snr', 1)[0]
                    return base
                    
                def get_original_utt(x):
                    base = x.rsplit('_w', 1)[0]
                    if '_snr' in base:
                        base = base.rsplit('_snr', 1)[0]
                    return base

                # Filter test_real predictions for this dataset (clean only)
                ds_test_real_group = df_test_real_group[
                    df_test_real_group["sample_id"].apply(lambda x: group_to_dataset.get(get_original_group(x))) == i
                ]
                ds_test_real_window = df_test_real_window[
                    df_test_real_window["sample_id"].apply(lambda x: group_to_dataset.get(get_original_group(x))) == i
                ]
                
                ds_test_real_group.to_csv(ds_output_dir / "predictions.test_real.csv", index=False)
                ds_test_real_window.to_csv(ds_output_dir / "predictions.test_real_window.csv", index=False)
                
                if not ds_test_real_group.empty:
                    ds_test_real_mae = float(np.mean(np.abs(ds_test_real_group["accuracy"] - ds_test_real_group["estimated_accuracy"])))
                    ds_test_real_mse = float(np.mean((ds_test_real_group["accuracy"] - ds_test_real_group["estimated_accuracy"]) ** 2))
                    ds_test_real_corr = safe_pearsonr(ds_test_real_group["accuracy"], ds_test_real_group["estimated_accuracy"])
                else:
                    ds_test_real_mae = 0.0
                    ds_test_real_mse = 0.0
                    ds_test_real_corr = None
                    
                # Filter test_real_part predictions for this dataset (clean only)
                ds_test_real_part_utt = df_test_real_part_utt[
                    df_test_real_part_utt["sample_id"].apply(lambda x: utt_to_dataset.get(get_original_utt(x))) == i
                ]
                ds_test_real_part_window = df_test_real_part_window[
                    df_test_real_part_window["sample_id"].apply(lambda x: utt_to_dataset.get(get_original_utt(x))) == i
                ]
                
                ds_test_real_part_utt.to_csv(ds_output_dir / "predictions.test_real_part.csv", index=False)
                ds_test_real_part_window.to_csv(ds_output_dir / "predictions.test_real_part_window.csv", index=False)
                
                if not ds_test_real_part_utt.empty:
                    ds_test_real_part_mae = float(np.mean(np.abs(ds_test_real_part_utt["accuracy"] - ds_test_real_part_utt["estimated_accuracy"])))
                    ds_test_real_part_mse = float(np.mean((ds_test_real_part_utt["accuracy"] - ds_test_real_part_utt["estimated_accuracy"]) ** 2))
                    ds_test_real_part_corr = safe_pearsonr(ds_test_real_part_utt["accuracy"], ds_test_real_part_utt["estimated_accuracy"])
                else:
                    ds_test_real_part_mae = 0.0
                    ds_test_real_part_mse = 0.0
                    ds_test_real_part_corr = None

                # Filter test_real_snr predictions for this dataset (including SNR)
                ds_test_real_snr_group = df_test_real_snr_group[
                    df_test_real_snr_group["sample_id"].apply(lambda x: group_to_dataset.get(get_original_group(x))) == i
                ]
                ds_test_real_snr_window = df_test_real_snr_window[
                    df_test_real_snr_window["sample_id"].apply(lambda x: group_to_dataset.get(get_original_group(x))) == i
                ]
                
                ds_test_real_snr_group.to_csv(ds_output_dir / "predictions.test_real_snr.csv", index=False)
                ds_test_real_snr_window.to_csv(ds_output_dir / "predictions.test_real_snr_window.csv", index=False)
                
                if not ds_test_real_snr_group.empty:
                    ds_test_real_snr_mae = float(np.mean(np.abs(ds_test_real_snr_group["accuracy"] - ds_test_real_snr_group["estimated_accuracy"])))
                    ds_test_real_snr_mse = float(np.mean((ds_test_real_snr_group["accuracy"] - ds_test_real_snr_group["estimated_accuracy"]) ** 2))
                    ds_test_real_snr_corr = safe_pearsonr(ds_test_real_snr_group["accuracy"], ds_test_real_snr_group["estimated_accuracy"])
                else:
                    ds_test_real_snr_mae = 0.0
                    ds_test_real_snr_mse = 0.0
                    ds_test_real_snr_corr = None
                    
                # Filter test_real_part_snr predictions for this dataset (including SNR)
                ds_test_real_part_snr_utt = df_test_real_part_snr_utt[
                    df_test_real_part_snr_utt["sample_id"].apply(lambda x: utt_to_dataset.get(get_original_utt(x))) == i
                ]
                ds_test_real_part_snr_window = df_test_real_part_snr_window[
                    df_test_real_part_snr_window["sample_id"].apply(lambda x: utt_to_dataset.get(get_original_utt(x))) == i
                ]
                
                ds_test_real_part_snr_utt.to_csv(ds_output_dir / "predictions.test_real_part_snr.csv", index=False)
                ds_test_real_part_snr_window.to_csv(ds_output_dir / "predictions.test_real_part_snr_window.csv", index=False)
                
                if not ds_test_real_part_snr_utt.empty:
                    ds_test_real_part_snr_mae = float(np.mean(np.abs(ds_test_real_part_snr_utt["accuracy"] - ds_test_real_part_snr_utt["estimated_accuracy"])))
                    ds_test_real_part_snr_mse = float(np.mean((ds_test_real_part_snr_utt["accuracy"] - ds_test_real_part_snr_utt["estimated_accuracy"]) ** 2))
                    ds_test_real_part_snr_corr = safe_pearsonr(ds_test_real_part_snr_utt["accuracy"], ds_test_real_part_snr_utt["estimated_accuracy"])
                else:
                    ds_test_real_part_snr_mae = 0.0
                    ds_test_real_part_snr_mse = 0.0
                    ds_test_real_part_snr_corr = None

                # Dataset-specific SNR-specific predictions and metrics
                ds_snr_evals = {}
                if args.snr:
                    for snr_val in args.snr:
                        snr_str = f"{int(snr_val)}" if snr_val == int(snr_val) else f"{snr_val}"
                        
                        # Predictions for this dataset & SNR value
                        ds_test_real_snr_val_group = df_test_real_snr_val_group_dict[snr_val][
                            df_test_real_snr_val_group_dict[snr_val]["sample_id"].apply(lambda x: group_to_dataset.get(get_original_group(x))) == i
                        ]
                        ds_test_real_snr_val_window = df_test_real_snr_val_window_dict[snr_val][
                            df_test_real_snr_val_window_dict[snr_val]["sample_id"].apply(lambda x: group_to_dataset.get(get_original_group(x))) == i
                        ]
                        
                        ds_test_real_snr_val_group.to_csv(ds_output_dir / f"predictions.test_real_snr_{snr_str}.csv", index=False)
                        ds_test_real_snr_val_window.to_csv(ds_output_dir / f"predictions.test_real_snr_{snr_str}_window.csv", index=False)
                        
                        if not ds_test_real_snr_val_group.empty:
                            ds_mae_s = float(np.mean(np.abs(ds_test_real_snr_val_group["accuracy"] - ds_test_real_snr_val_group["estimated_accuracy"])))
                            ds_mse_s = float(np.mean((ds_test_real_snr_val_group["accuracy"] - ds_test_real_snr_val_group["estimated_accuracy"]) ** 2))
                            ds_corr_s = safe_pearsonr(ds_test_real_snr_val_group["accuracy"], ds_test_real_snr_val_group["estimated_accuracy"])
                        else:
                            ds_mae_s = 0.0
                            ds_mse_s = 0.0
                            ds_corr_s = None
                        
                        # Part predictions for this dataset & SNR value
                        ds_test_real_part_snr_val_utt = df_test_real_part_snr_val_utt_dict[snr_val][
                            df_test_real_part_snr_val_utt_dict[snr_val]["sample_id"].apply(lambda x: utt_to_dataset.get(get_original_utt(x))) == i
                        ]
                        ds_test_real_part_snr_val_window = df_test_real_part_snr_val_window_dict[snr_val][
                            df_test_real_part_snr_val_window_dict[snr_val]["sample_id"].apply(lambda x: utt_to_dataset.get(get_original_utt(x))) == i
                        ]
                        
                        ds_test_real_part_snr_val_utt.to_csv(ds_output_dir / f"predictions.test_real_part_snr_{snr_str}.csv", index=False)
                        ds_test_real_part_snr_val_window.to_csv(ds_output_dir / f"predictions.test_real_part_snr_{snr_str}_window.csv", index=False)
                        
                        if not ds_test_real_part_snr_val_utt.empty:
                            ds_part_mae_s = float(np.mean(np.abs(ds_test_real_part_snr_val_utt["accuracy"] - ds_test_real_part_snr_val_utt["estimated_accuracy"])))
                            ds_part_mse_s = float(np.mean((ds_test_real_part_snr_val_utt["accuracy"] - ds_test_real_part_snr_val_utt["estimated_accuracy"]) ** 2))
                            ds_part_corr_s = safe_pearsonr(ds_test_real_part_snr_val_utt["accuracy"], ds_test_real_part_snr_val_utt["estimated_accuracy"])
                        else:
                            ds_part_mae_s = 0.0
                            ds_part_mse_s = 0.0
                            ds_part_corr_s = None
                            
                        ds_snr_evals[snr_val] = {
                            "snr_str": snr_str,
                            "group_mae": ds_mae_s,
                            "group_mse": ds_mse_s,
                            "group_corr": ds_corr_s,
                            "utt_mae": ds_part_mae_s,
                            "utt_mse": ds_part_mse_s,
                            "utt_corr": ds_part_corr_s,
                            "group_len": len(ds_test_real_snr_val_group),
                            "utt_len": len(ds_test_real_part_snr_val_utt),
                            "df_group": ds_test_real_snr_val_group,
                            "df_utt": ds_test_real_part_snr_val_utt
                        }
                
                # Save metrics.json for this dataset
                ds_metrics = {
                    "test": {
                        "MAE": ds_test_mae,
                        "MSE": ds_test_mse,
                        "pearson_correlation": ds_test_corr,
                        "n_points": len(y_ds_test)
                    },
                    "test_real": {
                        "MAE": ds_test_real_mae,
                        "MSE": ds_test_real_mse,
                        "pearson_correlation": ds_test_real_corr,
                        "n_points": len(ds_test_real_group)
                    },
                    "test_real_part": {
                        "MAE": ds_test_real_part_mae,
                        "MSE": ds_test_real_part_mse,
                        "pearson_correlation": ds_test_real_part_corr,
                        "n_points": len(ds_test_real_part_utt)
                    },
                    "test_real_snr": {
                        "MAE": ds_test_real_snr_mae,
                        "MSE": ds_test_real_snr_mse,
                        "pearson_correlation": ds_test_real_snr_corr,
                        "n_points": len(ds_test_real_snr_group)
                    },
                    "test_real_part_snr": {
                        "MAE": ds_test_real_part_snr_mae,
                        "MSE": ds_test_real_part_snr_mse,
                        "pearson_correlation": ds_test_real_part_snr_corr,
                        "n_points": len(ds_test_real_part_snr_utt)
                    }
                }
                if args.snr:
                    for snr_val, info in ds_snr_evals.items():
                        snr_str = info["snr_str"]
                        ds_metrics[f"test_real_snr_{snr_str}"] = {
                            "MAE": info["group_mae"],
                            "MSE": info["group_mse"],
                            "pearson_correlation": info["group_corr"],
                            "n_points": info["group_len"]
                        }
                        ds_metrics[f"test_real_part_snr_{snr_str}"] = {
                            "MAE": info["utt_mae"],
                            "MSE": info["utt_mse"],
                            "pearson_correlation": info["utt_corr"],
                            "n_points": info["utt_len"]
                        }
                with open(ds_output_dir / "metrics.json", "w", encoding="utf-8") as f:
                    json.dump(ds_metrics, f, indent=2, ensure_ascii=False)
                
                # Save plots for this dataset
                ds_plots_dir = ds_output_dir / "plots"
                ds_plots_dir.mkdir(parents=True, exist_ok=True)
                
                # Test scatter plot
                if not ds_test_preds.empty:
                    plt.figure(figsize=(6, 6))
                    plt.scatter(ds_test_preds["accuracy"], ds_test_preds["estimated_accuracy"], alpha=0.3, color="green")
                    plt.plot([0, 1], [0, 1], color="red", linestyle="--")
                    plt.xlabel("True Accuracy")
                    plt.ylabel("Estimated Accuracy")
                    plt.title(f"Test Scatter - {dataset_label} (MAE={ds_test_mae:.5f}, MSE={ds_test_mse:.5f})")
                    plt.xlim(-0.05, 1.05)
                    plt.ylim(-0.05, 1.05)
                    plt.grid(True)
                    plt.tight_layout()
                    plt.savefig(ds_plots_dir / "scatter_test.png")
                    plt.close()
                
                # Test_real scatter plot
                plt.figure(figsize=(6, 6))
                if not ds_test_real_group.empty:
                    plt.scatter(ds_test_real_group["accuracy"], ds_test_real_group["estimated_accuracy"], alpha=0.7, color="purple")
                plt.plot([0, 1], [0, 1], color="red", linestyle="--")
                plt.xlabel("True Accuracy")
                plt.ylabel("Estimated Accuracy")
                plt.title(f"Test_real Scatter - {dataset_label} (MAE={ds_test_real_mae:.5f}, MSE={ds_test_real_mse:.5f})")
                plt.xlim(-0.05, 1.05)
                plt.ylim(-0.05, 1.05)
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(ds_plots_dir / "scatter_test_real.png")
                plt.close()
                
                # Test_real_part scatter plot
                plt.figure(figsize=(6, 6))
                if not ds_test_real_part_utt.empty:
                    plt.scatter(ds_test_real_part_utt["accuracy"], ds_test_real_part_utt["estimated_accuracy"], alpha=0.5, color="orange")
                plt.plot([0, 1], [0, 1], color="red", linestyle="--")
                plt.xlabel("True Accuracy")
                plt.ylabel("Estimated Accuracy")
                plt.title(f"Test_real_part Scatter - {dataset_label} (MAE={ds_test_real_part_mae:.5f}, MSE={ds_test_real_part_mse:.5f})")
                plt.xlim(-0.05, 1.05)
                plt.ylim(-0.05, 1.05)
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(ds_plots_dir / "scatter_test_real_part.png")
                plt.close()

                # Test_real_snr scatter plot
                plt.figure(figsize=(6, 6))
                if not ds_test_real_snr_group.empty:
                    plt.scatter(ds_test_real_snr_group["accuracy"], ds_test_real_snr_group["estimated_accuracy"], alpha=0.7, color="teal")
                plt.plot([0, 1], [0, 1], color="red", linestyle="--")
                plt.xlabel("True Accuracy")
                plt.ylabel("Estimated Accuracy")
                plt.title(f"Test_real_snr Scatter - {dataset_label} (MAE={ds_test_real_snr_mae:.5f}, MSE={ds_test_real_snr_mse:.5f})")
                plt.xlim(-0.05, 1.05)
                plt.ylim(-0.05, 1.05)
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(ds_plots_dir / "scatter_test_real_snr.png")
                plt.close()
                
                # Test_real_part_snr scatter plot
                plt.figure(figsize=(6, 6))
                if not ds_test_real_part_snr_utt.empty:
                    plt.scatter(ds_test_real_part_snr_utt["accuracy"], ds_test_real_part_snr_utt["estimated_accuracy"], alpha=0.5, color="brown")
                plt.plot([0, 1], [0, 1], color="red", linestyle="--")
                plt.xlabel("True Accuracy")
                plt.ylabel("Estimated Accuracy")
                plt.title(f"Test_real_part_snr Scatter - {dataset_label} (MAE={ds_test_real_part_snr_mae:.5f}, MSE={ds_test_real_part_snr_mse:.5f})")
                plt.xlim(-0.05, 1.05)
                plt.ylim(-0.05, 1.05)
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(ds_plots_dir / "scatter_test_real_part_snr.png")
                plt.close()
                
                # Test_real_snr_{S} and Test_real_part_snr_{S} scatter plots
                if args.snr:
                    for snr_val, info in ds_snr_evals.items():
                        snr_str = info["snr_str"]
                        ds_group_s = info["df_group"]
                        ds_utt_s = info["df_utt"]
                        
                        plt.figure(figsize=(6, 6))
                        if not ds_group_s.empty:
                            plt.scatter(ds_group_s["accuracy"], ds_group_s["estimated_accuracy"], alpha=0.7, color="teal")
                        plt.plot([0, 1], [0, 1], color="red", linestyle="--")
                        plt.xlabel("True Accuracy")
                        plt.ylabel("Estimated Accuracy")
                        plt.title(f"Test_real_snr_{snr_str} Scatter - {dataset_label} (MAE={info['group_mae']:.5f}, MSE={info['group_mse']:.5f})")
                        plt.xlim(-0.05, 1.05)
                        plt.ylim(-0.05, 1.05)
                        plt.grid(True)
                        plt.tight_layout()
                        plt.savefig(ds_plots_dir / f"scatter_test_real_snr_{snr_str}.png")
                        plt.close()
                        
                        plt.figure(figsize=(6, 6))
                        if not ds_utt_s.empty:
                            plt.scatter(ds_utt_s["accuracy"], ds_utt_s["estimated_accuracy"], alpha=0.5, color="brown")
                        plt.plot([0, 1], [0, 1], color="red", linestyle="--")
                        plt.xlabel("True Accuracy")
                        plt.ylabel("Estimated Accuracy")
                        plt.title(f"Test_real_part_snr_{snr_str} Scatter - {dataset_label} (MAE={info['utt_mae']:.5f}, MSE={info['utt_mse']:.5f})")
                        plt.xlim(-0.05, 1.05)
                        plt.ylim(-0.05, 1.05)
                        plt.grid(True)
                        plt.tight_layout()
                        plt.savefig(ds_plots_dir / f"scatter_test_real_part_snr_{snr_str}.png")
                        plt.close()
                
                logger.info(f"Saved dataset-specific report for {dataset_label} to {ds_output_dir}")
                
        logger.info("Calibration workflow completed successfully.")
    finally:
        logger.removeHandler(file_handler)
        file_handler.close()

def main():
    parser = argparse.ArgumentParser(description="UWebASR Confidence Calibration Script")
    parser.add_argument("--dataset", action="append", required=True, help="Path to dataset manifest")
    parser.add_argument("--uwebasr-url", action="append", required=True, help="UWebASR model endpoint URL")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--target-segments", type=int, default=8000, help="Approximate target number of word-aligned segments")
    parser.add_argument("--jobs", type=int, default=6, help="Number of parallel ASR jobs")
    parser.add_argument("--seed", type=int, default=13, help="Random seed")
    parser.add_argument("--split-group", choices=["speaker", "utterance"], default="speaker", help="Group key for train/test split")
    parser.add_argument("--loss", choices=["mae", "mse"], default="mae", help="Loss function / optimization criterion for training the calibration model")
    parser.add_argument("--ensemble-min-words", type=int, default=512, help="Minimum number of reference words per ensemble sample")
    parser.add_argument("--ensemble-max-words", type=int, default=512, help="Maximum number of reference words per ensemble sample")
    parser.add_argument("--ensemble-min-segments", type=int, default=2, help="Minimum number of segments per ensemble sample")
    parser.add_argument("--skip-bad-rows", action="store_true", help="Skip rows with missing audio, empty references, etc.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of utterances to process for debugging")
    parser.add_argument("--snr", action="append", type=float, default=[], help="SNR levels for training data augmentation with additive noise")
    parser.add_argument("--target-ensemble", type=int, default=64000, help="Target number of ensemble samples for training/validation (test size will be 1/4 of this)")
    
    args = parser.parse_args()
    
    n_datasets = len(args.dataset)
    n_urls = len(args.uwebasr_url)
    
    if n_datasets != n_urls:
        parser.error(
            f"The number of --dataset ({n_datasets}) and --uwebasr-url ({n_urls}) arguments must match."
        )
        
    try:
        run_calibration_workflow(args)
    except Exception as e:
        logger.error(f"Calibration failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
