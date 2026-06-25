import os
import json
import csv
import logging
import random
from pathlib import Path
import numpy as np
import jiwer

from uwebasr_calibrate.normalizer import normalize_text, align_ref_and_hyp, TOKEN_RE
from uwebasr_calibrate.asr import validate_asr_result
from uwebasr_calibrate.features import extract_features

logger = logging.getLogger(__name__)

def load_manifest(manifest_path, skip_bad_rows=False):
    """
    Parses manifest path as JSON, JSONL, or CSV.
    Resolves relative audio_path relative to manifest directory.
    Validates audio file existence, non-empty reference, unique utt_id.
    """
    manifest_path = Path(manifest_path)
    manifest_dir = manifest_path.parent
    
    rows = []
    
    # Try parsing as JSON first (handles both MALACH array and JSONL/JSON array)
    content = None
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
    except Exception as e:
        raise ValueError(f"Failed to read manifest file: {e}")
        
    is_json = False
    data_list = []
    
    # Check if JSON array
    if content.startswith("[") and content.endswith("]"):
        try:
            data_list = json.loads(content)
            is_json = True
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON array in manifest: {e}")
    else:
        # Try JSONL
        try:
            for line in content.splitlines():
                if line.strip():
                    data_list.append(json.loads(line))
            is_json = True
        except json.JSONDecodeError:
            # Fallback to CSV
            is_json = False
            
    if not is_json:
        # Parse CSV
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    data_list.append(row)
        except Exception as e:
            raise ValueError(f"Failed to parse manifest as CSV: {e}")
            
    # Process rows
    seen_ids = set()
    for idx, item in enumerate(data_list):
        try:
            # Check for MALACH style
            if "filename" in item and "text" in item:
                filename = item["filename"]
                utt_id = Path(filename).stem
                audio_path = manifest_dir / "audio" / Path(filename).name
                reference = item["text"]
                speaker_id = item.get("speaker_id")
                video_id = item.get("video_id")
            else:
                audio_path_raw = item.get("audio_path")
                if not audio_path_raw:
                    raise ValueError(f"Missing 'audio_path' in row {idx}")
                audio_path = Path(audio_path_raw)
                if not audio_path.is_absolute():
                    audio_path = manifest_dir / audio_path
                    
                reference = item.get("reference")
                utt_id = item.get("utt_id")
                if not utt_id:
                    utt_id = audio_path.stem
                    
                speaker_id = item.get("speaker_id")
                video_id = item.get("video_id")
                
            # Validation
            if not reference or not reference.strip():
                raise ValueError(f"Empty reference in row {idx}")
            if not audio_path.exists():
                raise ValueError(f"Audio file does not exist: {audio_path}")
            if utt_id in seen_ids:
                raise ValueError(f"Duplicate utterance ID: {utt_id}")
                
            seen_ids.add(utt_id)
            
            rows.append({
                "utt_id": utt_id,
                "audio_path": str(audio_path),
                "reference": reference,
                "speaker_id": str(speaker_id) if speaker_id else None,
                "video_id": str(video_id) if video_id else None
            })
            
        except Exception as e:
            if skip_bad_rows:
                logger.warning(f"Skipping bad manifest row {idx}: {e}")
            else:
                raise ValueError(f"Invalid row in manifest at index {idx}: {e}")
                
    if not rows:
        raise ValueError("No valid rows parsed from manifest")
        
    return rows

import re
SPEAKER_RE = re.compile(r"(?<!\d)(\d{4,5})(?!\d)")

def get_speaker_id(utt_id, explicit_speaker_id=None):
    if explicit_speaker_id:
        return explicit_speaker_id
    match = SPEAKER_RE.search(utt_id)
    if match:
        return f"{int(match.group(1)):05d}"
    return None

VIDEO_PATTERNS = [
    re.compile(r"^(\d{5}_[A-Z]_\d{3})_"),
    re.compile(r"^(\d{5}_\d{2})_")
]

def get_video_id(utt_id, explicit_video_id=None):
    if explicit_video_id:
        return explicit_video_id
    for pattern in VIDEO_PATTERNS:
        match = pattern.match(utt_id)
        if match:
            return match.group(1)
    return utt_id

def split_dataset(rows, train_fraction=0.8, seed=13, split_group="speaker"):
    """
    Splits rows deterministically into train and test sets.
    If split_group is 'speaker', performs group-disjoint split.
    If split_group is 'utterance', performs utterance-level split.
    """
    if split_group == "speaker":
        # First assign speaker ID to each row
        for row in rows:
            row["speaker_id"] = get_speaker_id(row["utt_id"], row["speaker_id"])
            row["video_id"] = get_video_id(row["utt_id"], row["video_id"])
            
        # Get speakers
        speakers = sorted(list(set(r["speaker_id"] for r in rows if r["speaker_id"] is not None)))
        
        if len(speakers) < 2:
            raise ValueError(
                f"Fewer than two speaker groups found ({len(speakers)}). "
                "Disjoint speaker split is impossible. Please specify --split-group utterance."
            )
            
        rng = np.random.RandomState(seed)
        shuffled_speakers = list(speakers)
        rng.shuffle(shuffled_speakers)
        
        n_train = round(train_fraction * len(shuffled_speakers))
        # Ensure at least one train and one test speaker
        n_train = max(1, min(n_train, len(shuffled_speakers) - 1))
        
        train_speakers = set(shuffled_speakers[:n_train])
        test_speakers = set(shuffled_speakers[n_train:])
        
        train_rows = [r for r in rows if r["speaker_id"] in train_speakers]
        test_rows = [r for r in rows if r["speaker_id"] in test_speakers]
        
        # Verify disjointness
        assert not (set(r["speaker_id"] for r in train_rows) & set(r["speaker_id"] for r in test_rows))
        
        return train_rows, test_rows, list(train_speakers), list(test_speakers)
    else:
        # Utterance split
        rng = np.random.RandomState(seed)
        shuffled_idx = rng.permutation(len(rows))
        n_train = round(train_fraction * len(rows))
        n_train = max(1, min(n_train, len(rows) - 1))
        
        train_idx = set(shuffled_idx[:n_train])
        train_rows = [rows[i] for i in range(len(rows)) if i in train_idx]
        test_rows = [rows[i] for i in range(len(rows)) if i not in train_idx]
        
        for row in rows:
            row["speaker_id"] = get_speaker_id(row["utt_id"], row["speaker_id"]) or "unknown"
            row["video_id"] = get_video_id(row["utt_id"], row["video_id"])
            
        return train_rows, test_rows, [], []

def get_rng_seed(global_seed, variant_index, utt_id):
    import hashlib
    h = hashlib.sha256(utt_id.encode('utf-8')).hexdigest()
    utt_hash = int(h[:8], 16)
    return (global_seed + variant_index * 1000000 + utt_hash) % (2**32)

def generate_ref_spans(num_ref_words, min_words=10, max_words=256, rng=None):
    """
    Generates a list of (start, end) indices for reference words.
    Uses 'merge_to_previous_if_fits_else_drop' remainder policy.
    """
    if num_ref_words < min_words:
        return []
        
    spans = []
    curr = 0
    while curr < num_ref_words:
        if num_ref_words - curr < min_words:
            # Remainder
            if spans:
                prev_start, prev_end = spans[-1]
                if (num_ref_words - prev_start) <= max_words:
                    spans[-1] = (prev_start, num_ref_words)
            break
            
        limit = min(max_words, num_ref_words - curr)
        if limit < min_words:
            break
            
        if limit <= min_words:
            length = min_words
        else:
            if rng is None:
                length = random.randint(min_words, limit)
            else:
                length = rng.randint(min_words, limit + 1)
            
        spans.append((curr, curr + length))
        curr += length
        
    return spans

def slice_utterance(utt_id, reference, word_array, word_times, ctc_tokens, ctc_probs, ctc_frame_len, ref_spans, speaker_id=None, video_id=None):
    """
    Slices the utterance into segments based on reference word spans.
    """
    ref_words = normalize_text(reference)
    
    # Reconstruct hypothesis normalized words mapping to original word_array indices
    hyp_words = []
    hyp_word_to_original_idx = []
    for idx, w in enumerate(word_array):
        tokens = normalize_text(w)
        for tok in tokens:
            hyp_words.append(tok)
            hyp_word_to_original_idx.append(idx)
            
    if not ref_words or not hyp_words:
        return []
        
    ref_to_hyp = align_ref_and_hyp(ref_words, hyp_words)
    
    segments = []
    for ref_start, ref_end in ref_spans:
        # Find hypothesis index span corresponding to reference span
        hyp_indices = [ref_to_hyp[r] for r in range(ref_start, ref_end) if r in ref_to_hyp]
        if not hyp_indices:
            continue
            
        hyp_start = min(hyp_indices)
        hyp_end = max(hyp_indices) + 1
        
        # Clamp to valid hypothesis indices
        hyp_start = max(0, min(hyp_start, len(hyp_words) - 1))
        hyp_end_idx = max(0, min(hyp_end - 1, len(hyp_words) - 1))
        
        if hyp_start > hyp_end_idx:
            continue
            
        # Get start and end times
        orig_start = hyp_word_to_original_idx[hyp_start]
        orig_end = hyp_word_to_original_idx[hyp_end_idx]
        
        start_time = word_times[orig_start][0]
        end_time = word_times[orig_end][1]
        
        # Slicing CTC
        sel_tokens = []
        sel_probs = []
        for i, (tok, prob) in enumerate(zip(ctc_tokens, ctc_probs)):
            center_i = (i + 0.5) * ctc_frame_len
            if start_time <= center_i < end_time:
                sel_tokens.append(tok)
                sel_probs.append(prob)
                
        # Guards
        if len(sel_tokens) == 0:
            continue
            
        # Check implausibly long CTC span
        if len(sel_tokens) / (ref_end - ref_start) > 80:
            continue
            
        # Check if all three streams have data (blank, nonblank)
        has_blank = any(t == "<blk>" for t in sel_tokens)
        has_nonblank = any(t != "<blk>" for t in sel_tokens)
        if not has_blank or not has_nonblank:
            continue
            
        # Compute edit errors
        seg_ref = ref_words[ref_start:ref_end]
        seg_hyp = hyp_words[hyp_start:hyp_end]
        
        out = jiwer.process_words(" ".join(seg_ref), " ".join(seg_hyp))
        edit_errors = out.substitutions + out.insertions + out.deletions
        accuracy = max(0.0, 1.0 - edit_errors / len(seg_ref))
        
        segments.append({
            "utt_id": utt_id,
            "speaker_id": speaker_id,
            "video_id": video_id,
            "ref_start": ref_start,
            "ref_end": ref_end,
            "reference_words": len(seg_ref),
            "edit_errors": edit_errors,
            "accuracy": accuracy,
            "ctc_tokens": sel_tokens,
            "ctc_probs": sel_probs
        })
        
    return segments

def run_segmentation(rows, asr_results, seed, variant_index=0, min_words=10, max_words=256):
    """
    Runs segmentation on a set of rows for a single variant.
    """
    all_segments = []
    for row in rows:
        utt_id = row["utt_id"]
        asr_data = asr_results[utt_id]
        
        # Find final ASR result
        asr_result = validate_asr_result(asr_data["result"])
        
        ref_words = normalize_text(row["reference"])
        num_ref_words = len(ref_words)
        
        # Seeding
        utt_seed = get_rng_seed(seed, variant_index, utt_id)
        rng = np.random.RandomState(utt_seed)
        
        ref_spans = generate_ref_spans(num_ref_words, min_words, max_words, rng)
        
        segments = slice_utterance(
            utt_id=utt_id,
            reference=row["reference"],
            word_array=asr_result["word_array"],
            word_times=asr_result["word_times"],
            ctc_tokens=asr_result["ctc_tokens"],
            ctc_probs=asr_result["ctc_probs"],
            ctc_frame_len=asr_result["ctc_frame_len"],
            ref_spans=ref_spans,
            speaker_id=row["speaker_id"],
            video_id=row["video_id"]
        )
        
        all_segments.extend(segments)
        
    return all_segments

def create_deciles(segments):
    """
    Divides segments into 10 accuracy-sorted deciles.
    """
    sorted_segs = sorted(segments, key=lambda s: s["accuracy"])
    M = len(sorted_segs)
    if M == 0:
        return []
        
    deciles = []
    for k in range(10):
        start = k * M // 10
        end = (k + 1) * M // 10
        decile_segs = sorted_segs[start:end]
        if decile_segs:
            deciles.append(decile_segs)
            
    return deciles

def sample_ensemble_single(deciles, target_words, min_segments, rng):
    """
    Samples segments with replacement from the accuracy deciles.
    Uses 75% primary decile and 25% other deciles probability.
    """
    # Choose primary decile
    primary_idx = rng.choice(len(deciles))
    primary_pool = deciles[primary_idx]
    
    # Combined other pool
    other_pools = [deciles[i] for i in range(len(deciles)) if i != primary_idx]
    if other_pools:
        # Flatten other pools
        other_pool = [seg for pool in other_pools for seg in pool]
    else:
        other_pool = primary_pool
        
    chosen_segs = []
    total_words = 0
    
    while total_words < target_words or len(chosen_segs) < min_segments:
        # Decide which pool to draw from
        if len(deciles) == 1 or rng.uniform() < 0.75:
            seg = primary_pool[rng.choice(len(primary_pool))]
        else:
            seg = other_pool[rng.choice(len(other_pool))]
            
        chosen_segs.append(seg)
        total_words += seg["reference_words"]
        
    return chosen_segs

def generate_ensemble_samples(segments, sample_count, seed, target_words=512, min_segments=2):
    """
    Generates sample_count ensemble samples.
    """
    deciles = create_deciles(segments)
    if not deciles:
        raise ValueError("Cannot form deciles: no valid segments found")
        
    logger.info(f"Formed {len(deciles)} non-empty quantiles for ensemble sampling")
    
    rng = np.random.RandomState(seed)
    
    samples = []
    for i in range(sample_count):
        # We can seed each sample generation deterministically based on seed and i
        sample_seed = int((seed + i * 2026) % (2**32))
        sample_rng = np.random.RandomState(sample_seed)
        
        chosen_segs = sample_ensemble_single(deciles, target_words, min_segments, sample_rng)
        
        # Concatenate CTC tokens and probs
        tokens_all = []
        probs_all = []
        edit_errors = 0
        ref_words = 0
        
        for seg in chosen_segs:
            tokens_all.extend(seg["ctc_tokens"])
            probs_all.extend(seg["ctc_probs"])
            edit_errors += seg["edit_errors"]
            ref_words += seg["reference_words"]
            
        accuracy = max(0.0, 1.0 - edit_errors / ref_words)
        
        # Extract features
        features = extract_features(tokens_all, probs_all)
        
        samples.append({
            "sample_id": f"sample_{i}",
            "accuracy": accuracy,
            "ref_words": ref_words,
            "features": features
        })
        
    return samples, len(deciles)

def get_test_real_windows(test_rows, asr_results):
    """
    Converts test_rows into test_real speaker/video windows of approx 512 words.
    Returns window data ready for feature extraction.
    """
    # Group by video_id
    from collections import defaultdict
    video_groups = defaultdict(list)
    for row in test_rows:
        video_groups[row["video_id"]].append(row)
        
    windows_by_video = {}
    
    for vid, group in video_groups.items():
        # Sort by utt_id for determinism
        group = sorted(group, key=lambda r: r["utt_id"])
        
        # Generate at-most-512 spans for each utterance, and slice them
        video_chunks = []
        for row in group:
            utt_id = row["utt_id"]
            asr_data = asr_results[utt_id]
            asr_result = validate_asr_result(asr_data["result"])
            
            ref_words = normalize_text(row["reference"])
            num_ref_words = len(ref_words)
            
            # Slices for test_real windowing (approx 512 words chunks)
            ref_spans = []
            curr = 0
            while curr < num_ref_words:
                end = min(curr + 512, num_ref_words)
                ref_spans.append((curr, end))
                curr = end
                
            chunks = slice_utterance(
                utt_id=utt_id,
                reference=row["reference"],
                word_array=asr_result["word_array"],
                word_times=asr_result["word_times"],
                ctc_tokens=asr_result["ctc_tokens"],
                ctc_probs=asr_result["ctc_probs"],
                ctc_frame_len=asr_result["ctc_frame_len"],
                ref_spans=ref_spans,
                speaker_id=row["speaker_id"],
                video_id=row["video_id"]
            )
            video_chunks.extend(chunks)
            
        # Accumulate chunks into approx 512-word windows
        windows = []
        current_window = []
        current_words = 0
        
        for chunk in video_chunks:
            current_window.append(chunk)
            current_words += chunk["reference_words"]
            
            if current_words >= 512:
                windows.append(current_window)
                current_window = []
                current_words = 0
                
        # Remainder
        if current_words >= 10:
            windows.append(current_window)
            
        windows_by_video[vid] = windows
        
    return windows_by_video
