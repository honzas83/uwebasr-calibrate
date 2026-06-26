import os
import json
import time
import logging
import subprocess
from urllib.parse import urlparse, parse_qs, urlunparse
import requests

logger = logging.getLogger(__name__)

def prepare_url(url):
    """
    Ensures format=speechcloud_json is added to the URL and fails if format is set to something else.
    """
    parsed = urlparse(url)
    query = parsed.query
    params = parse_qs(query)
    
    if "format" in params:
        formats = params["format"]
        if len(formats) != 1 or formats[0] != "speechcloud_json":
            raise ValueError(f"URL format parameter must be 'speechcloud_json', got {formats}")
        return url
        
    if query:
        new_query = f"{query}&format=speechcloud_json"
    else:
        new_query = "format=speechcloud_json"
        
    return urlunparse(parsed._replace(query=new_query))

def convert_audio_to_ogg(audio_path):
    """
    Converts audio file to 16 kHz mono Ogg/Vorbis using ffmpeg.
    Returns the binary data.
    """
    cmd = [
        "ffmpeg", "-xerror", "-hide_banner", "-loglevel", "error",
        "-i", str(audio_path),
        "-ar", "16000", "-ac", "1", "-vn",
        "-c:a", "libvorbis", "-q:a", "10",
        "-f", "ogg", "-"
    ]
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return res.stdout
    except subprocess.CalledProcessError as e:
        logger.error(f"Ffmpeg error for {audio_path}: {e.stderr.decode('utf-8')}")
        raise e

def validate_asr_result(result_list):
    """
    Validates a raw SpeechCloud JSON response list.
    Returns the final ASR result dictionary if valid, otherwise raises ValueError.
    """
    if not isinstance(result_list, list):
        raise ValueError("ASR response is not a JSON list")
        
    final_asr = None
    for obj in result_list:
        if isinstance(obj, dict) and obj.get("type") == "asr_result" and not obj.get("partial_result", False):
            final_asr = obj
            break
            
    if not final_asr:
        raise ValueError("Missing final ASR result (type == 'asr_result' and partial_result == false)")
        
    # Check CTC data
    ctc_tokens = final_asr.get("ctc_tokens")
    ctc_probs = final_asr.get("ctc_probs")
    if not isinstance(ctc_tokens, list) or not isinstance(ctc_probs, list):
        raise ValueError("Missing or invalid ctc_tokens/ctc_probs arrays")
        
    if len(ctc_tokens) != len(ctc_probs):
        raise ValueError(f"ctc_tokens and ctc_probs lengths mismatch: {len(ctc_tokens)} vs {len(ctc_probs)}")
        
    # Check word array
    word_array = final_asr.get("word_array")
    word_times = final_asr.get("word_times")
    if not isinstance(word_array, list) or not isinstance(word_times, list):
        raise ValueError("Missing or invalid word_array/word_times arrays")
        
    if len(word_array) != len(word_times):
        raise ValueError(f"word_array and word_times lengths mismatch: {len(word_array)} vs {len(word_times)}")
        
    # Optional check for consistency of frames
    ctc_frame_len = final_asr.get("ctc_frame_len")
    audio_duration = final_asr.get("audio_duration")
    if ctc_frame_len is not None and audio_duration is not None:
        try:
            expected_frames = round(float(audio_duration) / float(ctc_frame_len))
            actual_frames = len(ctc_probs)
            if abs(expected_frames - actual_frames) > 2:
                logger.warning(
                    f"Inconsistent frame count: expected {expected_frames} (duration {audio_duration} / "
                    f"frame_len {ctc_frame_len}), got {actual_frames} frames."
                )
        except Exception as e:
            logger.warning(f"Error checking frame consistency: {e}")
            
    return final_asr

def convert_audio_to_ogg_with_noise(audio_path, snr, utt_id):
    """
    Converts audio file to 16 kHz mono PCM, adds white Gaussian noise at target SNR,
    and converts to 16 kHz mono Ogg/Vorbis using ffmpeg.
    Returns the binary data.
    """
    import numpy as np
    import hashlib
    
    # 1. Read audio as 16kHz mono 16-bit PCM
    cmd_read = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", str(audio_path),
        "-f", "s16le", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", "-"
    ]
    try:
        res = subprocess.run(cmd_read, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Ffmpeg read error for {audio_path}: {e.stderr.decode('utf-8')}")
        raise e
        
    samples = np.frombuffer(res.stdout, dtype=np.int16).astype(np.float32)
    
    # 2. Add additive white noise
    power_signal = np.mean(samples ** 2)
    if power_signal <= 0:
        power_signal = 1e-10
        
    power_noise = power_signal * (10.0 ** (-snr / 10.0))
    
    # Deterministic noise generation based on utt_id and snr
    seed_str = f"{utt_id or ''}_{snr}"
    seed_hash = hashlib.sha256(seed_str.encode("utf-8")).digest()
    seed = int.from_bytes(seed_hash[:4], byteorder="big")
    rng = np.random.default_rng(seed)
    
    noise = rng.normal(0, np.sqrt(power_noise), size=len(samples))
    
    # Add noise and clip to int16 range
    augmented = samples + noise
    augmented = np.clip(augmented, -32768.0, 32767.0).astype(np.int16)
    
    # 3. Convert PCM to Ogg/Vorbis
    cmd_write = [
        "ffmpeg", "-xerror", "-hide_banner", "-loglevel", "error",
        "-f", "s16le", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        "-i", "-",
        "-c:a", "libvorbis", "-q:a", "10",
        "-f", "ogg", "-"
    ]
    try:
        res_write = subprocess.run(cmd_write, input=augmented.tobytes(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return res_write.stdout
    except subprocess.CalledProcessError as e:
        logger.error(f"Ffmpeg write error converting noisy PCM to Ogg: {e.stderr.decode('utf-8')}")
        raise e

def run_recognition_single(session, url, audio_path, timeout_seconds=90, retries=7, snr=None, utt_id=None):
    """
    Sends the audio file to the UWebASR endpoint and returns the raw response list.
    Includes retries and backoff.
    """
    if snr is not None:
        ogg_bytes = convert_audio_to_ogg_with_noise(audio_path, snr, utt_id)
    else:
        ogg_bytes = convert_audio_to_ogg(audio_path)
    
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            response = session.post(
                url,
                headers={"Content-Type": "audio/ogg"},
                data=ogg_bytes,
                timeout=timeout_seconds
            )
            
            if response.status_code == 200:
                try:
                    result = response.json()
                    # Basic validation of json list
                    if not isinstance(result, list):
                        raise ValueError("Response is not a list")
                    return result
                except Exception as e:
                    last_error = ValueError(f"Failed to parse JSON response: {e}")
            elif 400 <= response.status_code < 500:
                # Permanent error, do not retry
                raise Exception(f"Permanent HTTP error: {response.status_code} - {response.text}")
            else:
                last_error = Exception(f"Server returned HTTP status {response.status_code}: {response.text}")
                
        except Exception as e:
            logger.warning(f"Attempt {attempt}/{retries} for {audio_path} failed: {e}")
            last_error = e
            
        if attempt < retries:
            backoff = min(30.0, 2.0 ** (attempt - 1))
            logger.info(f"Retrying in {backoff} seconds...")
            time.sleep(backoff)
            
    raise Exception(f"ASR failed after {retries} attempts. Last error: {last_error}")
