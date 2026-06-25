import re
import jiwer

# Matches unicode letters and digits, keeps internal straight/curly apostrophes, ignores underscores/punctuation
TOKEN_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)?", re.UNICODE)

def normalize_text(text):
    if not text:
        return []
    text_lower = text.lower()
    return TOKEN_RE.findall(text_lower)

def align_ref_and_hyp(ref_words, hyp_words):
    """
    Performs deterministic word-level alignment between reference and hypothesis tokens.
    Returns a dictionary mapping reference word index to hypothesis word index.
    
    If a reference word is deleted, it maps to the hypothesis index where it was deleted.
    """
    ref_str = " ".join(ref_words)
    hyp_str = " ".join(hyp_words)
    
    if not ref_str:
        return {}
    if not hyp_str:
        # All reference words are deleted, they map to hypothesis index 0
        return {i: 0 for i in range(len(ref_words))}
        
    out = jiwer.process_words(ref_str, hyp_str)
    chunks = out.alignments[0]
    
    ref_to_hyp = {}
    for chunk in chunks:
        if chunk.type == "equal" or chunk.type == "substitute":
            for offset in range(chunk.ref_end_idx - chunk.ref_start_idx):
                ref_to_hyp[chunk.ref_start_idx + offset] = chunk.hyp_start_idx + offset
        elif chunk.type == "delete":
            for offset in range(chunk.ref_end_idx - chunk.ref_start_idx):
                ref_to_hyp[chunk.ref_start_idx + offset] = chunk.hyp_start_idx
                
    # Fill in any missing reference indices just in case (e.g. if chunks didn't cover everything)
    for i in range(len(ref_words)):
        if i not in ref_to_hyp:
            ref_to_hyp[i] = 0
            
    return ref_to_hyp
