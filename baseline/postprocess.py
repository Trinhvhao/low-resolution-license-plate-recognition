"""
2.5 Structural Constraint Post-processing.

Correct CTC Beam Search (blank/non-blank state separation)
+ Brazilian/Mercosur plate format validation bonus.

Usage:
    from postprocess import beam_search_decode, batch_beam_search_decode
    from postprocess import validate_plate_format
"""

import math
import re
from collections import defaultdict

import torch


# ── Brazilian / Mercosur plate patterns ─────────────────────────────────────

PLATE_PATTERNS = [
    re.compile(r'^[A-Z]{3}[0-9]{4}$'),           # ABC1234  (old Brazilian)
    re.compile(r'^[A-Z]{3}[0-9][A-Z][0-9]{2}$'), # ABC1D23  (Mercosur)
]


def validate_plate_format(text):
    """Check if text matches Brazilian/Mercosur 7-char plate format."""
    text = text.upper().strip()
    if len(text) != 7:
        return False, 'wrong_length'
    if PLATE_PATTERNS[0].match(text):
        return True, 'brazilian_old'
    if PLATE_PATTERNS[1].match(text):
        return True, 'mercosur'
    return False, 'unknown'


# ── CTC Beam Search (correct blank/non-blank state) ────────────────────────

def _log_add(a, b):
    """Numerically stable log(exp(a) + exp(b))."""
    if a == float('-inf'):
        return b
    if b == float('-inf'):
        return a
    mx = max(a, b)
    return mx + math.log1p(math.exp(-abs(a - b)))


def beam_search_decode(log_probs, idx2char, beam_width=10, blank=0,
                       format_bonus_weight=1.5):
    """
    CTC prefix beam search with proper blank/non-blank state separation.

    Each beam tracks TWO scores:
      p_b  = probability of paths ending with blank
      p_nb = probability of paths ending with non-blank
    This correctly handles repeated characters (e.g. "AA" via blank separator).

    Args:
        log_probs:  [T, C] log-probabilities per timestep
        idx2char:   dict {int: str}
        beam_width: number of beams to keep
        blank:      CTC blank index
        format_bonus_weight: log-prob bonus for valid plate format candidates

    Returns:
        best_text:  str
        confidence: float (0-1)
        candidates: list of (text, adjusted_score) top-5
    """
    NEG_INF = float('-inf')
    T, C = log_probs.shape

    # beam dict: prefix_str -> (p_blank, p_non_blank)
    beams = {'': (0.0, NEG_INF)}  # start: blank path with score 0

    for t in range(T):
        new_beams = defaultdict(lambda: (NEG_INF, NEG_INF))

        # Pre-extract top-k chars to limit inner loop (speed optimisation)
        # For C=37 and beam_width=10 this is fine without pruning chars
        lp = log_probs[t]  # [C]

        for prefix, (p_b, p_nb) in beams.items():
            p_total = _log_add(p_b, p_nb)

            for c in range(C):
                lp_c = lp[c].item()

                if c == blank:
                    # Blank extends any path without changing prefix
                    old_b, old_nb = new_beams[prefix]
                    new_beams[prefix] = (_log_add(old_b, p_total + lp_c), old_nb)

                else:
                    char = idx2char.get(c, '')
                    # Last emitted character (from prefix string)
                    last_char = prefix[-1] if prefix else ''

                    if char == last_char:
                        # Same char as end of prefix:
                        #   - If previous ended with blank → can repeat → extend
                        #   - If previous ended with non-blank → collapse (stay)
                        # COLLAPSE (stay on same prefix, p_nb path)
                        old_b, old_nb = new_beams[prefix]
                        new_beams[prefix] = (old_b, _log_add(old_nb, p_nb + lp_c))

                        # EXTEND via blank separator → new char (from p_b only)
                        new_prefix = prefix + char
                        old_b2, old_nb2 = new_beams[new_prefix]
                        new_beams[new_prefix] = (old_b2, _log_add(old_nb2, p_b + lp_c))
                    else:
                        # Different char → always extend
                        new_prefix = prefix + char
                        old_b2, old_nb2 = new_beams[new_prefix]
                        new_beams[new_prefix] = (
                            old_b2,
                            _log_add(old_nb2, p_total + lp_c)
                        )

        # Prune: keep top beam_width by total score
        scored = []
        for pref, (pb, pnb) in new_beams.items():
            scored.append((pref, pb, pnb, _log_add(pb, pnb)))
        scored.sort(key=lambda x: x[3], reverse=True)

        beams = {}
        for pref, pb, pnb, _ in scored[:beam_width]:
            beams[pref] = (pb, pnb)

    # Final scoring with format bonus
    candidates = []
    for prefix, (pb, pnb) in beams.items():
        raw_score = _log_add(pb, pnb)
        valid, _ = validate_plate_format(prefix)
        bonus = format_bonus_weight if valid else 0.0
        candidates.append((prefix, raw_score + bonus, raw_score))

    candidates.sort(key=lambda x: x[1], reverse=True)

    if not candidates:
        return '', 0.0, []

    best_text = candidates[0][0]
    best_raw = candidates[0][2]
    # Confidence: exp(mean log-prob)
    avg_lp = best_raw / max(T, 1)
    confidence = min(1.0, max(0.0, math.exp(avg_lp)))

    return best_text, confidence, [(t, s) for t, s, _ in candidates[:5]]


# ── Batch wrapper ───────────────────────────────────────────────────────────

def batch_beam_search_decode(log_probs_batch, idx2char, beam_width=10, blank=0,
                             format_bonus_weight=1.5):
    """
    Beam search over a batch of CTC outputs.

    Args:
        log_probs_batch: [B, T, C] tensor or list of [T, C]
        idx2char:        dict
        beam_width:      int

    Returns:
        texts:       list[str]
        confidences: list[float]
    """
    if isinstance(log_probs_batch, torch.Tensor):
        batch = [log_probs_batch[i] for i in range(log_probs_batch.size(0))]
    else:
        batch = log_probs_batch

    texts, confidences = [], []
    for lp in batch:
        text, conf, _ = beam_search_decode(
            lp, idx2char, beam_width=beam_width, blank=blank,
            format_bonus_weight=format_bonus_weight
        )
        texts.append(text)
        confidences.append(conf)

    return texts, confidences
