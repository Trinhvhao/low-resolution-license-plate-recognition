import os
import random

import numpy as np
import torch


def seed_everything(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"🔒 Đã cố định Seed: {seed}")


def decode_predictions(preds, idx2char):
    """
    Decode CTC predictions to text strings.
    
    Args:
        preds: Tensor of shape [B, T] containing predicted character indices
        idx2char: Dictionary mapping indices to characters
    
    Returns:
        List of decoded strings
    """
    result_list = []
    for p in preds:
        pred_str = ""
        last_char = 0
        for char_idx in p:
            c = char_idx.item()
            if c != 0 and c != last_char:
                pred_str += idx2char.get(c, '')
            last_char = c
        result_list.append(pred_str)
    return result_list


def calculate_cer(predictions, targets):
    """
    Calculate Character Error Rate (CER).
    
    CER = (Substitutions + Deletions + Insertions) / Total_Characters
    Uses Levenshtein distance.
    
    Args:
        predictions: List of predicted strings
        targets: List of target strings
    
    Returns:
        CER as float (0.0 to 1.0+)
    """
    try:
        import editdistance
    except ImportError:
        print("⚠️ Warning: editdistance not installed. Install: pip install editdistance")
        return 0.0
    
    total_distance = 0
    total_length = 0
    
    for pred, target in zip(predictions, targets):
        distance = editdistance.eval(pred, target)
        total_distance += distance
        total_length += len(target)
    
    return total_distance / total_length if total_length > 0 else 0.0


def calculate_accuracy(predictions, targets):
    """
    Calculate exact match accuracy (Recognition Rate for ICPR 2026).
    
    Args:
        predictions: List of predicted strings
        targets: List of target strings
    
    Returns:
        Accuracy as float (0.0 to 1.0)
    """
    correct = sum(1 for pred, target in zip(predictions, targets) if pred == target)
    return correct / len(targets) if len(targets) > 0 else 0.0


def calculate_confidence_gap(confidences, is_correct):
    """
    Calculate Confidence Gap - ICPR 2026 tie-breaker metric.
    
    Gap = mean(confidence_correct) - mean(confidence_wrong)
    
    Higher gap is better: model should be confident on correct predictions
    and less confident on wrong predictions.
    
    Args:
        confidences: List of confidence scores (0 to 1)
        is_correct: List of booleans indicating if prediction is correct
    
    Returns:
        Confidence gap as float
    """
    import numpy as np
    
    confidences = np.array(confidences)
    is_correct = np.array(is_correct)
    
    correct_confs = confidences[is_correct]
    wrong_confs = confidences[~is_correct]
    
    mean_correct = correct_confs.mean() if len(correct_confs) > 0 else 0.0
    mean_wrong = wrong_confs.mean() if len(wrong_confs) > 0 else 0.0
    
    gap = mean_correct - mean_wrong
    
    return gap


def get_prediction_confidence(logits):
    """
    Extract confidence scores from CTC logits.
    
    Uses mean of max probabilities across sequence as confidence.
    
    Args:
        logits: Tensor [B, T, num_classes] (log probabilities)
    
    Returns:
        Confidence scores [B] as numpy array
    """
    import torch
    import numpy as np
    
    # Convert log probs to probs
    probs = torch.exp(logits)  # [B, T, num_classes]
    
    # Get max prob at each timestep and average
    max_probs, _ = probs.max(dim=2)  # [B, T]
    mean_confidence = max_probs.mean(dim=1)  # [B]
    
    return mean_confidence.cpu().numpy()
