"""
eval.py

Evaluation utilities: CER, WER, and helper to evaluate OCR models.
"""

from Levenshtein import distance as levenshtein_distance
from jiwer import wer


import torch
from torch.utils.data import DataLoader

from .models import CRNN


from jiwer import wer


from typing import Dict

def normalize_text(s: str) -> str:
    """
    Simple normalization: lowercase and collapse whitespace.
    You can adapt this depending on how much you want to preserve
    early-modern spelling and punctuation.
    """
    import re
    s = s.lower()
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def cer(gt: str, pred: str) -> float:
    """
    Character Error Rate: Levenshtein distance / #characters in GT.
    """
    gt_n = normalize_text(gt)
    pred_n = normalize_text(pred)
    if len(gt_n) == 0:
        return 0.0
    return levenshtein_distance(gt_n, pred_n) / len(gt_n)



def wer_jiwer(gt: str, pred: str) -> float:
    """
    Word Error Rate using jiwer.
    """
    return wer(normalize_text(gt), normalize_text(pred))



@torch.no_grad()
def evaluate_crnn(model: CRNN,
                  loader: DataLoader,
                  idx2char: dict,
                  device) -> Dict[str, float]:
    """
    Evaluate a trained CRNN on a dataset (line-level).
    Returns average CER and WER.
    """
    model.eval()
    all_cer, all_wer = [], []

    for batch in loader:
        images = batch["images"].to(device)
        gt_texts = batch["texts"]

        logits = model(images)  # (T, N, C)
        preds = model.decode_greedy(logits.cpu(), idx2char)

        for gt, pr in zip(gt_texts, preds):
            all_cer.append(cer(gt, pr))
            all_wer.append(wer_jiwer(gt, pr))

    avg_cer = sum(all_cer) / max(1, len(all_cer))
    avg_wer = sum(all_wer) / max(1, len(all_wer))

    return {"cer": avg_cer, "wer": avg_wer}
