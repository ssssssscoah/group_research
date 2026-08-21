from pathlib import Path
import random
from typing import Dict

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from .dataset import SplitData


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def l2_normalize_np(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norm, 1e-12, None)


def cosine_similarity_np(image: np.ndarray, text: np.ndarray) -> np.ndarray:
    image = l2_normalize_np(image)
    text = l2_normalize_np(text)
    return np.sum(image * text, axis=1)


def binary_metrics(labels: np.ndarray, preds: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
    }


def find_best_cosine_threshold(similarities: np.ndarray, labels: np.ndarray):
    """
    NewsCLIPpings label 1 means falsified/out-of-context.
    Lower CLIP similarity is therefore predicted as class 1.
    Threshold is selected only on validation data.
    """
    lo = float(similarities.min())
    hi = float(similarities.max())
    thresholds = np.linspace(lo, hi, 1001)

    best_threshold = thresholds[0]
    best_f1 = -1.0

    for threshold in thresholds:
        preds = (similarities < threshold).astype(np.int64)
        score = f1_score(labels, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = threshold

    return float(best_threshold), float(best_f1)


def save_predictions(
    split: SplitData,
    preds: np.ndarray,
    model_scores: np.ndarray,
    output_path: str | Path,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cosine = cosine_similarity_np(split.image, split.text)

    df = pd.DataFrame(
        {
            "text_id": split.text_ids,
            "image_id": split.image_ids,
            "label_falsified": split.labels,
            "prediction_falsified": preds,
            "model_score": model_scores,
            "cosine_similarity": cosine,
            "correct": (preds == split.labels).astype(int),
        }
    )
    df.to_csv(output_path, index=False)
