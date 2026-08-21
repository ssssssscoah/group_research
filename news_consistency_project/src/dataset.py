from dataclasses import dataclass
from pathlib import Path
import json
import pickle
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class SplitData:
    image: np.ndarray
    text: np.ndarray
    labels: np.ndarray
    text_ids: List[Any]
    image_ids: List[Any]

    def __len__(self) -> int:
        return len(self.labels)


def _lookup_embedding(mapping: Dict[Any, np.ndarray], key: Any) -> np.ndarray:
    """Robustly look up an embedding whether pickle keys are int or str."""
    candidates = [key]
    try:
        candidates.append(int(key))
    except (TypeError, ValueError):
        pass
    candidates.append(str(key))

    for candidate in candidates:
        if candidate in mapping:
            return np.asarray(mapping[candidate], dtype=np.float32).reshape(-1)

    raise KeyError(f"Embedding id {key!r} was not found in the pickle file.")


def _split_paths(data_root: str | Path, split: str):
    root = Path(data_root)
    ann_path = root / "data" / "merged_balanced" / f"{split}.json"
    image_path = (
        root
        / "embeddings"
        / "clip_image_embeddings"
        / f"clip_image_embeddings_{split}.pkl"
    )
    text_path = (
        root
        / "embeddings"
        / "clip_text_embeddings"
        / f"clip_text_embeddings_{split}.pkl"
    )
    return ann_path, image_path, text_path


def load_split(data_root: str | Path, split: str) -> SplitData:
    """
    Load one NewsCLIPpings merged_balanced split using the official file layout.

    Label convention:
        falsified = 0 -> pristine/matched
        falsified = 1 -> out-of-context/mismatched
    """
    ann_path, image_path, text_path = _split_paths(data_root, split)

    for path in (ann_path, image_path, text_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}\n"
                "Check --data-root and make sure NewsCLIPpings data/embeddings were downloaded."
            )

    with ann_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    annotations = data["annotations"]

    with image_path.open("rb") as f:
        image_embeddings = pickle.load(f)
    with text_path.open("rb") as f:
        text_embeddings = pickle.load(f)

    image_rows = []
    text_rows = []
    labels = []
    text_ids = []
    image_ids = []

    for ann in annotations:
        text_id = ann["id"]
        image_id = ann["image_id"]

        image_vec = _lookup_embedding(image_embeddings, image_id)
        text_vec = _lookup_embedding(text_embeddings, text_id)

        if image_vec.shape[0] != 512 or text_vec.shape[0] != 512:
            raise ValueError(
                f"Expected CLIP ViT-B/32 embeddings of length 512, got "
                f"image={image_vec.shape}, text={text_vec.shape}."
            )

        image_rows.append(image_vec)
        text_rows.append(text_vec)
        labels.append(int(ann["falsified"]))
        text_ids.append(text_id)
        image_ids.append(image_id)

    return SplitData(
        image=np.stack(image_rows).astype(np.float32),
        text=np.stack(text_rows).astype(np.float32),
        labels=np.asarray(labels, dtype=np.int64),
        text_ids=text_ids,
        image_ids=image_ids,
    )


class TorchEmbeddingDataset(Dataset):
    def __init__(self, split_data: SplitData):
        self.image = torch.from_numpy(split_data.image)
        self.text = torch.from_numpy(split_data.text)
        self.labels = torch.from_numpy(split_data.labels.astype(np.float32))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.image[idx], self.text[idx], self.labels[idx]
