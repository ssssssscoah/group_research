import copy

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .dataset import SplitData, TorchEmbeddingDataset
from .utils import binary_metrics


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_arg)


def predict_probabilities(
    model: nn.Module,
    split: SplitData,
    device: torch.device,
    batch_size: int = 512,
) -> np.ndarray:
    loader = DataLoader(
        TorchEmbeddingDataset(split),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    model.eval()
    probs = []

    with torch.no_grad():
        for image, text, _ in loader:
            image = image.to(device)
            text = text.to(device)
            logits = model(image, text)
            probs.append(torch.sigmoid(logits).cpu().numpy())

    return np.concatenate(probs)


def train_model(
    model: nn.Module,
    train_split: SplitData,
    val_split: SplitData,
    device: torch.device,
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 5,
):
    train_loader = DataLoader(
        TorchEmbeddingDataset(train_split),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    best_state = copy.deepcopy(model.state_dict())
    best_val_f1 = -1.0
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_examples = 0

        for image, text, labels in train_loader:
            image = image.to(device)
            text = text.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(image, text)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            batch_n = labels.shape[0]
            total_loss += loss.item() * batch_n
            total_examples += batch_n

        val_probs = predict_probabilities(model, val_split, device, batch_size)
        val_preds = (val_probs >= 0.5).astype(np.int64)
        val_metrics = binary_metrics(val_split.labels, val_preds)
        train_loss = total_loss / max(total_examples, 1)

        print(
            f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | val_f1={val_metrics['f1']:.4f}"
        )

        if val_metrics["f1"] > best_val_f1 + 1e-6:
            best_val_f1 = val_metrics["f1"]
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping after epoch {epoch}.")
            break

    model.load_state_dict(best_state)
    return model, best_val_f1
