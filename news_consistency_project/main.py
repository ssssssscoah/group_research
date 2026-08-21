import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression

from src.dataset import load_split
from src.models import SimpleFusionMLP, ConsistencyAwareMLP
from src.train import train_model, predict_probabilities, resolve_device
from src.utils import (
    binary_metrics,
    cosine_similarity_np,
    find_best_cosine_threshold,
    l2_normalize_np,
    save_predictions,
    set_seed,
)


MAIN_EXPERIMENTS = [
    "cosine",
    "text_lr",
    "image_lr",
    "simple_mlp",
    "consistency_mlp",
]


ABLATIONS = {
    "base_only": (False, False, False),
    "plus_cosine": (True, False, False),
    "plus_abs_diff": (False, True, False),
    "plus_interaction": (False, False, True),
    "all_features": (True, True, True),
}


def print_metrics(name, metrics):
    print(f"\n{name}")
    print("-" * len(name))
    for k, v in metrics.items():
        print(f"{k:>9}: {v:.4f}")


def run_cosine(val, test, output_dir):
    val_sim = cosine_similarity_np(val.image, val.text)
    threshold, val_f1 = find_best_cosine_threshold(val_sim, val.labels)

    test_sim = cosine_similarity_np(test.image, test.text)
    test_preds = (test_sim < threshold).astype(np.int64)
    metrics = binary_metrics(test.labels, test_preds)
    metrics["val_selected_threshold"] = threshold
    metrics["val_f1_at_threshold"] = val_f1

    save_predictions(
        test,
        test_preds,
        test_sim,
        output_dir / "predictions_cosine.csv",
    )
    print_metrics("CLIP Cosine Similarity", metrics)
    return metrics


def run_logistic(train, test, modality, output_dir, seed):
    if modality == "text":
        x_train = l2_normalize_np(train.text)
        x_test = l2_normalize_np(test.text)
        name = "Text-only Logistic Regression"
        slug = "text_lr"
    else:
        x_train = l2_normalize_np(train.image)
        x_test = l2_normalize_np(test.image)
        name = "Image-only Logistic Regression"
        slug = "image_lr"

    clf = LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
        random_state=seed,
    )
    clf.fit(x_train, train.labels)
    probs = clf.predict_proba(x_test)[:, 1]
    preds = (probs >= 0.5).astype(np.int64)
    metrics = binary_metrics(test.labels, preds)

    save_predictions(test, preds, probs, output_dir / f"predictions_{slug}.csv")
    print_metrics(name, metrics)
    return metrics


def run_neural(
    train,
    val,
    test,
    model,
    name,
    slug,
    output_dir,
    device,
    epochs,
    batch_size,
    lr,
    patience,
):
    model, best_val_f1 = train_model(
        model=model,
        train_split=train,
        val_split=val,
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        patience=patience,
    )

    probs = predict_probabilities(model, test, device, batch_size=batch_size)
    preds = (probs >= 0.5).astype(np.int64)
    metrics = binary_metrics(test.labels, preds)
    metrics["best_val_f1"] = best_val_f1

    save_predictions(test, preds, probs, output_dir / f"predictions_{slug}.csv")
    torch.save(model.state_dict(), output_dir / f"{slug}.pt")
    print_metrics(name, metrics)
    return metrics


def run_ablation_suite(
    train,
    val,
    test,
    output_dir,
    device,
    epochs,
    batch_size,
    lr,
    patience,
):
    rows = []
    for ablation_name, flags in ABLATIONS.items():
        use_cosine, use_abs_diff, use_interaction = flags
        print(f"\n===== Ablation: {ablation_name} =====")
        model = ConsistencyAwareMLP(
            use_cosine=use_cosine,
            use_abs_diff=use_abs_diff,
            use_interaction=use_interaction,
        )
        metrics = run_neural(
            train,
            val,
            test,
            model,
            name=f"Ablation - {ablation_name}",
            slug=f"ablation_{ablation_name}",
            output_dir=output_dir,
            device=device,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            patience=patience,
        )
        rows.append({"experiment": ablation_name, **metrics})

    pd.DataFrame(rows).to_csv(output_dir / "ablation_results.csv", index=False)
    return rows


def parse_args():
    parser = argparse.ArgumentParser(
        description="NewsCLIPpings image-text consistency experiments"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Path to the official news_clippings folder containing data/ and embeddings/.",
    )
    parser.add_argument(
        "--experiment",
        choices=MAIN_EXPERIMENTS + ["all", "ablation", "full"],
        default="all",
    )
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="auto, cpu, cuda, cuda:0, mps, etc.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading NewsCLIPpings merged_balanced splits...")
    train = load_split(args.data_root, "train")
    val = load_split(args.data_root, "val")
    test = load_split(args.data_root, "test")

    print(f"train={len(train):,}, val={len(val):,}, test={len(test):,}")
    print(
        f"train class balance: pristine={(train.labels == 0).mean():.3f}, "
        f"falsified={(train.labels == 1).mean():.3f}"
    )

    device = resolve_device(args.device)
    print(f"Device: {device}")

    rows = []

    experiments_to_run = []
    if args.experiment == "all":
        experiments_to_run = MAIN_EXPERIMENTS
    elif args.experiment == "full":
        experiments_to_run = MAIN_EXPERIMENTS
    elif args.experiment not in ("ablation",):
        experiments_to_run = [args.experiment]

    for experiment in experiments_to_run:
        if experiment == "cosine":
            metrics = run_cosine(val, test, output_dir)

        elif experiment == "text_lr":
            metrics = run_logistic(
                train, test, "text", output_dir, args.seed
            )

        elif experiment == "image_lr":
            metrics = run_logistic(
                train, test, "image", output_dir, args.seed
            )

        elif experiment == "simple_mlp":
            metrics = run_neural(
                train,
                val,
                test,
                SimpleFusionMLP(),
                "Simple CLIP Fusion MLP",
                "simple_mlp",
                output_dir,
                device,
                args.epochs,
                args.batch_size,
                args.lr,
                args.patience,
            )

        elif experiment == "consistency_mlp":
            metrics = run_neural(
                train,
                val,
                test,
                ConsistencyAwareMLP(),
                "Consistency-Aware MLP",
                "consistency_mlp",
                output_dir,
                device,
                args.epochs,
                args.batch_size,
                args.lr,
                args.patience,
            )

        else:
            raise ValueError(experiment)

        rows.append({"experiment": experiment, **metrics})

    if rows:
        pd.DataFrame(rows).to_csv(output_dir / "main_results.csv", index=False)

    if args.experiment in ("ablation", "full"):
        run_ablation_suite(
            train,
            val,
            test,
            output_dir,
            device,
            args.epochs,
            args.batch_size,
            args.lr,
            args.patience,
        )

    print(f"\nFinished. Results saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
