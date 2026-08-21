import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Inspect correct/error cases.")
    p.add_argument("--predictions", required=True, help="A predictions_*.csv file")
    p.add_argument(
        "--visual-news-json",
        default=None,
        help="Optional VisualNews origin/data.json. If provided, captions and image paths are added.",
    )
    p.add_argument("--output", default="outputs/error_cases.csv")
    p.add_argument("--top-k", type=int, default=50)
    return p.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.predictions)

    # False positive: pristine sample predicted as falsified.
    fp = df[(df["label_falsified"] == 0) & (df["prediction_falsified"] == 1)].copy()
    fp["error_type"] = "false_positive"

    # False negative: falsified sample predicted as pristine.
    fn = df[(df["label_falsified"] == 1) & (df["prediction_falsified"] == 0)].copy()
    fn["error_type"] = "false_negative"

    # Cases near the decision boundary are often useful for qualitative analysis.
    fp = fp.sort_values("model_score").head(args.top_k)
    fn = fn.sort_values("model_score", ascending=False).head(args.top_k)
    errors = pd.concat([fp, fn], ignore_index=True)

    if args.visual_news_json:
        with open(args.visual_news_json, "r", encoding="utf-8") as f:
            visual_news = json.load(f)
        mapping = {str(item["id"]): item for item in visual_news}

        def caption_for(text_id):
            item = mapping.get(str(text_id), {})
            return item.get("caption", "")

        def image_path_for(image_id):
            item = mapping.get(str(image_id), {})
            return item.get("image_path", "")

        errors["caption"] = errors["text_id"].map(caption_for)
        errors["image_path"] = errors["image_id"].map(image_path_for)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    errors.to_csv(out, index=False)

    print(f"Saved {len(errors)} error cases to {out.resolve()}")
    print(errors.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
