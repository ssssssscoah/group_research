# NewsCLIPpings consistency experiment

## 1. Environment

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

## 2. Expected official NewsCLIPpings layout

```text
news_clippings/
├── data/
│   └── merged_balanced/
│       ├── train.json
│       ├── val.json
│       └── test.json
└── embeddings/
    ├── clip_image_embeddings/
    │   ├── clip_image_embeddings_train.pkl
    │   ├── clip_image_embeddings_val.pkl
    │   └── clip_image_embeddings_test.pkl
    └── clip_text_embeddings/
        ├── clip_text_embeddings_train.pkl
        ├── clip_text_embeddings_val.pkl
        └── clip_text_embeddings_test.pkl
```

## 3. Quick feasibility check

```bash
python main.py --data-root /path/to/news_clippings --experiment cosine
python main.py --data-root /path/to/news_clippings --experiment text_lr
python main.py --data-root /path/to/news_clippings --experiment image_lr
```

## 4. Main neural comparison

```bash
python main.py --data-root /path/to/news_clippings --experiment simple_mlp
python main.py --data-root /path/to/news_clippings --experiment consistency_mlp
```

## 5. Run all five main experiments

```bash
python main.py --data-root /path/to/news_clippings --experiment all
```

## 6. Ablation study

```bash
python main.py --data-root /path/to/news_clippings --experiment ablation
```

## 7. Everything

```bash
python main.py --data-root /path/to/news_clippings --experiment full
```

Outputs are written to `outputs/` by default.

## 8. Optional: download only the required NewsCLIPpings subset

```bash
python download_required.py
```

This downloads only `merged_balanced` plus CLIP image/text embeddings.

## 9. Error analysis

Without VisualNews metadata:

```bash
python error_analysis.py --predictions outputs/predictions_consistency_mlp.csv
```

If you also downloaded VisualNews and have `visual_news/origin/data.json`:

```bash
python error_analysis.py \
  --predictions outputs/predictions_consistency_mlp.csv \
  --visual-news-json /path/to/visual_news/origin/data.json
```
