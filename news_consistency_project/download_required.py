from pathlib import Path
from urllib.request import urlretrieve

BASE = "https://huggingface.co/g-luo/news-clippings/resolve/main"
ROOT = Path("news_clippings")
SPLITS = ["train", "val", "test"]


def download(url: str, target: Path):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0:
        print(f"Skip existing: {target}")
        return
    print(f"Downloading: {target}")
    urlretrieve(url, target)


def main():
    # Only the subset needed by this project.
    for split in SPLITS:
        rel = f"data/merged_balanced/{split}.json"
        download(f"{BASE}/{rel}?download=true", ROOT / rel)

    for embedding in ["clip_image_embeddings", "clip_text_embeddings"]:
        for split in SPLITS:
            filename = f"{embedding}_{split}.pkl"
            rel = f"embeddings/{embedding}/{filename}"
            download(f"{BASE}/{rel}?download=true", ROOT / rel)

    print(f"\nDone. Data root: {ROOT.resolve()}")


if __name__ == "__main__":
    main()
