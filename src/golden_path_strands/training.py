"""Simple training workflow stub."""
from __future__ import annotations

from pathlib import Path


def train(dataset_path: str) -> int:
    """Pretend to train a model using the given dataset.

    The function counts the number of examples in the dataset and returns that count.
    """
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(dataset_path)
    count = sum(1 for _ in path.open())
    # In real life, launch a fine-tuning job here
    return count


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train model on golden paths dataset")
    parser.add_argument("--dataset", default="dataset.jsonl", help="Path to dataset JSONL")
    args = parser.parse_args()
    count = train(args.dataset)
    print(f"Training invoked on {args.dataset} with {count} examples")


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
