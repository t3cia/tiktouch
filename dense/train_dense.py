from __future__ import annotations

"""Fine-tune a Sentence Transformer on data/emulated_training.jsonl.

This uses MultipleNegativesRankingLoss: every other positive in a minibatch is
an in-batch negative. The generated hard-negative IDs remain available for
future explicit-negative training or evaluation.
"""

import argparse
import hashlib
import json
from pathlib import Path


def load_examples(path: str | Path, validation_fraction: float, seed: int, limit: int | None):
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            # Product-level split prevents duplicate positives crossing splits.
            key = str(row.get("positive_id", row.get("example_id", "")))
            digest = hashlib.blake2b(f"{seed}\0{key}".encode(), digest_size=8).digest()
            is_validation = int.from_bytes(digest, "big") / 2**64 < validation_fraction
            if not is_validation:
                rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/emulated_training.jsonl")
    parser.add_argument("--base-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--output", default="trained-retriever")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--limit", type=int, default=None, help="Smoke-test with only N training rows")
    parser.add_argument("--device", default=None, help="cpu, cuda, or mps")
    args = parser.parse_args()
    if not 0 <= args.validation_fraction < 1:
        parser.error("--validation-fraction must be in [0, 1)")

    try:
        from sentence_transformers import InputExample, SentenceTransformer, losses
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise SystemExit(
            "Install dependencies first: python -m pip install -r dense/requirements.txt"
        ) from exc

    rows = load_examples(args.data, args.validation_fraction, args.seed, args.limit)
    if not rows:
        raise SystemExit("No training rows found")
    examples = [InputExample(texts=[str(row["query"]), str(row["positive_text"])]) for row in rows]
    model = SentenceTransformer(args.base_model, device=args.device)
    loader = DataLoader(
        examples,
        shuffle=True,
        batch_size=args.batch_size,
        drop_last=True,  # MNRL needs at least two distinct examples per batch.
    )
    loss = losses.MultipleNegativesRankingLoss(model)
    warmup_steps = max(1, int(len(loader) * args.epochs * 0.1))
    model.fit(
        train_objectives=[(loader, loss)],
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": args.learning_rate},
        output_path=args.output,
        show_progress_bar=True,
        use_amp=args.device not in (None, "cpu"),
    )
    print(json.dumps({"output": args.output, "rows": len(rows), "epochs": args.epochs}, indent=2))


if __name__ == "__main__":
    main()
