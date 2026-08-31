from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

try:  # Works both as `python dense/validate_dense.py` and as a package import.
    from .retrieve import DenseRetriever, encode_sentences
except ImportError:  # pragma: no cover - script execution path
    from retrieve import DenseRetriever, encode_sentences


def validation_rows(path: str | Path, fraction: float, seed: int):
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            key = str(row.get("positive_id", row.get("example_id", "")))
            digest = hashlib.blake2b(f"{seed}\0{key}".encode(), digest_size=8).digest()
            if int.from_bytes(digest, "big") / 2**64 < fraction:
                yield row


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate dense retrieval Recall@K and MRR")
    parser.add_argument("--data", default="data/emulated_training.jsonl")
    parser.add_argument("--index-dir", default="dense_index")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--device", default=None)
    parser.add_argument("--fraction", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()
    if not 0 < args.fraction <= 1 or args.top_k < 1:
        parser.error("fraction must be in (0, 1] and top-k must be positive")

    rows = list(validation_rows(args.data, args.fraction, 2026))
    retriever = DenseRetriever(args.model, args.index_dir, device=args.device)
    retriever.load()
    assert retriever._embeddings is not None and retriever._ids is not None
    id_to_index = {parent_asin: index for index, parent_asin in enumerate(retriever._ids)}
    hits = 0
    reciprocal_rank = 0.0
    evaluated = 0
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        queries = encode_sentences(
            retriever.model,
            [str(row["query"]) for row in batch],
            batch_size=args.batch_size,
        )
        scores = queries @ np.asarray(retriever._embeddings, dtype=np.float32).T
        k = min(args.top_k, scores.shape[1])
        candidates = np.argpartition(scores, -k, axis=1)[:, -k:]
        for row, indexes, row_scores in zip(batch, candidates, scores):
            ranked = indexes[np.argsort(-row_scores[indexes], kind="stable")]
            target_index = id_to_index.get(str(row["positive_id"]))
            if target_index is None:
                continue
            evaluated += 1
            matches = np.flatnonzero(ranked == target_index)
            if len(matches):
                rank = int(matches[0]) + 1
                hits += 1
                reciprocal_rank += 1.0 / rank
    result = {
        "examples": evaluated,
        f"recall_at_{args.top_k}": round(hits / evaluated, 6) if evaluated else 0.0,
        "mrr": round(reciprocal_rank / evaluated, 6) if evaluated else 0.0,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
