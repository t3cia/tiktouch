from __future__ import annotations

"""Local Sentence Transformer dense retrieval for the frozen product catalog.

Examples:
  python dense/retrieve.py index --model sentence-transformers/all-MiniLM-L6-v2
  python dense/retrieve.py search --model ./my-trained-model --query "black leather hiking boots"

Use the same model (or trained checkpoint) for indexing and querying.
"""

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence


SPACE_RE = re.compile(r"\s+")
TEXT_FORMAT_VERSION = 1
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _clean(value: object, limit: int) -> str:
    if value in (None, "", []):
        return ""
    if isinstance(value, dict):
        value = "; ".join(
            f"{key}: {item}" for key, item in value.items() if item not in (None, "", [])
        )
    elif isinstance(value, list):
        value = "; ".join(str(item) for item in value if item not in (None, ""))
    return SPACE_RE.sub(" ", str(value)).strip(" -;,.")[:limit].rstrip()


def product_to_text(product: dict, max_chars: int = 1800) -> str:
    """Serialize all useful released fields into a stable embedding document."""
    sections = [
        ("title", _clean(product.get("title"), 400)),
        ("category", _clean(product.get("categories"), 300)),
        ("brand", _clean(product.get("store"), 120)),
        ("features", _clean(product.get("features"), 800)),
        ("details", _clean(product.get("details"), 600)),
        ("description", _clean(product.get("description"), 800)),
    ]
    if product.get("price") not in (None, ""):
        sections.append(("price", f"${product['price']}"))
    return " | ".join(f"{name}: {value}" for name, value in sections if value)[:max_chars].rstrip()


def sentence_transformer(model_name_or_path: str = DEFAULT_MODEL, device: str | None = None):
    """Lazy helper so importing retrieve.py does not require ML dependencies."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Install dense-retrieval dependencies with: "
            "python -m pip install sentence-transformers numpy"
        ) from exc
    return SentenceTransformer(model_name_or_path, device=device)


def encode_sentences(
    model,
    texts: Sequence[str] | Iterable[str],
    *,
    batch_size: int = 64,
    show_progress_bar: bool = False,
):
    """Return normalized float32 NumPy embeddings suitable for cosine search."""
    return model.encode(
        list(texts),
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=show_progress_bar,
    )


def compose_query(
    messages: str | Sequence[str],
    profile: dict | None = None,
    *,
    max_chars: int = 1400,
) -> str:
    """Compose current dialogue state, retaining corrections near the end."""
    turns = [messages] if isinstance(messages, str) else list(messages)
    turns = [_clean(turn, 500) for turn in turns if _clean(turn, 500)]
    profile_parts: list[str] = []
    if profile:
        tags = _clean(profile.get("preference_tags"), 180)
        summary = _clean(profile.get("summary"), 260)
        if tags:
            profile_parts.append(f"preference tags: {tags}")
        if summary:
            profile_parts.append(f"profile: {summary}")
    dialogue = " ".join(f"turn {index + 1}: {turn}" for index, turn in enumerate(turns))
    result = " | ".join([*profile_parts, dialogue])
    # Latest messages include overrides, so preserve the tail when truncating.
    return result[-max_chars:].lstrip()


def _load_catalog(path: str | Path) -> Iterator[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


@dataclass(frozen=True)
class SearchResult:
    parent_asin: str
    score: float
    product: dict | None = None

    def as_dict(self) -> dict:
        result = {"parent_asin": self.parent_asin, "score": self.score}
        if self.product is not None:
            result["product"] = self.product
        return result


class DenseRetriever:
    """Exact cosine retriever over a memory-mapped embedding matrix."""

    def __init__(
        self,
        model_name_or_path: str = DEFAULT_MODEL,
        index_dir: str | Path = "dense_index",
        *,
        device: str | None = None,
        search_chunk_size: int = 16384,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.index_dir = Path(index_dir)
        self.device = device
        self.search_chunk_size = search_chunk_size
        self._model = None
        self._embeddings = None
        self._ids: list[str] | None = None
        self._products: list[dict] | None = None

    @property
    def model(self):
        if self._model is None:
            self._model = sentence_transformer(self.model_name_or_path, self.device)
        return self._model

    def build(
        self,
        catalog_path: str | Path = "data/catalog.jsonl",
        *,
        batch_size: int = 64,
        store_dtype: str = "float32",
    ) -> dict:
        import numpy as np

        products = list(_load_catalog(catalog_path))
        if not products:
            raise ValueError(f"Catalog is empty: {catalog_path}")
        texts = [product_to_text(product) for product in products]
        embeddings = encode_sentences(
            self.model,
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
        )
        if store_dtype not in {"float16", "float32"}:
            raise ValueError("store_dtype must be float16 or float32")
        embeddings = embeddings.astype(store_dtype)

        self.index_dir.mkdir(parents=True, exist_ok=True)
        np.save(self.index_dir / "embeddings.npy", embeddings, allow_pickle=False)
        with (self.index_dir / "ids.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
            for product in products:
                handle.write(json.dumps({"parent_asin": str(product["parent_asin"])}) + "\n")
        metadata = {
            "model": self.model_name_or_path,
            "catalog": str(catalog_path),
            "count": len(products),
            "dimension": int(embeddings.shape[1]),
            "dtype": str(embeddings.dtype),
            "normalized": True,
            "text_format_version": TEXT_FORMAT_VERSION,
        }
        (self.index_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        self._embeddings = np.load(self.index_dir / "embeddings.npy", mmap_mode="r")
        self._ids = [str(product["parent_asin"]) for product in products]
        self._products = products
        return metadata

    def load(self, *, load_products: bool = False, catalog_path: str | Path | None = None) -> None:
        import numpy as np

        metadata_path = self.index_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"No dense index found in {self.index_dir}; run the index command first")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("text_format_version") != TEXT_FORMAT_VERSION:
            raise ValueError("Index text format is incompatible; rebuild the index")
        if str(metadata.get("model")) != str(self.model_name_or_path):
            raise ValueError(
                "This index was produced by a different model. Rebuild it with the current "
                f"checkpoint (index={metadata.get('model')!r}, current={self.model_name_or_path!r})."
            )
        self._embeddings = np.load(self.index_dir / "embeddings.npy", mmap_mode="r")
        self._ids = [
            str(json.loads(line)["parent_asin"])
            for line in (self.index_dir / "ids.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(self._ids) != self._embeddings.shape[0]:
            raise ValueError("Index is corrupt: ID and embedding counts differ")
        if load_products:
            selected_catalog = catalog_path or metadata.get("catalog") or "data/catalog.jsonl"
            self._products = list(_load_catalog(selected_catalog))
            if len(self._products) != len(self._ids):
                raise ValueError("Catalog and dense index counts differ")

    def search(
        self,
        query: str,
        top_k: int = 10,
        *,
        batch_size: int = 64,
        include_products: bool = False,
    ) -> list[SearchResult]:
        import numpy as np

        if top_k <= 0:
            return []
        if self._embeddings is None or self._ids is None:
            self.load(load_products=include_products)
        if not query.strip():
            return []
        assert self._embeddings is not None and self._ids is not None
        query_vector = encode_sentences(self.model, [query], batch_size=batch_size)[0].astype("float32")
        candidate_indices = np.empty(0, dtype=np.int64)
        candidate_scores = np.empty(0, dtype=np.float32)
        k = min(top_k, len(self._ids))

        # Chunking bounds temporary float32 conversion when an index is stored as float16.
        for start in range(0, len(self._ids), self.search_chunk_size):
            stop = min(start + self.search_chunk_size, len(self._ids))
            scores = np.asarray(self._embeddings[start:stop], dtype=np.float32) @ query_vector
            local_k = min(k, len(scores))
            local = np.argpartition(scores, -local_k)[-local_k:] if local_k < len(scores) else np.arange(len(scores))
            candidate_indices = np.concatenate((candidate_indices, local.astype(np.int64) + start))
            candidate_scores = np.concatenate((candidate_scores, scores[local]))

        order = np.argsort(-candidate_scores, kind="stable")[:k]
        results: list[SearchResult] = []
        for position in order:
            index = int(candidate_indices[position])
            product = self._products[index] if include_products and self._products is not None else None
            results.append(SearchResult(self._ids[index], float(candidate_scores[position]), product))
        return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="Embed and save the catalog")
    index_parser.add_argument("--catalog", default="data/catalog.jsonl")
    index_parser.add_argument("--index-dir", default="dense_index")
    index_parser.add_argument("--model", default=DEFAULT_MODEL)
    index_parser.add_argument("--device", default=None, help="For example cuda, mps, or cpu")
    index_parser.add_argument("--batch-size", type=int, default=64)
    index_parser.add_argument("--dtype", choices=("float16", "float32"), default="float32")

    search_parser = subparsers.add_parser("search", help="Search a previously built index")
    search_parser.add_argument("--query", required=True)
    search_parser.add_argument("--index-dir", default="dense_index")
    search_parser.add_argument("--model", default=DEFAULT_MODEL)
    search_parser.add_argument("--device", default=None)
    search_parser.add_argument("--top-k", type=int, default=10)
    search_parser.add_argument("--catalog", default=None, help="Include full product records")
    args = parser.parse_args()

    retriever = DenseRetriever(args.model, args.index_dir, device=args.device)
    if args.command == "index":
        metadata = retriever.build(args.catalog, batch_size=args.batch_size, store_dtype=args.dtype)
        print(json.dumps(metadata, indent=2))
    else:
        retriever.load(load_products=bool(args.catalog), catalog_path=args.catalog)
        results = retriever.search(args.query, args.top_k, include_products=bool(args.catalog))
        print(json.dumps([result.as_dict() for result in results], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
