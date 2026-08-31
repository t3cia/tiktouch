from __future__ import annotations

"""Create deterministic bi-encoder training examples from the released data.

The output contains one emulated example for every catalog product plus one
profile-aware example for every public session.  It does not invent labels:
each positive is an actual catalog item, and hard negatives come from the same
fine-grained category when possible.
"""

import argparse
import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Iterator


SPACE_RE = re.compile(r"\s+")
MATERIAL_RE = re.compile(
    r"\b(cotton|polyester|nylon|leather|wool|spandex|silk|rayon|linen|fleece|fabric)\b",
    re.IGNORECASE,
)
COLOR_RE = re.compile(
    r"\b(black|white|blue|red|pink|green|brown|gray|grey|purple|yellow|orange|beige|navy)\b",
    re.IGNORECASE,
)
GENERIC_CATEGORIES = {
    "clothing",
    "clothing shoes & jewelry",
    "clothing, shoes & jewelry",
}
SCENARIOS = ("buying", "browsing", "intent_override", "boundary")


def clean(value: object, limit: int = 360) -> str:
    if value in (None, "", []):
        return ""
    if isinstance(value, dict):
        value = "; ".join(
            f"{key}: {item}" for key, item in value.items() if item not in (None, "", [])
        )
    elif isinstance(value, list):
        value = "; ".join(str(item) for item in value if item not in (None, ""))
    return SPACE_RE.sub(" ", str(value)).strip(" -;,.")[:limit].rstrip()


def load_jsonl(path: str | Path) -> Iterator[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def leaf_category(product: dict) -> str:
    categories = [clean(value, 100) for value in product.get("categories") or []]
    useful = [value for value in categories if value and value.lower() not in GENERIC_CATEGORIES]
    return useful[-1] if useful else "clothing item"


def category_phrase(product: dict) -> str:
    categories = [clean(value, 100) for value in product.get("categories") or []]
    useful = [value for value in categories if value and value.lower() not in GENERIC_CATEGORIES]
    return " ".join(useful[-2:]).lower() if useful else "clothing item"


def product_text(product: dict, max_chars: int = 1800) -> str:
    """Canonical document representation shared with retrieve.py."""
    sections = [
        ("title", clean(product.get("title"), 400)),
        ("category", clean(product.get("categories"), 300)),
        ("brand", clean(product.get("store"), 120)),
        ("features", clean(product.get("features"), 800)),
        ("details", clean(product.get("details"), 600)),
        ("description", clean(product.get("description"), 800)),
    ]
    if product.get("price") not in (None, ""):
        sections.append(("price", f"${product['price']}"))
    text = " | ".join(f"{name}: {value}" for name, value in sections if value)
    return text[:max_chars].rstrip()


def constraints(product: dict) -> list[str]:
    corpus = product_text(product, 3000)
    values: list[str] = []
    material = MATERIAL_RE.search(corpus)
    color = COLOR_RE.search(corpus)
    if material:
        values.append(material.group(1).lower())
    if color:
        values.append(f"{color.group(1).lower()} color")
    values.extend(clean(item, 180) for item in (product.get("features") or [])[:4])
    details = product.get("details") or {}
    if isinstance(details, dict):
        values.extend(clean(f"{key}: {value}", 180) for key, value in list(details.items())[:4])
    if product.get("price") not in (None, ""):
        values.append(f"around ${product['price']}")
    return list(dict.fromkeys(value for value in values if value))


def stable_rng(seed: int, key: str) -> random.Random:
    digest = hashlib.blake2b(f"{seed}\0{key}".encode(), digest_size=8).digest()
    return random.Random(int.from_bytes(digest, "big"))


def scenario_for(parent_asin: str, seed: int) -> str:
    # Match the official 40/40/15/5 scenario mix over the full generated set.
    value = stable_rng(seed, parent_asin).random()
    if value < 0.40:
        return "buying"
    if value < 0.80:
        return "browsing"
    if value < 0.95:
        return "intent_override"
    return "boundary"


def emulate_query(product: dict, scenario: str, rng: random.Random) -> tuple[str, list[str]]:
    category = category_phrase(product)
    values = constraints(product)
    first = values[0] if values else clean(product.get("title"), 180)
    second = values[1] if len(values) > 1 else first
    brand = clean(product.get("store"), 100)
    turns: list[str]
    if scenario == "buying":
        turns = [f"I'm looking for {category}.", f"A key requirement is {first}."]
    elif scenario == "browsing":
        turns = [
            f"I'm exploring options for {category}.",
            f"I tend to prefer {first}.",
        ]
    elif scenario == "intent_override":
        old = brand or "a simple everyday style"
        turns = [
            f"I need {category} and was initially considering {old}.",
            f"Actually, ignore that earlier preference; prioritize {first} and {second}.",
        ]
    else:
        turns = [
            f"I'm looking for {category}.",
            "I don't have a preference for the other details; use your judgment.",
            f"The one thing that matters is {first}.",
        ]
    # Small deterministic surface variation prevents every query having one prefix.
    if rng.random() < 0.5:
        turns[0] = turns[0].replace("I'm looking for", "Please help me find")
    return " ".join(turns), turns


def choose_negatives(
    parent_asin: str,
    category: str,
    category_ids: dict[str, list[str]],
    all_ids: list[str],
    count: int,
    rng: random.Random,
) -> list[str]:
    pool = category_ids[category]
    # Sampling only O(count) candidates avoids copying a large category once per product.
    sampled = rng.sample(pool, min(len(pool), count + 1)) if pool else []
    negatives = [item for item in sampled if item != parent_asin][:count]
    if len(negatives) < count:
        # Fill sparse categories without repeatedly sampling or producing duplicates.
        start = rng.randrange(len(all_ids)) if all_ids else 0
        for offset in range(len(all_ids)):
            candidate = all_ids[(start + offset) % len(all_ids)]
            if candidate != parent_asin and candidate not in negatives:
                negatives.append(candidate)
                if len(negatives) == count:
                    break
    return negatives


def public_query(product: dict, sample: dict, rng: random.Random) -> tuple[str, list[str]]:
    scenario = str(sample.get("scenario_type") or "buying")
    query, turns = emulate_query(product, scenario, rng)
    profile = sample.get("user_profile") or {}
    tags = ", ".join(str(value) for value in profile.get("preference_tags") or [])
    summary = clean(profile.get("summary"), 220)
    context = summary or (f"Past preferences: {tags}." if tags else "")
    if context:
        query = f"{query} Shopper profile: {context}"
    return query, turns


def generate(
    catalog_path: str | Path,
    public_path: str | Path,
    output_path: str | Path,
    negative_count: int = 7,
    seed: int = 2026,
    max_products: int | None = None,
) -> dict:
    products = list(load_jsonl(catalog_path))
    if max_products is not None:
        products = products[:max_products]
    by_id = {str(product["parent_asin"]): product for product in products}
    all_ids = list(by_id)
    category_ids: dict[str, list[str]] = defaultdict(list)
    for product in products:
        category_ids[leaf_category(product)].append(str(product["parent_asin"]))

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    counts = {scenario: 0 for scenario in SCENARIOS}
    written = 0
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for product in products:
            parent_asin = str(product["parent_asin"])
            rng = stable_rng(seed, f"catalog:{parent_asin}")
            scenario = scenario_for(parent_asin, seed)
            query, turns = emulate_query(product, scenario, rng)
            row = {
                "example_id": f"catalog:{parent_asin}",
                "source": "catalog_emulation",
                "scenario_type": scenario,
                "query": query,
                "turns": turns,
                "positive_id": parent_asin,
                "positive_text": product_text(product),
                "hard_negative_ids": choose_negatives(
                    parent_asin,
                    leaf_category(product),
                    category_ids,
                    all_ids,
                    negative_count,
                    rng,
                ),
            }
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            counts[scenario] += 1
            written += 1

        if Path(public_path).exists():
            for sample in load_jsonl(public_path):
                parent_asin = str((sample.get("ground_truth") or {}).get("parent_asin", ""))
                product = by_id.get(parent_asin)
                if product is None:
                    continue
                sample_id = str(sample.get("sample_id") or written)
                rng = stable_rng(seed, f"public:{sample_id}")
                query, turns = public_query(product, sample, rng)
                scenario = str(sample.get("scenario_type") or "buying")
                row = {
                    "example_id": f"public:{sample_id}",
                    "source": "public_session_emulation",
                    "scenario_type": scenario,
                    "difficulty_bucket": sample.get("difficulty_bucket"),
                    "query": query,
                    "turns": turns,
                    "profile": sample.get("user_profile") or {},
                    "positive_id": parent_asin,
                    "positive_text": product_text(product),
                    "hard_negative_ids": choose_negatives(
                        parent_asin,
                        leaf_category(product),
                        category_ids,
                        all_ids,
                        negative_count,
                        rng,
                    ),
                }
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                counts[scenario] = counts.get(scenario, 0) + 1
                written += 1

    return {
        "output": str(output),
        "examples": written,
        "catalog_products": len(products),
        "scenario_counts": counts,
        "hard_negatives_per_example": negative_count,
        "seed": seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--public", default="data/public_set.jsonl")
    parser.add_argument("--output", default="data/emulated_training.jsonl")
    parser.add_argument("--hard-negatives", type=int, default=7)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-products", type=int, default=None, help="Useful for a quick smoke test")
    args = parser.parse_args()
    if args.hard_negatives < 0:
        parser.error("--hard-negatives must be non-negative")
    summary = generate(
        args.catalog,
        args.public,
        args.output,
        args.hard_negatives,
        args.seed,
        args.max_products,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
