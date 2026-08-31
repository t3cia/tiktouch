# Preference-Aware Reranking

## Overview

TikTouch uses a two-stage product-ranking pipeline:

1. BM25 retrieves a candidate pool using SQLite FTS5.
2. A preference-aware reranker adjusts the order using the user's structured conversational memory.

The purpose of reranking is to prioritise products that satisfy the user's latest preferences without discarding the strong lexical relevance provided by BM25.

## Inputs

### Candidate products

Each recommendation contains:

```python
{
    "parent_asin": str,
    "title": str,
    "categories": str,
    "features": str,
    "details": str,
    "store": str,
    "description": str,
    "price": float | None,
    "search_score": float,
}
```

### Conversational memory

The ranker uses the following memory fields:

```python
{
    "category": str | None,
    "material": str | None,
    "color": str | None,
    "size": str | None,
    "style": str | None,
    "brand": str | None,
    "budget": int | None,
    "feature": str | None,
    "use_case": str | None,
    "requirements": list[str],
}
```

The `requirements` list identifies fields that the user described as hard requirements.

## Ranking Process

### 1. Preserve BM25 recall

SQLite FTS5 returns lower BM25 scores for more relevant products.

The candidates are first sorted by their BM25 score:

```python
recommendations = sorted(
    recommendations,
    key=lambda product: product["search_score"],
)
```

The reranker only reorders BM25's original top 10 products. This prevents weaker candidates from positions 11–50 from displacing products that BM25 already considered highly relevant.

### 2. Normalise BM25 scores

Because lower FTS5 BM25 values are better, the scores are inverted and normalised:

```text
lexical_score = normalise(-BM25)
```

This produces a lexical relevance score between zero and one.

### 3. Match preferences against relevant fields

Each memory attribute is checked against the most appropriate product metadata.

| Memory field | Product fields searched |
|---|---|
| Color | title, features, details, description |
| Material | features, details, description |
| Brand | title, store |
| Feature | features, details, description |
| Style | title, features, details, description |
| Use case | title, features, details, description |
| Size | title, features, details, description |
| Budget | price |

Category is not used as a direct reranking bonus because product catalogues can contain broad category paths such as `Clothing, Shoes & Jewelry`. These paths can create misleading matches.

### 4. Calculate preference bonuses

A product receives a small bonus when it matches a stored preference.

```text
final score =
0.80 x normalised BM25 score
+ preference bonuses
```

Hard requirements receive a larger bonus than ordinary preferences:

```python
boost = soft_weight * (
    3.0 if field in hard_requirements else 1.0
)
```

This allows explicit requirements to influence ranking more strongly without filtering products out completely.

### 5. Handle budget constraints

If a product has a known price:

- Products within budget receive a bonus.
- Products over budget receive a penalty.
- Products with missing prices are retained without a budget adjustment.

Unknown prices are not treated as failures because the catalogue contains many products without reliable price information.

### 6. Handle free-text preferences

Open-ended values such as feature, style, and use case may be stored from clarification replies.

The reranker removes conversational wrapper text such as:

```text
For that, what matters is:
```

It also separates multiple constraints divided by semicolons before matching them against product metadata.

### 7. Return final results

Products are sorted by their final reranking score, and the strongest ten are returned:

```python
return sorted(
    ranked,
    key=lambda product: product["rank_score"],
    reverse=True,
)[:10]
```

## Design Decisions

### Why use bonuses instead of hard filtering?

Early experiments removed products that did not appear to satisfy hard requirements. This caused a large performance decrease because catalogue metadata was often incomplete or ambiguous.

The final implementation uses ranking bonuses and penalties instead. This preserves retrieval recall while still allowing user preferences to influence ordering.

### Why rerank only the original top 10?

Reranking all 50 candidates initially reduced Hit Rate at 10 because weak candidates could displace relevant BM25 results.

Restricting reranking to BM25's original top 10 preserves its candidate set while allowing preference matches to improve Mean Reciprocal Rank.

## Evaluation

The reranker provides a small positive improvement by refining the order of already-relevant BM25 candidates.