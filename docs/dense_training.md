# Emulated dense-retrieval data

`data/emulated_training.jsonl` is deterministic and contains:

- one emulated conversational query for every released catalog product;
- one extra profile-aware query for every public session whose target is in the catalog;
- the official 40/40/15/5 scenario mix for catalog-derived rows;
- a positive product ID and canonical positive document text;
- seven hard-negative IDs sampled from the same leaf category where possible.

Each row has this shape:

```json
{
  "example_id": "catalog:B07K34RX5J",
  "source": "catalog_emulation",
  "scenario_type": "buying",
  "query": "Please help me find earrings ...",
  "turns": ["Please help me find earrings.", "A key requirement is fabric."],
  "positive_id": "B07K34RX5J",
  "positive_text": "title: ... | category: ... | features: ...",
  "hard_negative_ids": ["B...", "B..."]
}
```

The simplest Sentence Transformers training setup uses `query` and
`positive_text` with `MultipleNegativesRankingLoss`, treating other positives in
the batch as negatives. The supplied hard-negative IDs can be joined back to
`data/catalog.jsonl` with `dense.retrieve.product_to_text` for a triplet or ranking
loss. Keep validation examples grouped by `positive_id` to prevent the same
product appearing in both training and validation.

The included trainer does this baseline fine-tuning:

```powershell
python dense/train_dense.py --base-model sentence-transformers/all-MiniLM-L6-v2 `
  --output C:\models\tiktouch-retriever --device cuda --epochs 2
```

For a quick smoke test, add `--limit 512 --epochs 1`. Start with a batch size
that fits GPU memory (32 or 64); larger batches provide more in-batch negatives.

Validate dense retrieval before integrating it into the conversational agent:

```powershell
python dense/validate_dense.py --model C:\models\tiktouch-retriever `
  --index-dir dense_index --device cuda --top-k 10
```

This reports Recall@10 and MRR on a deterministic 10% product-held-out split.
The official conversational score still requires an Agent wrapper and should be
run with `python -m evaluator.local_evaluator` after replacing the starter's
BM25 search with calls to `DenseRetriever.search`.

Regenerate the artifact at any time:

```powershell
python dense/generate_emulated_training.py
```

Install retrieval dependencies and build an index from the trained checkpoint:

```powershell
python -m pip install -r dense/requirements.txt
python dense/retrieve.py index --model C:\path\to\trained-model --device cuda
python dense/retrieve.py search --model C:\path\to\trained-model --query "women's waterproof hiking boots"
```

The catalog must be re-indexed after training because query and product vectors
must come from the same checkpoint. Use `--dtype float16` when disk space matters;
the default float32 index is roughly 77 MB for a 384-dimensional model and 50,000
products.
