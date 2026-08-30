# BM25 validation protocol

The public 200-session score is a development result, not an unbiased estimate of
the private score. To reduce tuning against those same examples, run:

```powershell
python -m evaluator.cross_validate_bm25
```

The harness uses five deterministic folds grouped by target `parent_asin` and
stratified by scenario and difficulty. For each fold it selects among only three
predeclared field-weight configurations using the other four folds, then evaluates
the selected configuration on the held-out fold. A deliberately small candidate
set limits the capacity of the hyperparameter search itself.

It also repeats the full evaluation with deterministic surface paraphrases. The
constraints and labels are unchanged, so this measures sensitivity to conversational
wording rather than semantic difficulty. Results are written to
`docs/bm25_validation.json`.

## Current result

- The `feature_aware` weights were selected independently in all five folds.
- Cross-validated Hit Rate@10: `0.970`
- Cross-validated MRR: `0.520956`
- Cross-validated technical score: `0.782687`
- Fold-score standard deviation: `0.023984`
- Paraphrased Hit Rate@10: `0.930`
- Paraphrased technical score: `0.741348`

The default agent therefore uses `feature_aware` weights. Its clarification policy
asks specific attributes in a fixed order and uses `other` only as a late fallback;
it does not select a question policy by repeatedly optimizing the public score.

## Remaining limitations

Cross-validation reduces selection bias but cannot remove dataset-shift risk because
all folds come from the same public simulator. The paraphrase suite is deliberately
small and should not become another tuning target. After this configuration is
locked, the organizer's private 800 sessions remain the only unbiased final test.
