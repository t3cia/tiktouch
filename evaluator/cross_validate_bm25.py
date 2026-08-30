from __future__ import annotations

"""Grouped BM25 model selection plus deterministic paraphrase robustness checks."""

import argparse
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import evaluator.local_evaluator as local
from starter.agent import Agent, AgentConfig


CONFIGS = {
    # Keep this family intentionally small: a large grid would overfit 200 sessions.
    "starter_weights": AgentConfig((6.0, 4.0, 2.5, 2.5, 1.5, 1.0)),
    "balanced": AgentConfig((6.0, 6.0, 4.0, 3.0, 2.5, 1.0)),
    "feature_aware": AgentConfig((6.0, 6.0, 6.0, 3.0, 3.0, 1.0)),
}


def technical_summary(sessions: list[dict]) -> dict:
    summary = local.metric_summary(sessions)
    mttc = float(summary["mttc"])
    efficiency = max(0.0, min(1.0, (11.0 - mttc) / 10.0))
    score = 0.50 * summary["hit_rate_at_10"] + 0.30 * summary["mrr"] + 0.20 * efficiency
    return {**summary, "efficiency": round(efficiency, 6), "technical_score": round(score, 6)}


def grouped_folds(samples: list[dict], folds: int, seed: int) -> dict[str, int]:
    """Stratify by scenario/difficulty while grouping identical product targets."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for sample in samples:
        groups[str(sample["ground_truth"]["parent_asin"])].append(sample)

    strata: dict[tuple[str, str], list[tuple[str, list[dict]]]] = defaultdict(list)
    for target, members in groups.items():
        first = members[0]
        key = (str(first.get("scenario_type", "")), str(first.get("difficulty_bucket", "")))
        strata[key].append((target, members))

    assignment: dict[str, int] = {}
    fold_sizes = [0] * folds
    for key in sorted(strata):
        ordered = sorted(
            strata[key],
            key=lambda item: hashlib.blake2b(
                f"{seed}\0{item[0]}".encode(), digest_size=8
            ).digest(),
        )
        for target, members in ordered:
            fold = min(range(folds), key=lambda value: fold_sizes[value])
            fold_sizes[fold] += len(members)
            for sample in members:
                assignment[str(sample["sample_id"])] = fold
    return assignment


def evaluate_config(
    config: AgentConfig,
    samples: list[dict],
    catalog_path: str,
    catalog_ids: set[str],
    categories: dict[str, list[str]],
    products: dict[str, dict],
) -> dict:
    return local.evaluate(Agent(catalog_path, config), samples, catalog_ids, categories, products)


def paraphrase_mode(enabled: bool):
    """Context manager-like helper that swaps only message surface forms."""
    original_initial = local.initial_message
    original_reply = local.customer_reply
    if not enabled:
        return original_initial, original_reply

    def initial(sample, category, disclosed):
        message = original_initial(sample, category, disclosed)
        replacements = (
            ("I'm looking for", "Could you recommend"),
            ("A key requirement is:", "It must have"),
            (", but I'm still exploring.", "; I am open to suggestions."),
        )
        for old, new in replacements:
            message = message.replace(old, new)
        return message

    def reply(sample, ask_attribute, disclosed, boundary_used):
        message, used = original_reply(sample, ask_attribute, disclosed, boundary_used)
        replacements = (
            ("For that, what matters is:", "My preference there is"),
            ("I don't have an additional preference for", "I have no extra requirement concerning"),
            ("Those options are not quite right yet.", "These are not suitable yet."),
        )
        for old, new in replacements:
            message = message.replace(old, new)
        return message, used

    local.initial_message = initial
    local.customer_reply = reply
    return original_initial, original_reply


def run(args) -> dict:
    samples = local.load_jsonl(args.dataset)
    catalog_ids, categories, products = local.catalog_index(args.catalog)
    assignments = grouped_folds(samples, args.folds, args.seed)
    original_results: dict[str, dict] = {}
    robust_results: dict[str, dict] = {}

    for name, config in CONFIGS.items():
        original_results[name] = evaluate_config(
            config, samples, args.catalog, catalog_ids, categories, products
        )

    saved_initial, saved_reply = paraphrase_mode(True)
    try:
        for name, config in CONFIGS.items():
            robust_results[name] = evaluate_config(
                config, samples, args.catalog, catalog_ids, categories, products
            )
    finally:
        local.initial_message = saved_initial
        local.customer_reply = saved_reply

    fold_reports = []
    selected_sessions: list[dict] = []
    selections: Counter[str] = Counter()
    for fold in range(args.folds):
        validation_ids = {sample_id for sample_id, value in assignments.items() if value == fold}
        train_scores: dict[str, float] = {}
        for name, result in original_results.items():
            training = [item for item in result["sessions"] if item["sample_id"] not in validation_ids]
            train_scores[name] = technical_summary(training)["technical_score"]
        selected = max(sorted(train_scores), key=train_scores.get)
        selections[selected] += 1
        validation = [
            item for item in original_results[selected]["sessions"]
            if item["sample_id"] in validation_ids
        ]
        selected_sessions.extend(validation)
        fold_reports.append({
            "fold": fold,
            "sample_count": len(validation),
            "selected_config": selected,
            "training_scores": train_scores,
            "validation": technical_summary(validation),
        })

    full_results = {}
    for name in CONFIGS:
        original = technical_summary(original_results[name]["sessions"])
        paraphrased = technical_summary(robust_results[name]["sessions"])
        full_results[name] = {
            "original": original,
            "paraphrased": paraphrased,
            "worst_case_technical_score": min(
                original["technical_score"], paraphrased["technical_score"]
            ),
        }

    fold_scores = [report["validation"]["technical_score"] for report in fold_reports]
    return {
        "method": "grouped_stratified_cross_validation",
        "folds": args.folds,
        "seed": args.seed,
        "candidate_configs": {
            name: {"field_weights": list(config.field_weights)} for name, config in CONFIGS.items()
        },
        "selection_counts": dict(selections),
        "cross_validated": {
            **technical_summary(selected_sessions),
            "fold_score_mean": round(statistics.fmean(fold_scores), 6),
            "fold_score_stdev": round(statistics.pstdev(fold_scores), 6),
        },
        "full_set_robustness": full_results,
        "fold_reports": fold_reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--output", default="docs/bm25_validation.json")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.folds < 2:
        parser.error("folds must be at least two")
    result = run(args)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "fold_reports"}, indent=2))


if __name__ == "__main__":
    main()
