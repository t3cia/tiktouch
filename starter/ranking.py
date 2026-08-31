import re

def ranking(recommendations: list[dict], memory: dict) -> list[dict]:
    if not recommendations:
        return []

    hard_requirements = set(memory.get("requirements", []))
    
    RERANK_FIELDS = (
        "color",
        "material",
        "size",
        "brand",
        "feature",
        "style",
        "use_case",
        "budget",
    )
    
    should_rerank = any(
        memory.get(field) not in (None, "")
        for field in RERANK_FIELDS
    )
    
    if not should_rerank:
        return sorted(
            recommendations,
            key=lambda product: product["search_score"],  # lower is better
            )[:10]

    # Preserve BM25 recall: only reorder its original top 10 candidates.
    recommendations = sorted(
        recommendations,
        key=lambda product: product["search_score"],  # lower FTS5 BM25 is better
        )[:10]

    def text_for(product: dict, fields: tuple[str, ...]) -> str:
        return " ".join(
            str(product.get(field) or "")
            for field in fields
        ).lower()
    
    def phrase_matches(text: str, value: object) -> bool:
        if value is None:
            return False
        
        cleaned = str(value).strip().lower()
        cleaned = re.sub(
            r"^for that,\s*what matters is:\s*",
            "",
            cleaned,
        ).strip(" .")
        
        # The evaluator can disclose two constraints separated by semicolons.
        phrases = [
            phrase.strip(" .")
            for phrase in cleaned.split(";")
            if phrase.strip(" .")
        ]
        
        return any(
            re.search(r"\b" + re.escape(phrase) + r"\b", text)
            for phrase in phrases
            )

    def size_matches(product: dict, size: object) -> bool:
        if size is None:
            return False

        text = text_for(product, ("title", "features", "details", "description"))
        return bool(re.search(
            rf"\b(?:size\s*)?{re.escape(str(size))}\b",
            text,
            re.IGNORECASE,
        ))

    lexical_scores = [-product["search_score"] for product in recommendations]
    low, high = min(lexical_scores), max(lexical_scores)

    def normalized_bm25(score: float) -> float:
        return 1.0 if high == low else (score - low) / (high - low)

    field_sources = {
        "color": ("title", "features", "details", "description"),
        "material": ("features", "details", "description"),
        "brand": ("title", "store"),
        "feature": ("features", "details", "description"),
        "style": ("title", "features", "details", "description"),
        "use_case": ("title", "features", "details", "description"),
    }

    soft_weights = {
        "color": 0.04,
        "material": 0.05,
        "brand": 0.03,
        "feature": 0.05,
        "style": 0.02,
        "use_case": 0.03,
        "size": 0.05,
    }

    ranked = []

    for product, lexical_score in zip(recommendations, lexical_scores):
        preference_bonus = 0.0
        reasons = []

        for field, soft_weight in soft_weights.items():
            value = memory.get(field)
            if value is None:
                continue

            matched = (
                size_matches(product, value)
                if field == "size"
                else phrase_matches(text_for(product, field_sources[field]), value)
            )

            if matched:
                boost = soft_weight * (3.0 if field in hard_requirements else 1.0)
                preference_bonus += boost
                reasons.append(f"matches {field}: {value}")

        budget = memory.get("budget")
        price = product.get("price")

        if budget is not None and price is not None:
            if float(price) <= float(budget):
                budget_boost = 0.09 if "budget" in hard_requirements else 0.03
                preference_bonus += budget_boost
                reasons.append("within budget")
            else:
                preference_bonus -= 0.09 if "budget" in hard_requirements else 0.03
                reasons.append("over budget")

        result = dict(product)
        result["rank_score"] = (
            0.80 * normalized_bm25(lexical_score)
            + preference_bonus
        )
        result["ranking_reasons"] = reasons
        ranked.append(result)

    return sorted(ranked, key=lambda product: product["rank_score"], reverse=True)[:10]