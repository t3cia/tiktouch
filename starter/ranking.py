def ranking(recommendations: list[dict], memory: dict) -> list[dict]:
        if not recommendations:
            return []

        hard_requirements = memory.get("requirements")

        def product_text(product: dict) -> str:
            return " ".join(
                str(product.get(field) or "")
                for field in ("title", "categories", "details")
            ).lower()

        def matches(text: str, value: object) -> bool:
            return value is not None and str(value).strip().lower() in text

        def matches_size(text: str, size: object) -> bool:
            if size is None:
                return False
            return bool(
                re.search(
                    rf"\b(?:size\s*)?{re.escape(str(size))}\b",
                    text,
                    re.IGNORECASE,
                )
            )

    # Step 1: remove candidates that violate hard requirements.
        eligible = []
        for product in recommendations:
            text = product_text(product)
            is_eligible = True

            for field in hard_requirements:
                value = memory.get(field)

                if value is None:
                    continue  # A field cannot be enforced without a value.

                if field == "budget":                
                    price = product.get("price")
                    if price is None or float(price) > float(value):
                        is_eligible = False
                        break
                elif field == "size":
                    if not matches_size(text, value):
                        is_eligible = False
                        break
                elif not matches(text, value):
                    is_eligible = False
                    break

            if is_eligible:
                eligible.append(product)

        if not eligible:
            return []

        # FTS5 BM25: lower values are better, so invert and normalize.
        lexical_scores = [-product["search_score"] for product in eligible]
        low, high = min(lexical_scores), max(lexical_scores)

        def normalized_bm25(score: float) -> float:
            return 1.0 if high == low else (score - low) / (high - low)

        preference_weights = {
            "category": 1.0,
            "material": 1.0,
            "color": 0.9,
            "style": 0.6,
            "brand": 0.8,
            "feature": 1.0,
            "use_case": 0.8,
            "size": 0.8,
        }

        ranked = []
        for product, lexical_score in zip(eligible, lexical_scores):
            text = product_text(product)
            matched_weight = 0.0
            possible_weight = 0.0
            reasons = []

            # Only non-hard fields contribute to preference ranking.
            for field, weight in preference_weights.items():
                if field in hard_requirements:
                    continue

                value = memory.get(field)
                if value is None:
                    continue

                possible_weight += weight
                did_match = (
                    matches_size(text, value)
                    if field == "size"
                    else matches(text, value)
                )

                if did_match:
                    matched_weight += weight
                    reasons.append(f"matches {field}: {value}")

            preference_score = (
                matched_weight / possible_weight
                if possible_weight else 0.0
            )

            product = dict(product)
            product["rank_score"] = (
                0.60 * normalized_bm25(lexical_score)
                + 0.40 * preference_score
            )
            product["ranking_reasons"] = reasons
            ranked.append(product)

        return sorted(ranked, key=lambda product: product["rank_score"], reverse=True)[:10]
