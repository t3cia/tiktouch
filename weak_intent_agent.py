from __future__ import annotations

"""Train and run a lightweight intent-aware hybrid shopping agent.

The agent deliberately uses small, inspectable models:

* a TF-IDF + logistic-loss SGD conversation router;
* weighted SQLite FTS5/BM25 retrieval;
* TF-IDF cosine retrieval over the frozen catalog;
* a logistic-regression confidence calibrator trained on synthetic query prefixes.

Typical use:

    python weak_intent_agent.py train
    python weak_intent_agent.py evaluate --output weak_results.json
"""

import argparse
import hashlib
import json
import math
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Sequence


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
SPACE_RE = re.compile(r"\s+")
OVERRIDE_RE = re.compile(r"\b(?:actually|instead|ignore|changed? my mind|rather)\b", re.I)
NO_PREFERENCE_RE = re.compile(
    r"(?:not quite right|not suitable|ask me about|do not have|don't have|don.t have|"
    r"no (?:additional |extra )?(?:preference|requirement)|use your judgment|open to suggestions)",
    re.I,
)
MATERIAL_RE = re.compile(
    r"\b(?:cotton|polyester|nylon|leather|wool|spandex|silk|rayon|linen|fleece|fabric)\b",
    re.I,
)
COLOR_RE = re.compile(
    r"\b(?:black|white|blue|red|pink|green|brown|gr[ae]y|purple|yellow|orange|beige|navy)\b",
    re.I,
)
BUDGET_RE = re.compile(r"(?:\$\s*\d|\b(?:budget|under|below|less than|around \$)\b)", re.I)
SIZE_RE = re.compile(r"\b(?:size|sizing|fit|wide|narrow|small|medium|large|xl|xxl)\b", re.I)
STYLE_RE = re.compile(r"\b(?:style|casual|formal|classic|modern|vintage|sleeve|neckline)\b", re.I)
USE_CASE_RE = re.compile(r"\b(?:hiking|running|gym|winter|outdoor|work|wedding|travel)\b", re.I)

STOPWORDS = {
    "a", "about", "an", "and", "are", "as", "at", "be", "but", "by", "do",
    "for", "from", "here", "i", "in", "is", "it", "me", "my", "need", "of",
    "on", "or", "please", "some", "that", "the", "this", "to", "want", "what",
    "with", "would", "you", "your", "looking", "find", "help", "matters", "thing",
    "key", "requirement", "options", "exploring", "prefer", "preference", "prioritize",
}
QUESTION_ORDER = (
    "material", "color", "feature", "size", "style", "use_case", "brand", "budget", "other"
)
ATTRIBUTE_PRIORS = {
    "material": 1.00,
    "color": 0.92,
    "feature": 0.95,
    "size": 0.68,
    "style": 0.76,
    "use_case": 0.78,
    "brand": 0.52,
    "budget": 0.70,
}
PROFILE_ATTRIBUTE_TAGS = {
    "material": {"material", "comfort", "warmth"},
    "color": {"color", "style"},
    "feature": {"comfort", "durability", "weather"},
    "size": {"fit", "size"},
    "style": {"style"},
    "use_case": {"weather", "use case"},
    "brand": {"brand"},
    "budget": {"budget", "price"},
}
STYLE_TERMS = (
    "casual", "formal", "classic", "modern", "vintage", "slim fit", "regular fit",
    "relaxed fit", "crew neck", "v-neck", "long sleeve", "short sleeve",
)
USE_CASE_TERMS = (
    "hiking", "running", "gym", "winter", "outdoor", "work", "wedding", "travel",
    "cycling", "walking", "sports", "swimming", "yoga",
)
FEATURE_TERMS = (
    "waterproof", "water resistant", "lightweight", "breathable", "moisture wicking",
    "quick dry", "stretch", "compression", "insulated", "warm", "comfortable",
    "durable", "uv protection", "non slip", "arch support", "padded", "adjustable",
)
PROMPTS = {
    "material": "Do you have a material preference?",
    "color": "Is there a color you prefer?",
    "feature": "Which product feature matters most?",
    "size": "Do you have a size or fit requirement?",
    "style": "What style are you looking for?",
    "use_case": "What will you mainly use it for?",
    "brand": "Do you prefer a particular brand?",
    "budget": "What budget should I stay within?",
    "other": "Is there another requirement I should prioritize?",
}
SEARCH_FIELDS = ("title", "categories", "features", "details", "store", "description")
ARTIFACT_VERSION = 2


def _clean(value: object, limit: int = 1800) -> str:
    if value in (None, "", []):
        return ""
    if isinstance(value, dict):
        value = "; ".join(
            f"{key}: {item}" for key, item in value.items() if item not in (None, "", [])
        )
    elif isinstance(value, list):
        value = "; ".join(str(item) for item in value if item not in (None, ""))
    return SPACE_RE.sub(" ", str(value)).strip(" -;,.\t\n")[:limit].rstrip()


def _index_text(value: object) -> str:
    """Match the proven starter's lossless FTS field serialization."""
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def product_text(product: dict) -> str:
    sections = [
        ("title", _clean(product.get("title"), 400)),
        ("category", _clean(product.get("categories"), 300)),
        ("brand", _clean(product.get("store"), 120)),
        ("features", _clean(product.get("features"), 700)),
        ("details", _clean(product.get("details"), 450)),
        ("description", _clean(product.get("description"), 600)),
    ]
    if product.get("price") not in (None, ""):
        sections.append(("price", f"${product['price']}"))
    return " | ".join(f"{name}: {value}" for name, value in sections if value)[:1800]


def _matched_terms(text: str, terms: Sequence[str]) -> list[str]:
    lowered = text.lower()
    return [term for term in terms if re.search(rf"\b{re.escape(term)}\b", lowered)]


def _budget_bucket(value: object) -> str | None:
    try:
        price = float(value)
    except (TypeError, ValueError):
        return None
    if price < 25:
        return "under $25"
    if price < 50:
        return "$25–$50"
    if price < 100:
        return "$50–$100"
    return "$100 or more"


def extract_product_attributes(product: dict) -> dict[str, list[str]]:
    """Extract compact, user-questionable attributes from released metadata."""
    text = product_text(product).lower()
    attributes: dict[str, list[str]] = {}
    materials = sorted({match.group(0).lower() for match in MATERIAL_RE.finditer(text)})
    colors = sorted({match.group(0).lower().replace("grey", "gray") for match in COLOR_RE.finditer(text)})
    if materials:
        attributes["material"] = materials
    if colors:
        attributes["color"] = colors
    styles = _matched_terms(text, STYLE_TERMS)
    use_cases = _matched_terms(text, USE_CASE_TERMS)
    features = _matched_terms(text, FEATURE_TERMS)
    if styles:
        attributes["style"] = styles
    if use_cases:
        attributes["use_case"] = use_cases
    if features:
        attributes["feature"] = features
    details = product.get("details") or {}
    if isinstance(details, dict):
        size_text = " ".join(
            str(value) for key, value in details.items()
            if any(token in str(key).lower() for token in ("size", "width", "fit"))
        )
        size_values = [value.strip().lower() for value in re.split(r"[,;/]", size_text) if value.strip()]
        if size_values:
            attributes["size"] = size_values[:4]
    brand = _clean(product.get("store"), 80).lower()
    if brand:
        attributes["brand"] = [brand]
    budget = _budget_bucket(product.get("price"))
    if budget:
        attributes["budget"] = [budget]
    return attributes


def classify_constraint(value: str) -> str:
    """Map a disclosed constraint to one of the evaluator's question slots."""
    lowered = value.lower()
    if BUDGET_RE.search(lowered):
        return "budget"
    if MATERIAL_RE.search(lowered):
        return "material"
    if COLOR_RE.search(lowered):
        return "color"
    if SIZE_RE.search(lowered):
        return "size"
    if STYLE_RE.search(lowered):
        return "style"
    if USE_CASE_RE.search(lowered):
        return "use_case"
    return "feature"


def load_jsonl(path: str | Path) -> Iterator[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _terms(text: str) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if len(token) > 1 and token.lower() not in STOPWORDS
    ]


def _stable_fraction(key: str, seed: int = 2026) -> float:
    digest = hashlib.blake2b(f"{seed}\0{key}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") / 2**64


def _top_indexes(scores, count: int):
    import numpy as np

    count = min(count, len(scores))
    if count <= 0:
        return np.empty(0, dtype=np.int64)
    indexes = np.argpartition(scores, -count)[-count:]
    return indexes[np.argsort(-scores[indexes], kind="stable")]


def confidence_features(scores, turn: int, query: str) -> list[float]:
    import numpy as np

    indexes = _top_indexes(scores, 10)
    top = np.asarray(scores[indexes], dtype=np.float64)
    if len(top) == 0:
        top = np.zeros(10, dtype=np.float64)
    top1 = float(top[0])
    top2 = float(top[1]) if len(top) > 1 else 0.0
    mean10 = float(top.mean())
    return [
        top1,
        top1 - top2,
        top1 - mean10,
        mean10,
        min(max(turn, 1), 10) / 10.0,
        min(len(_terms(query)), 100) / 100.0,
    ]


@dataclass(frozen=True)
class AgentConfig:
    field_weights: tuple[float, float, float, float, float, float] = (
        6.0, 6.0, 6.0, 3.0, 3.0, 1.0
    )
    max_terms: int = 64
    candidate_count: int = 160
    question_candidate_count: int = 100
    minimum_attribute_coverage: float = 0.08
    rrf_constant: float = 60.0
    # Ablation on the public set showed that cosine helps as a gentle high-
    # confidence tie-breaker, but hurts when allowed to influence uncertain cases.
    cosine_weight_low: float = 0.0
    cosine_weight_medium: float = 0.0
    cosine_weight_high: float = 0.05
    medium_confidence: float = 0.45
    high_confidence: float = 0.80
    include_debug: bool = False


@dataclass
class SessionState:
    profile: dict
    category_message: str = ""
    evidence: list[str] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)
    known_attributes: set[str] = field(default_factory=lambda: {"category"})
    asked_attributes: set[str] = field(default_factory=set)
    no_preference_attributes: set[str] = field(default_factory=set)
    last_asked_attribute: str | None = None
    last_query: tuple[str, ...] = ()
    page: int = 0
    mode: str = "uncertain"
    override_active: bool = False


class WeakArtifacts:
    def __init__(self, artifact_dir: str | Path) -> None:
        import joblib
        import numpy as np
        from scipy import sparse

        root = Path(artifact_dir)
        metadata_path = root / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Missing trained artifacts in {root}. Run: python weak_intent_agent.py train"
            )
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if self.metadata.get("artifact_version") != ARTIFACT_VERSION:
            raise ValueError("Weak-model artifacts are incompatible; retrain them")
        self.vectorizer = joblib.load(root / "catalog_vectorizer.joblib")
        self.router = joblib.load(root / "router.joblib")
        self.confidence_model = joblib.load(root / "confidence.joblib")
        self.matrix = sparse.load_npz(root / "catalog_matrix.npz").astype(np.float32)
        self.ids = [
            str(json.loads(line)["parent_asin"])
            for line in (root / "catalog_ids.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.attributes = {
            str(row["parent_asin"]): {
                str(attribute): [str(value) for value in values]
                for attribute, values in (row.get("attributes") or {}).items()
            }
            for row in load_jsonl(root / "catalog_attributes.jsonl")
        }
        if self.matrix.shape[0] != len(self.ids):
            raise ValueError("Catalog artifact ID and matrix counts differ")
        if len(self.attributes) != len(self.ids):
            raise ValueError("Catalog artifact attribute and matrix counts differ")


class Agent:
    """Stateful hybrid agent matching the competition Agent API."""

    def __init__(
        self,
        catalog_path: str | Path = "data/catalog.jsonl",
        artifact_dir: str | Path = "checkpoints/weak_intent",
        config: AgentConfig | None = None,
    ) -> None:
        self.catalog_path = Path(catalog_path)
        self.config = config or AgentConfig()
        self.artifacts = WeakArtifacts(artifact_dir)
        self.connection = sqlite3.connect(":memory:")
        self._sessions: dict[str, SessionState] = {}
        self._build_bm25()

    def _build_bm25(self) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        batch: list[tuple[str, str, str, str, str, str, str]] = []
        for product in load_jsonl(self.catalog_path):
            values = [str(product["parent_asin"])]
            values.extend(_index_text(product.get(field)) for field in SEARCH_FIELDS)
            batch.append(tuple(values))
            if len(batch) >= 1000:
                cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
                batch.clear()
        if batch:
            cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
        self.connection.commit()

    def reset(self, session_id: str, user_profile: dict) -> None:
        self._sessions[session_id] = SessionState(profile=dict(user_profile or {}))

    @staticmethod
    def _opening_category(message: str) -> str:
        first = re.split(r"(?<=[.!?])\s+", message, maxsplit=1)[0]
        return re.split(r"[,;]\s*(?:but\b|i am\b|i'm\b)", first, maxsplit=1, flags=re.I)[0]

    def _route(self, state: SessionState, message: str) -> tuple[str, dict[str, float]]:
        if OVERRIDE_RE.search(message):
            return "intent_override", {"intent_override": 1.0}
        if NO_PREFERENCE_RE.search(message):
            return "boundary", {"boundary": 1.0}
        text = " ".join([*state.messages[-2:], message])
        probabilities = self.artifacts.router.predict_proba([text])[0]
        classes = self.artifacts.router.classes_
        result = {str(label): float(value) for label, value in zip(classes, probabilities)}
        label = max(result, key=result.get)
        if result[label] < 0.55 and state.mode != "uncertain":
            label = state.mode
        return label, result

    @staticmethod
    def _mark_attributes(state: SessionState, message: str) -> None:
        # Keep implicit detection conservative.  Broad words such as "fit" or
        # "style" occur inside catalog-derived feature sentences and caused the
        # agent to skip useful follow-up questions during ablation testing.
        checks = (
            ("material", MATERIAL_RE), ("color", COLOR_RE), ("budget", BUDGET_RE),
        )
        for attribute, pattern in checks:
            if pattern.search(message):
                state.known_attributes.add(attribute)
        disclosed = re.search(
            r"(?:key requirement is|what matters is|what i need is)\s*:\s*(.+)",
            message,
            re.I,
        )
        if disclosed:
            for value in disclosed.group(1).split(";"):
                state.known_attributes.add(classify_constraint(value))

    def _update_state(self, state: SessionState, message: str, turn: int) -> dict[str, float]:
        if turn == 1:
            state.category_message = self._opening_category(message)
        mode, probabilities = self._route(state, message)
        # Clearing constraints is destructive, so require explicit correction
        # language.  The probabilistic route remains available for reporting and
        # non-destructive policy choices, but cannot erase valid evidence alone.
        if OVERRIDE_RE.search(message):
            state.evidence.clear()
            state.known_attributes = {"category"}
            state.asked_attributes.clear()
            state.no_preference_attributes.clear()
            state.last_asked_attribute = None
            state.page = 0
            state.override_active = True
        state.mode = mode
        state.messages.append(message)
        if NO_PREFERENCE_RE.search(message):
            if state.last_asked_attribute:
                state.no_preference_attributes.add(state.last_asked_attribute)
            state.last_asked_attribute = None
        else:
            if state.last_asked_attribute and not OVERRIDE_RE.search(message):
                state.known_attributes.add(state.last_asked_attribute)
            state.evidence.append(message)
            self._mark_attributes(state, message)
            state.last_asked_attribute = None
        return probabilities

    def _query(self, state: SessionState) -> tuple[str, list[str]]:
        evidence = " ".join([state.category_message, *state.evidence])
        unique_terms = list(dict.fromkeys(_terms(evidence)))[: self.config.max_terms]
        return evidence, unique_terms

    def _bm25(self, terms: Sequence[str], count: int) -> list[str]:
        expression = " OR ".join(f'"{term}"' for term in terms)
        if not expression:
            return []
        weights = ", ".join(str(value) for value in self.config.field_weights)
        rows = self.connection.execute(
            "SELECT parent_asin FROM products WHERE products MATCH ? "
            f"ORDER BY bm25(products, 0.0, {weights}) LIMIT ?",
            (expression, count),
        ).fetchall()
        return [str(row[0]) for row in rows]

    def _cosine(self, query: str):
        vector = self.artifacts.vectorizer.transform([query])
        return (self.artifacts.matrix @ vector.T).toarray().ravel()

    def _hybrid_rank(
        self, bm25_ids: Sequence[str], cosine_scores, cosine_weight: float
    ) -> list[tuple[str, float]]:
        cosine_indexes = _top_indexes(cosine_scores, self.config.candidate_count)
        cosine_ids = [self.artifacts.ids[int(index)] for index in cosine_indexes]
        bm25_rank = {item: rank for rank, item in enumerate(bm25_ids, 1)}
        cosine_rank = {item: rank for rank, item in enumerate(cosine_ids, 1)}
        candidates = set(bm25_rank) | set(cosine_rank)
        bm25_weight = 1.0 - cosine_weight
        constant = self.config.rrf_constant
        ranked = []
        for item in candidates:
            score = 0.0
            if item in bm25_rank:
                score += bm25_weight / (constant + bm25_rank[item])
            if item in cosine_rank:
                score += cosine_weight / (constant + cosine_rank[item])
            ranked.append((item, score))
        ranked.sort(key=lambda pair: (-pair[1], bm25_rank.get(pair[0], math.inf), pair[0]))
        return ranked

    def _attribute_utilities(
        self,
        state: SessionState,
        ranked: Sequence[tuple[str, float]],
    ) -> tuple[dict[str, float], dict[str, list[str]]]:
        """Score unanswered slots by coverage and diversity in the candidate pool."""
        candidates = ranked[: self.config.question_candidate_count]
        total_weight = sum(1.0 / math.log2(rank + 2) for rank in range(len(candidates))) or 1.0
        profile_tags = {
            str(value).lower() for value in state.profile.get("preference_tags") or []
        }
        utilities: dict[str, float] = {}
        options: dict[str, list[str]] = {}
        for attribute in QUESTION_ORDER:
            if attribute == "other":
                continue
            if (
                attribute in state.known_attributes
                or attribute in state.asked_attributes
                or attribute in state.no_preference_attributes
            ):
                continue
            covered_weight = 0.0
            value_weights: dict[str, float] = {}
            for rank, (parent_asin, _) in enumerate(candidates):
                values = self.artifacts.attributes.get(parent_asin, {}).get(attribute, [])
                values = list(dict.fromkeys(value for value in values if value))
                if not values:
                    continue
                weight = 1.0 / math.log2(rank + 2)
                covered_weight += weight
                share = weight / len(values)
                for value in values:
                    value_weights[value] = value_weights.get(value, 0.0) + share
            coverage = covered_weight / total_weight
            if coverage < self.config.minimum_attribute_coverage or not value_weights:
                continue
            value_total = sum(value_weights.values())
            probabilities = [weight / value_total for weight in value_weights.values()]
            if len(probabilities) > 1:
                entropy = -sum(value * math.log2(value) for value in probabilities)
                diversity = entropy / math.log2(len(probabilities))
            else:
                diversity = 0.0
            profile_boost = 0.08 if profile_tags & PROFILE_ATTRIBUTE_TAGS[attribute] else 0.0
            utilities[attribute] = (
                ATTRIBUTE_PRIORS[attribute] * (0.58 * coverage + 0.42 * diversity)
                + profile_boost
            )
            options[attribute] = [
                value for value, _ in sorted(
                    value_weights.items(), key=lambda pair: (-pair[1], pair[0])
                )[:3]
            ]
        return utilities, options

    @staticmethod
    def _question_message(attribute: str, options: Sequence[str]) -> str:
        clean_options = [value for value in options if 1 < len(value) <= 28]
        if attribute in {"material", "color", "brand", "style"} and len(clean_options) >= 2:
            first, second = clean_options[:2]
            label = attribute.replace("_", " ")
            return f"Do you prefer {first}, {second}, or another {label}?"
        if attribute == "feature" and len(clean_options) >= 2:
            return f"Would {clean_options[0]}, {clean_options[1]}, or another feature matter most?"
        if attribute == "use_case" and len(clean_options) >= 2:
            return f"Will you mainly use it for {clean_options[0]}, {clean_options[1]}, or something else?"
        if attribute == "budget" and len(clean_options) >= 2:
            return f"Should I focus on options {clean_options[0]}, {clean_options[1]}, or another budget?"
        return PROMPTS[attribute]

    def _next_question(
        self,
        state: SessionState,
        ranked: Sequence[tuple[str, float]],
    ) -> tuple[str, str, dict[str, float], list[str]]:
        utilities, options = self._attribute_utilities(state, ranked)
        if state.override_active:
            # Corrections have few remaining turns.  Use the validated stable
            # order after an override instead of experimenting with broad slots.
            conservative_known = state.known_attributes & {"material", "color", "budget"}
            attribute = next(
                (
                    value for value in QUESTION_ORDER
                    if value not in conservative_known
                    and value not in state.asked_attributes
                    and value not in state.no_preference_attributes
                ),
                "other",
            )
        elif utilities:
            order = {attribute: index for index, attribute in enumerate(QUESTION_ORDER)}
            attribute = max(
                utilities,
                key=lambda value: (utilities[value], -order[value]),
            )
        else:
            attribute = next(
                (
                    value for value in QUESTION_ORDER
                    if value not in state.known_attributes
                    and value not in state.asked_attributes
                    and value not in state.no_preference_attributes
                ),
                "other",
            )
        selected_options = options.get(attribute, [])
        state.asked_attributes.add(attribute)
        state.last_asked_attribute = attribute
        return (
            attribute,
            self._question_message(attribute, selected_options),
            utilities,
            selected_options,
        )

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        state = self._sessions.get(session_id)
        if state is None:
            raise RuntimeError("reset must be called before respond")
        route_probabilities = self._update_state(state, user_message, turn)
        query, terms = self._query(state)
        signature = tuple(terms)
        if signature == state.last_query:
            state.page += 1
        else:
            state.last_query = signature
            state.page = 0

        candidate_count = max(self.config.candidate_count, (state.page + 1) * top_k)
        bm25_ids = self._bm25(terms, candidate_count)
        cosine_scores = self._cosine(query)
        features = confidence_features(cosine_scores, turn, query)
        confidence = float(self.artifacts.confidence_model.predict_proba([features])[0, 1])
        if confidence >= self.config.high_confidence:
            cosine_weight = self.config.cosine_weight_high
        elif confidence >= self.config.medium_confidence:
            cosine_weight = self.config.cosine_weight_medium
        else:
            cosine_weight = self.config.cosine_weight_low
        ranked = self._hybrid_rank(bm25_ids, cosine_scores, cosine_weight)
        start = state.page * top_k
        selected = ranked[start : start + top_k]

        ask_attribute, question_message, question_utilities, question_options = self._next_question(
            state, ranked
        )
        response = {
            "message": question_message,
            "ask_attribute": ask_attribute,
            "recommendations": [
                {"parent_asin": parent_asin, "score": round(score, 8)}
                for parent_asin, score in selected
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
        if self.config.include_debug:
            response["debug"] = {
                "mode": state.mode,
                "route_probabilities": route_probabilities,
                "retrieval_confidence": round(confidence, 6),
                "cosine_weight": cosine_weight,
                "known_attributes": sorted(state.known_attributes),
                "no_preference_attributes": sorted(state.no_preference_attributes),
                "question_utilities": {
                    key: round(value, 6) for key, value in question_utilities.items()
                },
                "question_options": question_options,
            }
        return response


def build_catalog_artifacts(catalog_path: Path, artifact_dir: Path) -> dict:
    import joblib
    import numpy as np
    from scipy import sparse
    from sklearn.feature_extraction.text import TfidfVectorizer

    products = list(load_jsonl(catalog_path))
    if not products:
        raise ValueError(f"Catalog is empty: {catalog_path}")
    texts = [product_text(product) for product in products]
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.995,
        max_features=60_000,
        sublinear_tf=True,
        norm="l2",
        dtype=np.float32,
    )
    matrix = vectorizer.fit_transform(texts).astype(np.float32)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, artifact_dir / "catalog_vectorizer.joblib", compress=3)
    sparse.save_npz(artifact_dir / "catalog_matrix.npz", matrix, compressed=True)
    with (artifact_dir / "catalog_ids.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for product in products:
            handle.write(json.dumps({"parent_asin": str(product["parent_asin"])}) + "\n")
    with (artifact_dir / "catalog_attributes.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        for product in products:
            handle.write(json.dumps({
                "parent_asin": str(product["parent_asin"]),
                "attributes": extract_product_attributes(product),
            }, ensure_ascii=False, separators=(",", ":")) + "\n")
    return {
        "catalog_count": len(products),
        "vocabulary_size": len(vectorizer.vocabulary_),
        "matrix_shape": list(matrix.shape),
        "matrix_nnz": int(matrix.nnz),
        "attribute_fields": list(ATTRIBUTE_PRIORS),
    }


def router_examples(training_path: Path) -> tuple[list[str], list[str], list[str]]:
    texts: list[str] = []
    labels: list[str] = []
    groups: list[str] = []
    for row in load_jsonl(training_path):
        turns = [str(value) for value in row.get("turns") or [] if str(value).strip()]
        if not turns:
            continue
        scenario = str(row.get("scenario_type") or "browsing")
        group = str(row.get("positive_id") or row.get("example_id"))
        for index in range(len(turns)):
            if index == 0 and scenario in {"intent_override", "boundary"}:
                label = "browsing"
            elif scenario == "intent_override" and index > 0:
                label = "intent_override"
            elif scenario == "boundary" and index > 0:
                label = "boundary"
            else:
                label = scenario
            texts.append(" ".join(turns[: index + 1]))
            labels.append(label)
            groups.append(group)
    return texts, labels, groups


def train_router(training_path: Path, artifact_dir: Path) -> dict:
    import joblib
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import SGDClassifier
    from sklearn.metrics import accuracy_score, log_loss
    from sklearn.pipeline import Pipeline

    texts, labels, groups = router_examples(training_path)
    validation = np.asarray([_stable_fraction(group) < 0.20 for group in groups])
    train_indexes = np.flatnonzero(~validation)
    validation_indexes = np.flatnonzero(validation)

    def new_model() -> Pipeline:
        return Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2), min_df=2, max_features=35_000,
                sublinear_tf=True, strip_accents="unicode",
            )),
            ("classifier", SGDClassifier(
                loss="log_loss", alpha=2e-5, max_iter=1000, tol=1e-4,
                class_weight="balanced", random_state=2026,
            )),
        ])

    model = new_model()
    model.fit([texts[i] for i in train_indexes], [labels[i] for i in train_indexes])
    predicted = model.predict([texts[i] for i in validation_indexes])
    probabilities = model.predict_proba([texts[i] for i in validation_indexes])
    true_labels = [labels[i] for i in validation_indexes]
    metrics = {
        "examples": len(texts),
        "train_examples": int(len(train_indexes)),
        "validation_examples": int(len(validation_indexes)),
        "validation_accuracy": round(float(accuracy_score(true_labels, predicted)), 6),
        "validation_log_loss": round(float(log_loss(true_labels, probabilities, labels=model.classes_)), 6),
        "classes": [str(value) for value in model.classes_],
    }
    model.fit(texts, labels)
    joblib.dump(model, artifact_dir / "router.joblib", compress=3)
    return metrics


def train_confidence(
    training_path: Path,
    artifact_dir: Path,
    sample_count: int,
) -> dict:
    import joblib
    import numpy as np
    from scipy import sparse
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

    vectorizer = joblib.load(artifact_dir / "catalog_vectorizer.joblib")
    matrix = sparse.load_npz(artifact_dir / "catalog_matrix.npz").astype(np.float32)
    ids = [
        str(json.loads(line)["parent_asin"])
        for line in (artifact_dir / "catalog_ids.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    id_to_index = {value: index for index, value in enumerate(ids)}
    candidates: list[tuple[float, str, int, str, str]] = []
    for row in load_jsonl(training_path):
        target = str(row.get("positive_id", ""))
        if target not in id_to_index:
            continue
        turns = [str(value) for value in row.get("turns") or []]
        for turn in range(1, len(turns) + 1):
            query = " ".join(turns[:turn])
            key = f"{row.get('example_id')}:{turn}"
            candidates.append((_stable_fraction(key, 9917), query, turn, target, key))
    candidates.sort(key=lambda item: item[0])
    selected = candidates[: min(sample_count, len(candidates))]

    features: list[list[float]] = []
    labels: list[int] = []
    validation_flags: list[bool] = []
    batch_size = 64
    for start in range(0, len(selected), batch_size):
        batch = selected[start : start + batch_size]
        vectors = vectorizer.transform([item[1] for item in batch])
        similarities = (vectors @ matrix.T).toarray()
        for item, scores in zip(batch, similarities):
            _, query, turn, target, key = item
            top10 = _top_indexes(scores, 10)
            features.append(confidence_features(scores, turn, query))
            labels.append(int(id_to_index[target] in set(int(value) for value in top10)))
            validation_flags.append(_stable_fraction(key, 31337) < 0.20)

    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    validation = np.asarray(validation_flags)
    if len(np.unique(y)) < 2:
        raise ValueError("Confidence training produced only one class; increase --confidence-samples")
    model = LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=2026, C=0.5
    )
    model.fit(x[~validation], y[~validation])
    probabilities = model.predict_proba(x[validation])[:, 1]
    metrics = {
        "examples": int(len(y)),
        "positive_rate": round(float(y.mean()), 6),
        "validation_examples": int(validation.sum()),
        "validation_brier": round(float(brier_score_loss(y[validation], probabilities)), 6),
        "validation_log_loss": round(float(log_loss(y[validation], probabilities)), 6),
        "validation_roc_auc": round(float(roc_auc_score(y[validation], probabilities)), 6),
        "feature_names": [
            "top1", "top1_minus_top2", "top1_minus_mean10", "mean10",
            "turn_fraction", "query_term_fraction",
        ],
    }
    model.fit(x, y)
    joblib.dump(model, artifact_dir / "confidence.joblib", compress=3)
    return metrics


def train(args: argparse.Namespace) -> dict:
    artifact_dir = Path(args.artifact_dir)
    catalog_metrics = build_catalog_artifacts(Path(args.catalog), artifact_dir)
    router_metrics = train_router(Path(args.training_data), artifact_dir)
    confidence_metrics = train_confidence(
        Path(args.training_data), artifact_dir, args.confidence_samples
    )
    result = {
        "artifact_version": ARTIFACT_VERSION,
        "catalog": str(args.catalog),
        "training_data": str(args.training_data),
        "catalog_model": catalog_metrics,
        "router": router_metrics,
        "confidence": confidence_metrics,
    }
    (artifact_dir / "metadata.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def evaluate_agent(args: argparse.Namespace) -> dict:
    from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl as load_samples

    samples = load_samples(args.dataset)
    catalog_ids, categories, products = catalog_index(args.catalog)
    agent = Agent(args.catalog, args.artifact_dir)
    result = evaluate(agent, samples, catalog_ids, categories, products)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return {key: value for key, value in result.items() if key != "sessions"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train all weak-model artifacts")
    train_parser.add_argument("--catalog", default="data/catalog.jsonl")
    train_parser.add_argument("--training-data", default="data/emulated_training.jsonl")
    train_parser.add_argument("--artifact-dir", default="checkpoints/weak_intent")
    train_parser.add_argument("--confidence-samples", type=int, default=8000)

    evaluate_parser = subparsers.add_parser("evaluate", help="Run the official local evaluator")
    evaluate_parser.add_argument("--catalog", default="data/catalog.jsonl")
    evaluate_parser.add_argument("--dataset", default="data/public_set.jsonl")
    evaluate_parser.add_argument("--artifact-dir", default="checkpoints/weak_intent")
    evaluate_parser.add_argument("--output", default="weak_results.json")
    args = parser.parse_args()

    if args.command == "train":
        if args.confidence_samples < 100:
            parser.error("--confidence-samples must be at least 100")
        result = train(args)
    else:
        result = evaluate_agent(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
