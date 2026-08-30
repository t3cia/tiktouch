from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
OVERRIDE_RE = re.compile(r"\b(?:actually|instead|ignore|changed? my mind|rather)\b", re.I)
NO_EVIDENCE_RE = re.compile(
    r"(?:not quite right|not suitable|ask me about|do not have|don't have|don’t have|"
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
BUDGET_RE = re.compile(r"(?:\$\s*\d|\b(?:budget|under|below|less than)\b)", re.I)
STOPWORDS = {
    "a", "about", "an", "and", "are", "as", "at", "be", "but", "by", "do",
    "for", "from", "here", "i", "in", "is", "it", "me", "my", "need", "of",
    "on", "or", "please", "some", "that", "the", "this", "to", "want", "what",
    "with", "would", "you", "your", "looking", "find", "help", "matters", "thing",
    "key", "requirement", "options", "exploring", "prefer", "preference", "prioritize",
}
QUESTION_ORDER = (
    "material", "color", "feature", "size", "style",
    "use_case", "brand", "budget", "other",
)


@dataclass(frozen=True)
class AgentConfig:
    """Small, explicit search configuration suitable for held-out selection."""

    field_weights: tuple[float, float, float, float, float, float] = (
        6.0, 6.0, 6.0, 3.0, 3.0, 1.0,
    )
    max_terms: int = 64


@dataclass
class SessionState:
    profile: dict
    category_message: str = ""
    evidence: list[str] = field(default_factory=list)
    asked_attributes: set[str] = field(default_factory=set)
    last_query: tuple[str, ...] = ()
    page: int = 0


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def _terms(text: str) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if len(token) > 1 and token.lower() not in STOPWORDS
    ]


class Agent:
    """Conservative stateful BM25 retriever with specific clarification questions."""

    def __init__(
        self,
        catalog_path: str | Path = "data/catalog.jsonl",
        config: AgentConfig | None = None,
    ) -> None:
        self.catalog_path = Path(catalog_path)
        self.config = config or AgentConfig()
        if len(self.config.field_weights) != 6:
            raise ValueError("field_weights must contain six values")
        self.connection = sqlite3.connect(":memory:")
        self._sessions: dict[str, SessionState] = {}
        self._build_index()

    def _build_index(self) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        batch: list[tuple[str, str, str, str, str, str, str]] = []
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                product = json.loads(line)
                batch.append(
                    (
                        str(product["parent_asin"]),
                        _text(product.get("title")),
                        _text(product.get("categories")),
                        _text(product.get("features")),
                        _text(product.get("details")),
                        _text(product.get("store")),
                        _text(product.get("description")),
                    )
                )
                if len(batch) >= 1000:
                    cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
                    batch.clear()
        if batch:
            cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
        self.connection.commit()

    def reset(self, session_id: str, user_profile: dict) -> None:
        self._sessions[session_id] = SessionState(profile=dict(user_profile or {}))

    @staticmethod
    def _opening_category(user_message: str) -> str:
        # The first sentence contains the category in the released protocol. Keeping
        # only that sentence prevents an obsolete override preference leaking back in.
        first_sentence = re.split(r"(?<=[.!?])\s+", user_message, maxsplit=1)[0]
        return re.split(r"[,;]\s*(?:but\b|i am\b|i'm\b)", first_sentence, maxsplit=1, flags=re.I)[0]

    @staticmethod
    def _update_state(state: SessionState, user_message: str, turn: int) -> None:
        if turn == 1:
            state.category_message = Agent._opening_category(user_message)
        if OVERRIDE_RE.search(user_message):
            state.evidence.clear()
            state.asked_attributes.clear()
        if not NO_EVIDENCE_RE.search(user_message):
            state.evidence.append(user_message)
            if MATERIAL_RE.search(user_message):
                state.asked_attributes.add("material")
            if COLOR_RE.search(user_message):
                state.asked_attributes.add("color")
            if BUDGET_RE.search(user_message):
                state.asked_attributes.add("budget")

    def _query_terms(self, state: SessionState) -> list[str]:
        evidence = " ".join([state.category_message, *state.evidence])
        return list(dict.fromkeys(_terms(evidence)))[: self.config.max_terms]

    @staticmethod
    def _next_question(state: SessionState) -> str:
        for attribute in QUESTION_ORDER:
            if attribute not in state.asked_attributes:
                state.asked_attributes.add(attribute)
                return attribute
        return "other"

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        state = self._sessions.get(session_id)
        if state is None:
            raise RuntimeError("reset must be called before respond")
        self._update_state(state, user_message, turn)
        unique_terms = self._query_terms(state)
        signature = tuple(unique_terms)
        if signature == state.last_query:
            state.page += 1
        else:
            state.last_query = signature
            state.page = 0

        recommendations: list[dict] = []
        expression = " OR ".join(f'"{term}"' for term in unique_terms)
        if expression:
            offset = state.page * top_k
            weights = ", ".join(str(value) for value in self.config.field_weights)
            rows = self.connection.execute(
                "SELECT parent_asin FROM products WHERE products MATCH ? "
                f"ORDER BY bm25(products, 0.0, {weights}) LIMIT ? OFFSET ?",
                (expression, top_k, offset),
            ).fetchall()
            recommendations = [{"parent_asin": str(row[0])} for row in rows]

        ask_attribute = self._next_question(state)
        prompts = {
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
        return {
            "message": prompts[ask_attribute],
            "ask_attribute": ask_attribute,
            "recommendations": recommendations,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
