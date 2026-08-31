from __future__ import annotations
from starter.ranking import ranking
from starter.memory import new_profile, update_profile

import json
import math
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = {
    "a", "about", "an", "and", "are", "as", "at", "be", "but", "by", "do",
    "for", "from", "here", "i", "in", "is", "it", "me", "my", "need", "of",
    "on", "or", "please", "some", "that", "the", "this", "to", "want", "what",
    "with", "would", "you", "your", "looking", "find", "help", "matters", "thing",
    "key", "requirement", "options", "exploring", "prefer", "preference", "prioritize",
}
QUESTION_ORDER = (
    "material", "color", "feature", "size", "style",
    "use_case", "brand", "budget",
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
    last_query: tuple[str, ...] = ()
    page: int = 0
    last_asked: str | None = None


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def _numeric_price(value: object) -> float | None:
    """Return a SQLite-safe numeric price; treat unavailable labels as NULL."""
    if value in (None, "") or isinstance(value, bool):
        return None
    try:
        price = float(value)
    except (TypeError, ValueError):
        return None
    return price if math.isfinite(price) and price >= 0 else None


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
        cursor.execute(
            "CREATE TABLE product_metadata("
            "parent_asin TEXT PRIMARY KEY, price REAL)"
        )
        cursor.execute(
            "CREATE INDEX product_metadata_price_idx "
            "ON product_metadata(price) WHERE price IS NOT NULL"
        )
        batch: list[tuple[str, str, str, str, str, str, str]] = []
        metadata_batch: list[tuple[str, float | None]] = []
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                product = json.loads(line)
                parent_asin = str(product["parent_asin"])
                batch.append(
                    (
                        parent_asin,
                        _text(product.get("title")),
                        _text(product.get("categories")),
                        _text(product.get("features")),
                        _text(product.get("details")),
                        _text(product.get("store")),
                        _text(product.get("description")),
                    )
                )
                metadata_batch.append((parent_asin, _numeric_price(product.get("price"))))
                if len(batch) >= 1000:
                    cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
                    cursor.executemany(
                        "INSERT INTO product_metadata(parent_asin, price) VALUES (?, ?)",
                        metadata_batch,
                    )
                    batch.clear()
                    metadata_batch.clear()
        if batch:
            cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
            cursor.executemany(
                "INSERT INTO product_metadata(parent_asin, price) VALUES (?, ?)",
                metadata_batch,
            )
        self.connection.commit()

    def reset(self, session_id: str, user_profile: dict) -> None:
        self._sessions[session_id] = SessionState(profile=new_profile())

    def _query_terms(self, state: SessionState) -> list[str]:
        structured_values = [
            str(value)
            for key, value in state.profile.items()
            if key not in {"history", "requirements", "budget"}
            and value is not None
        ]
        # History improves BM25 recall. Ranking still trusts structured memory,
        # so stale history cannot become a hard requirement.
        retrieval_text = " ".join([
            *structured_values,
            *state.profile["history"],
        ])
        return list(dict.fromkeys(
            _terms(retrieval_text)
            ))[:self.config.max_terms]


    @staticmethod
    def _next_question(state: SessionState) -> str:
        for attribute in QUESTION_ORDER:
            if state.profile.get(attribute) is None:
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
        update_profile(state.profile, user_message, last_asked=state.last_asked)
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
            SEARCH_POOL_SIZE = 50
            rows = self.connection.execute(
                #modified this line by adding title, categories, details to the returned products
                f"""SELECT 
                    p.parent_asin, 
                    p.title,
                    p.categories,
                    p.features,
                    p.details,
                    p.store,
                    p.description,
                    bm25(products, 0.0, {weights}) AS search_score,
                    m.price
        """
                " FROM products p "
                " JOIN product_metadata m on p.parent_asin = m.parent_asin"
                " WHERE products MATCH ? "
                f"ORDER BY bm25(products, 0.0, {weights}) LIMIT ? OFFSET ?",
                (expression, SEARCH_POOL_SIZE, offset),
            ).fetchall()
            recommendations = [
                {
                    "parent_asin": str(row[0]),
                    "title": row[1],
                    "categories": row[2],
                    "features": row[3],
                    "details": row[4],
                    "store": row[5],
                    "description": row[6],
                    "search_score": row[7],
                    "price": row[8],
                } 
                for row in rows
            ]
        
        recommendations = ranking(recommendations, state.profile)
        #recommendations = recommendations[:top_k]

        ask_attribute = self._next_question(state)
        state.last_asked = ask_attribute
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
