# TikTouch

TikTouch is a conversational product-search agent that combines semantic retrieval, keyword search, structured memory, and preference-aware reranking to refine recommendations over multiple turns.

## The Problem

Online shopping search works well when users know exactly what they want, but real shopping conversations are often vague. A customer might ask for something like “comfortable shoes for travelling” without specifying a brand, material, style, or budget. Traditional keyword search fails to understand the context and user intent behind queries, and may return products that are technically related but not actually useful when faced with incomplete preferences.

TikTouch addresses this problem by acting as a conversational e-commerce search agent. The system keeps track of their preferences across multiple turns, asks useful clarification questions when information is missing, and continuously refines its product recommendations.

The goal is to identify and rank the product that best matches the user's underlying intent within a maximum of 10 conversational turns.


## Key Features

- Hybrid semantic and keyword retrieval
- Intent-aware retrieval selection
- Multi-turn structured memory
- Preference override and negation handling
- Targeted clarification questions
- BM25 candidate retrieval
- Preference-aware reranking
- Budget and hard-requirement handling

## How It Works

```text
User message
    ↓
Intent detection
    ↓
Conversational memory update
    ↓
Dense / TF-IDF / BM25 retrieval
    ↓
Preference-aware reranking
    ↓
Top product recommendations
```

Broad exploratory requests use semantic retrieval, while specific buying requests benefit from BM25 keyword retrieval and structured reranking.
The memory component stores attributes such as category, material, color, size, style, brand, budget, feature, and use case. It also tracks hard requirements and handles preference changes across turns.
The reranker combines BM25 relevance with field-specific preference matches while preserving the strongest retrieval candidates.

## How We Built It

TikTouch uses a **hybrid conversational product-retrieval pipeline** that combines lexical search, semantic similarity, conversational memory, and preference-aware reranking.

For lexical retrieval, we use **BM25 through SQLite FTS5** to identify products containing important keywords and attributes from the user's request. We also use **TF-IDF with cosine similarity** to compare user preferences with product metadata. For broader exploratory requests, TikTouch supports **dense semantic retrieval** using the `sentence-transformers/all-MiniLM-L6-v2` model. This allows semantically related products to match even when the query and product description use different words.

The system selects its retrieval approach according to the user's intent. Vague browsing requests benefit from semantic retrieval, while more specific buying requests use BM25 to retrieve a focused candidate pool.

TikTouch maintains a **structured conversational memory** containing attributes such as category, material, color, size, style, brand, budget, feature, and use case. It also records hard requirements and conversation history. When users change their minds—for example, “I don't want black anymore”—the outdated preference is removed without clearing unrelated context. Answers to clarification questions are associated with the relevant memory field so they can influence later retrieval and ranking.

After retrieval, a **preference-aware reranker** combines normalised BM25 relevance with bonuses for products that match the user's current preferences. Hard requirements receive stronger bonuses, known over-budget products are penalised, and products with missing prices are retained to avoid unnecessarily reducing recall. The reranker only reorders BM25's strongest candidates, allowing it to refine result quality without displacing already-relevant products.

The agent prioritises clarification questions that reveal useful product features early. As more information is collected, TikTouch updates its memory, retrieval query, and ranking signals. This allows the system to balance two goals: **asking useful questions and recommending relevant products as quickly as possible**.

For further implementation details, refer to the section below.

## Technical Documentation

Detailed implementation documentation
- [BM25 Validation Protocol](docs/bm25_validation.md)
- [Emulated Dense-retrieval Data](docs/dense_training.md)
- [Conversational memory](docs/memory.md)
- [Preference-aware reranking](docs/ranking.md)

## Results

We evaluate TikTouch using Hit Rate@10, Mean Reciprocal Rank (MRR), Mean Turns to Completion (MTTC), and the challenge's combined technical score.
On the 200-session public evaluation set, our lightweight intent-aware agent achieved:
- Hit Rate@10: 89.5%
- MRR: 0.533
- Mean Turns to Completion: 3.94
- Efficiency: 0.707
- Technical Score: 0.749
- Reported Token Usage: 0

The lightweight agent runs entirely with local retrieval and machine-learning components and does not rely on token-based LLM API calls during evaluation. As a result, it reports zero token usage while still achieving a substantial improvement over the provided weak BM25 baseline, which achieved a 12.5% Hit Rate@10.

## Set Up and Installation

1. Clone the repository
2. Create a virtual environment
3. Install dependencies
```
pip install -r requirements.txt
```

### Reproducing Our Results

Run the local evaluator from the repository root:
```
python3 -m evaluator.local_evaluator
```

## Limitations 

Our biggest limitation was the re-ranking algorithm. Given the limited time, we were unable to implement a complex trained learning-to which could reliably recover relevant products ranked outside the initial result set. With more time, we would intend to train or validate ranking weights using a separate held-out dataset and explore a cross-encoder or learning-to-rank model for second-stage ranking to further improve our solution.

## Team Member Contributions

| Name | Contribution |
| -- | -- |
| Angela | Conversational memory and preference override handling |
| Phoebe | Fine tuning of BM25, dense retrieval, missing attribute follow ups, video |
| Tricia | Preference-aware reranking and evaluation, integration |
| Yao Teck | User intent detection, fine tuning of BM25, dense retrieval, missing attribute follow ups |

All members: Documentation and demo
