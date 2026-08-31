# Memory Component

Tracks a customer's accumulated preferences across a conversation, so the agent
"remembers" what was said in earlier turns instead of only reacting to the
latest message.

## What it does

Every turn, `update_profile()` reads the customer's message and updates a
`profile` dictionary with anything relevant it finds — colors, materials,
categories, sizes, budgets — while keeping everything from previous turns
intact unless it's explicitly contradicted or replaced.

## Profile shape

```python
{
    "category": None,      # e.g. "shoes", "bag"
    "material": None,      # e.g. "leather", "cotton"
    "color": None,         # e.g. "black"
    "size": None,
    "style": None,
    "brand": None,
    "budget": None,        # max price, as an int
    "feature": None,       # free-text, only filled when the agent explicitly asked about it
    "use_case": None,      # free-text, same as above
    "requirements": [],    # names of fields the customer flagged as a hard must-have
    "history": []          # raw list of every message seen this session
}
```

## Key behaviors

- **Requirement detection**: if a message contains a trigger phrase like "must be" or "needs to be" (e.g. "it must be leather"), that specific attribute is added to `requirements` — so the ranker can treat it as non-negotiable rather than a soft preference.
- **Clause splitting**: messages are split on commas and "and" before extraction, so a message like "it must be leather, black color" correctly flags only `material` as a requirement, not `color`.
- **Negation**: phrases like "not black" or "don't want leather" prevent that value from being set, and will actively clear a previously stored value if the customer changes their mind.
- **Free-text fields** (`feature`, `style`, `use_case`, `brand`): these can't be pattern-matched from a fixed word list, so they're only filled in when the agent specifically asked about that attribute last turn (`update_profile(profile, message, last_asked="feature")`) — the raw reply is then trusted as the answer.

## Known limitations

- Category/color/material extraction picks whichever known word appears earliest in the sentence — it's a simple keyword scan, not true NLP understanding.
- Free-text fields are only captured when directly asked about; if a customer volunteers a feature unprompted, it may be missed.
- Negation detection requires the negation word to appear within a few words of the target word.

## Usage

```python
from memory import new_profile, update_profile

profile = new_profile()
profile = update_profile(profile, "it must be leather, black color")
profile = update_profile(profile, "actually not black anymore", last_asked=None)
```

## Testing

Run it directly to see example output:
​```
python3 starter/memory.py
​```

## Dependencies

None — uses only Python's built-in `re` module. No network calls, no external packages, works fully offline.