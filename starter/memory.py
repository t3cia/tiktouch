import re

KNOWN_COLORS = ["black", "white", "red", "blue", "green", "brown", "pink", "grey", "gray", "yellow"]
KNOWN_CATEGORIES = ["shoes", "sneakers", "jacket", "dress", "shirt", "necklace", "ring", "bag", "watch"]
KNOWN_MATERIALS = ["leather", "cotton", "wool", "silk", "polyester", "denim", "suede", "canvas", "fabric"]

NEGATION_WORDS = ["not", "no", "don't want", "dont want"]
REQUIREMENT_TRIGGERS = ["must", "must have", "must be", "needs to be", "has to be", "required", "non-negotiable"]


def is_negated(text: str, word: str) -> bool:
    """Checks if a word appears shortly after a negation phrase.
    Allows up to 3 filler words in between (e.g. "don't want it in black"),
    not just direct adjacency (e.g. "not black")."""
    pattern = (
        r"\b(" + "|".join(re.escape(n) for n in NEGATION_WORDS) + r")\b"
        r"(?:\s+\w+){0,3}\s+" + re.escape(word) + r"\b"
    )
    return bool(re.search(pattern, text))


def has_requirement_trigger(text: str) -> bool:
    """Checks if the message signals a hard must-have, not just a casual preference.
    Uses word boundaries so "must" doesn't accidentally match inside words
    like "mustard"."""
    return any(
        re.search(r"\b" + re.escape(trigger) + r"\b", text)
        for trigger in REQUIREMENT_TRIGGERS
    )


def get_clauses(message: str) -> list:
    """Splits a message into smaller parts on commas and 'and', for closer matching."""
    return [c.strip() for c in re.split(r",|\band\b", message) if c.strip()]


def find_earliest_match(text: str, options: list):
    """Finds which known option appears FIRST in the text (by position),
    not just first in our options list. Also reports whether that mention
    was negated (e.g. "not black").
    Returns (option_or_None, was_negated)."""
    matches = []
    for option in options:
        idx = text.find(option)
        if idx != -1:
            matches.append((idx, option))
    if not matches:
        return None, False
    matches.sort(key=lambda pair: pair[0])
    _, option = matches[0]
    return option, is_negated(text, option)


def new_profile():
    """Called once per session to start fresh."""
    return {
        "category": None,
        "material": None,
        "color": None,
        "size": None,
        "style": None,
        "brand": None,
        "budget": None,
        "feature": None,
        "use_case": None,
        "requirements": [],   # names of fields flagged as hard must-haves
        "history": []
    }


def extract_attributes(message: str):
    """Pulls out whatever attributes we can spot in one raw message/clause.
    Returns (found, negated):
      - found: dict of attributes to SET, e.g. {"color": "black"}
      - negated: set of attribute names that were explicitly REJECTED,
        e.g. {"color"} for "not black" -- these should be cleared, not set."""
    text = message.lower()
    found = {}
    negated = set()

    size_match = re.search(r"size\s*(\d+(\.\d+)?)", text)
    if size_match:
        found["size"] = size_match.group(1)

    budget_match = re.search(r"(?:\$|under|below|budget)\s*\$?(\d+)", text)
    if budget_match:
        found["budget"] = int(budget_match.group(1))

    color, color_negated = find_earliest_match(text, KNOWN_COLORS)
    if color:
        if color_negated:
            negated.add("color")
        else:
            found["color"] = color

    material, material_negated = find_earliest_match(text, KNOWN_MATERIALS)
    if material:
        if material_negated:
            negated.add("material")
        else:
            found["material"] = material

    category, category_negated = find_earliest_match(text, KNOWN_CATEGORIES)
    if category:
        if category_negated:
            negated.add("category")
        else:
            found["category"] = category

    return found, negated


def update_profile(profile: dict, message: str) -> dict:
    """Called every turn. Processes message clause by clause so requirement
    detection and negation both apply at the right precision."""
    for clause in get_clauses(message):
        clause_attrs, negated_keys = extract_attributes(clause)
        is_requirement = has_requirement_trigger(clause.lower())

        # Explicit rejections clear the field (e.g. "not black" wipes out
        # any earlier color we'd stored), instead of silently doing nothing.
        for key in negated_keys:
            profile[key] = None
            if key in profile["requirements"]:
                profile["requirements"].remove(key)

        for key, value in clause_attrs.items():
            profile[key] = value
            if is_requirement and key not in profile["requirements"]:
                profile["requirements"].append(key)

    profile["history"].append(message)
    return profile


if __name__ == "__main__":
    profile = new_profile()
    profile = update_profile(profile, "it must be leather, black color")
    print(profile)
    profile = update_profile(profile, "actually not black anymore")
    print(profile)