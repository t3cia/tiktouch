import re

KNOWN_COLORS = ["black", "white", "red", "blue", "green", "brown", "pink", "grey", "gray", "yellow"]
KNOWN_CATEGORIES = ["shoes", "sneakers", "jacket", "dress", "shirt", "necklace", "ring", "bag", "watch"]


def new_profile():
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
        "history": []
    }


def extract_attributes(message: str) -> dict:
    text = message.lower()
    found = {}

    size_match = re.search(r"size\s*(\d+(\.\d+)?)", text)
    if size_match:
        found["size"] = size_match.group(1)

    budget_match = re.search(r"\$?(\d+)", text)
    if budget_match and ("$" in text or "under" in text or "budget" in text or "below" in text):
        found["budget"] = int(budget_match.group(1))

    for color in KNOWN_COLORS:
        if color in text:
            found["color"] = color
            break

    for category in KNOWN_CATEGORIES:
        if category in text:
            found["category"] = category
            break

    return found


def update_profile(profile: dict, message: str) -> dict:
    new_attrs = extract_attributes(message)
    for key, value in new_attrs.items():
        profile[key] = value
    profile["history"].append(message)
    return profile


if __name__ == "__main__":
    profile = new_profile()
    profile = update_profile(profile, "I need running shoes")
    profile = update_profile(profile, "black, size 9, under $80")
    print(profile)
