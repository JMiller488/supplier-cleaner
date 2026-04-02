"""
generate_synthetic.py
---------------------
Generates a labelled dataset of supplier name pairs for evaluation.

Produces two kinds of pairs:
  - Positive pairs (match=True):  same supplier, noisy variant
  - Negative pairs (match=False): two genuinely different suppliers

Output is a CSV file used as ground truth by run_evaluation.py.

Usage:
    python scripts/generate_synthetic.py
"""

import csv
import random

random.seed(42)

# ---------------------------------------------------------------------------
# Base supplier names — clean, preprocessed versions
# These represent the "true" supplier names in your system
# ---------------------------------------------------------------------------

BASE_SUPPLIERS = [
    "accenture",
    "deloitte",
    "bhp billiton",
    "commonwealth bank",
    "woolworths group",
    "wesfarmers",
    "macquarie group",
    "kpmg",
    "ernst young",
    "price waterhouse coopers",
    "booz allen hamilton",
    "mckinsey company",
    "boston consulting group",
    "johnson controls",
    "schneider electric",
]

# ---------------------------------------------------------------------------
# Noise functions
# Each takes a name string and returns a modified version of it
# ---------------------------------------------------------------------------

LEGAL_SUFFIXES = ["consulting", "group", "services", "solutions", "partners"]
GENERIC_WORDS = ["consulting", "solutions", "services", "group", "partners", "advisory"]


def add_suffix(name: str) -> str:
    """Append a random legal or generic suffix."""
    suffix = random.choice(LEGAL_SUFFIXES)
    return f"{name} {suffix}"


def drop_word(name: str) -> str:
    """Remove one word at random. No-ops if the name has fewer than 3 words."""
    words = name.split()
    if len(words) < 3:
        return name
    idx = random.randint(0, len(words) - 1)
    return " ".join(w for i, w in enumerate(words) if i != idx)


def swap_chars(name: str) -> str:
    """Transpose two adjacent characters at a random position.

    Simulates a realistic typo. Only operates on a single word
    within the name to keep the variant recognisable.
    """
    words = name.split()
    # Pick a word long enough to swap characters in
    candidates = [i for i, w in enumerate(words) if len(w) >= 3]
    if not candidates:
        return name
    word_idx = random.choice(candidates)
    word = words[word_idx]
    # Pick a position that isn't the last character
    char_idx = random.randint(0, len(word) - 2)
    swapped = word[:char_idx] + word[char_idx + 1] + word[char_idx] + word[char_idx + 2:]
    words[word_idx] = swapped
    return " ".join(words)


def add_generic_word(name: str) -> str:
    """Append a generic business descriptor word."""
    word = random.choice(GENERIC_WORDS)
    return f"{name} {word}"


# All noise functions collected in one list for easy iteration
NOISE_FUNCTIONS = [
    add_suffix,
    drop_word,
    swap_chars,
    add_generic_word,
]

# ---------------------------------------------------------------------------
# Pair generation
# ---------------------------------------------------------------------------


def generate_positive_pairs(suppliers: list[str]) -> list[tuple[str, str, bool]]:
    """Generate positive pairs by applying each noise function to each supplier.

    Each (original, noisy_variant) combination is a positive pair.
    We apply every noise function to every supplier name, giving us
    len(suppliers) * len(NOISE_FUNCTIONS) positive pairs in total.

    Args:
        suppliers: List of clean base supplier name strings.

    Returns:
        List of (name_a, name_b, match=True) tuples.
    """
    pairs = []
    for name in suppliers:
        for noise_fn in NOISE_FUNCTIONS:
            variant = noise_fn(name)
            # Only add the pair if the variant is actually different
            if variant != name:
                pairs.append((name, variant, True))
    return pairs


def generate_negative_pairs(
    suppliers: list[str], n: int
) -> list[tuple[str, str, bool]]:
    """Generate negative pairs by sampling random distinct supplier pairs.

    Shuffles the supplier list and pairs up adjacent entries.
    Repeats until we have at least n pairs.

    Args:
        suppliers: List of clean base supplier name strings.
        n: Number of negative pairs to generate.

    Returns:
        List of (name_a, name_b, match=False) tuples.
    """
    pairs = []
    while len(pairs) < n:
        shuffled = suppliers.copy()
        random.shuffle(shuffled)
        for i in range(0, len(shuffled) - 1, 2):
            if shuffled[i] != shuffled[i + 1]:
                pairs.append((shuffled[i], shuffled[i + 1], False))
            if len(pairs) >= n:
                break
    return pairs[:n]


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------


def write_csv(
    pairs: list[tuple[str, str, bool]],
    output_path: str,
) -> None:
    """Write labelled pairs to a CSV file.

    Args:
        pairs: List of (name_a, name_b, match) tuples.
        output_path: File path for the output CSV.
    """
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name_a", "name_b", "match"])
        writer.writerows(pairs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    positive_pairs = generate_positive_pairs(BASE_SUPPLIERS)
    # Generate the same number of negative pairs as positive for a balanced dataset
    negative_pairs = generate_negative_pairs(BASE_SUPPLIERS, n=len(positive_pairs))

    all_pairs = positive_pairs + negative_pairs
    random.shuffle(all_pairs)

    output_path = "data/synthetic_pairs.csv"
    import os
    os.makedirs("data", exist_ok=True)
    write_csv(all_pairs, output_path)

    n_pos = len(positive_pairs)
    n_neg = len(negative_pairs)
    print(f"Generated {len(all_pairs)} pairs ({n_pos} positive, {n_neg} negative)")
    print(f"Saved to {output_path}")