"""
audit_B1_tokens.py — quick check which B1 concept labels are single-token in
Qwen3-4B's tokenizer.

Each label is encoded with a leading space (because the scaffold ends with
'Answer:' and the answer token carries the leading space).

Usage: python scripts/audit_B1_tokens.py
"""
from transformers import AutoTokenizer

LABELS = [
    # concept: (label_a, label_b)
    ("grammar_number",   "singular", "plural"),
    ("lang_id",          "English",  "Russian"),
    ("sentiment",        "positive", "negative"),
    ("metal_recall",     "metal",    "nonmetal"),
    ("metal_recall_alt", "metal",    "metallic"),
    ("fission_fusion",   "fission",  "fusion"),
    ("alpha_beta",       "alpha",    "beta"),
    ("ab_fallback",      "A",        "B"),
]

def main():
    print(f"Loading Qwen3-4B tokenizer...")
    tok = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-4B", trust_remote_code=True
    )
    print(f"vocab_size = {tok.vocab_size}\n")
    print(f"{'concept':22s} {'label_a':14s} {'label_b':14s} {'a':5s} {'b':5s} {'verdict':12s}")
    print("-" * 80)
    for concept, la, lb in LABELS:
        ids_a = tok.encode(" " + la, add_special_tokens=False)
        ids_b = tok.encode(" " + lb, add_special_tokens=False)
        a_ok = len(ids_a) == 1
        b_ok = len(ids_b) == 1
        if a_ok and b_ok:
            verdict = "DIRECT OK"
        elif a_ok or b_ok:
            verdict = "MIXED -> A/B"
        else:
            verdict = "BOTH MULTI -> A/B"
        print(
            f"{concept:22s} "
            f"{repr(' ' + la):14s} {repr(' ' + lb):14s} "
            f"{len(ids_a):5d} {len(ids_b):5d} {verdict:12s}"
        )

    print()
    print("Notes:")
    print("- DIRECT OK    → use the label words as-is in the scaffold")
    print("- MIXED        → either drop the multi-token one, find a synonym, "
          "or fall back to A/B with order counterbalance")
    print("- BOTH MULTI   → use A/B fallback with order counterbalance")


if __name__ == "__main__":
    main()
