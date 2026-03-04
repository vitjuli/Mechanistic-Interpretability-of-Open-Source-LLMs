#!/usr/bin/env python3
"""
Tokenization audit for multilingual_antonym behaviour.

Audits two things that are NOT covered by b_tokenize_audit_antonyms.py:
  1. EN synonym pairs (tiny/big/warm/cool/…) — needed for C1 operation-swap.
  2. FR antonym pairs beyond the 8 in antonym_operation (chaud/froid etc.)
     — needed for concept_index=1 (hot/cold) in C2/C3.
  3. FR synonym pairs (stretch goal; unlikely to pass, but checked).

Usage (no GPU needed — tokenizer only):
    python scripts/b_tokenize_audit_multilingual.py
    python scripts/b_tokenize_audit_multilingual.py \
        --output_jsonl data/prompts/multilingual_antonym_audit.jsonl

Output format (JSONL, one row per candidate):
    {"role": "synonym_en|antonym_fr|synonym_fr",
     "concept_index": int,
     "language": "en"|"fr",
     "word": str,           # source adjective
     "target": str,         # synonym/antonym to predict
     "word_single_token": bool,
     "target_single_token": bool,
     "both_single_token": bool,
     "word_token_ids": [...],
     "target_token_ids": [...]}
"""

import argparse
import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Concept table.
# Indexed by concept_index (0-8).  These are the 9 cross-lang concepts we
# want to use in multilingual_antonym.
# EN antonym pairs are already audited (all 25 EN pairs pass).
# ---------------------------------------------------------------------------
_CONCEPTS = [
    # (concept_idx, en_word, en_antonym, fr_word, fr_antonym)
    (0, "small",  "large",   "petit",   "grand"),
    (1, "hot",    "cold",    "chaud",   "froid"),    # FR needs audit (not in existing 8)
    (2, "fast",   "slow",    "vite",    "lent"),
    (3, "new",    "old",     "nouveau", "vieux"),
    (4, "empty",  "full",    "vide",    "plein"),
    (5, "high",   "low",     "haut",    "bas"),
    (6, "long",   "short",   "long",    "court"),
    (7, "clean",  "dirty",   "propre",  "sale"),
    (8, "easy",   "hard",    "facile",  "difficile"),
]

# ---------------------------------------------------------------------------
# EN synonym candidates.
# For each concept word, propose (synonym_of_word, synonym_of_antonym).
# Both must be single-token; if either fails, the concept is dropped from
# the EN synonym prompts.
# ---------------------------------------------------------------------------
_EN_SYNONYM_CANDIDATES = {
    # concept_idx → (syn_of_word, syn_of_antonym)
    # Confident:
    0: ("tiny",     "big"),       # small→tiny, large→big
    1: ("warm",     "cool"),      # hot→warm,   cold→cool
    2: ("quick",    "gradual"),   # fast→quick,  slow→gradual
    3: ("fresh",    "aged"),      # new→fresh,   old→aged
    4: ("bare",     "packed"),    # empty→bare,  full→packed
    5: ("tall",     "short"),     # high→tall,   low→short  (note: "short" also in antonym of "long")
    6: ("lengthy",  "brief"),     # long→lengthy, short→brief
    7: ("neat",     "messy"),     # clean→neat,  dirty→messy
    8: ("simple",   "tough"),     # easy→simple, hard→tough
}

# FR synonym candidates (stretch goal).
_FR_SYNONYM_CANDIDATES = {
    # concept_idx → (fr_syn_of_word, fr_syn_of_antonym)
    0: ("minuscule", "énorme"),   # petit→minuscule, grand→énorme  (likely multi-token)
    1: ("chaud",     "frais"),    # FR hot synonyms — chaud IS the word, frais for froid
    7: ("propre",    "crasseux"), # propre is the word; try a stretch
    8: ("aisé",      "ardu"),     # facile→aisé, difficile→ardu
}


def check_single_token(tokenizer, word: str, with_space: bool = True) -> tuple:
    """Return (is_single_token, token_ids) for word, tested with leading space."""
    s = (" " + word) if with_space else word
    ids = tokenizer.encode(s, add_special_tokens=False)
    # Also test without leading space as fallback
    if len(ids) != 1 and with_space:
        ids_no_sp = tokenizer.encode(word, add_special_tokens=False)
        if len(ids_no_sp) == 1:
            return True, ids_no_sp
    return len(ids) == 1, ids


def audit_en_synonyms(tokenizer, concepts, candidates) -> list:
    rows = []
    print("\n" + "=" * 60)
    print("EN SYNONYM AUDIT")
    print(f"{'concept':>3}  {'word':<10} {'syn_word':<12} {'ant':<10} {'syn_ant':<12} {'syn_w':<6} {'syn_a':<6} valid")
    print("-" * 70)
    for cidx, en_word, en_ant, _fw, _fa in concepts:
        if cidx not in candidates:
            continue
        syn_w, syn_a = candidates[cidx]
        sw_ok, sw_ids = check_single_token(tokenizer, syn_w)
        sa_ok, sa_ids = check_single_token(tokenizer, syn_a)
        both = sw_ok and sa_ok
        flag = "✓" if both else "✗"
        print(f"  {cidx:>3}  {en_word:<10} {syn_w:<12} {en_ant:<10} {syn_a:<12} {str(sw_ids):<6} {str(sa_ids):<6} {flag}")
        rows.append({
            "role": "synonym_en",
            "concept_index": cidx,
            "language": "en",
            "word": en_word,
            "antonym": en_ant,
            "synonym_word": syn_w,
            "synonym_antonym": syn_a,
            "syn_word_single_token": sw_ok,
            "syn_ant_single_token": sa_ok,
            "both_single_token": both,
            "syn_word_token_ids": sw_ids,
            "syn_ant_token_ids": sa_ids,
        })
    n_valid = sum(r["both_single_token"] for r in rows)
    print(f"\n  EN synonyms valid: {n_valid}/{len(rows)}")
    return rows


def audit_fr_antonyms(tokenizer, concepts) -> list:
    """Audit FR antonym pairs that are NOT already in the existing 8 cross-lang pairs.
    Specifically checks concept_index=1 (chaud/froid) and others."""
    # Already audited (from antonym_operation): 0,2,3,4,5,6,7,8 (using original 8)
    # Need to audit: concept_index=1 (chaud/froid)
    print("\n" + "=" * 60)
    print("FR ANTONYM AUDIT (new pairs not in existing _ANTONYM_CROSS_LANG)")
    print(f"{'concept':>3}  {'fr_word':<12} {'fr_ant':<14} {'w_ids':<12} {'a_ids':<12} valid")
    print("-" * 68)
    rows = []
    for cidx, en_word, en_ant, fr_word, fr_ant in concepts:
        w_ok, w_ids = check_single_token(tokenizer, fr_word)
        a_ok, a_ids = check_single_token(tokenizer, fr_ant)
        both = w_ok and a_ok
        flag = "✓" if both else "✗"
        print(f"  {cidx:>3}  {fr_word:<12} {fr_ant:<14} {str(w_ids):<12} {str(a_ids):<12} {flag}")
        rows.append({
            "role": "antonym_fr",
            "concept_index": cidx,
            "language": "fr",
            "word": fr_word,
            "antonym": fr_ant,
            "word_single_token": w_ok,
            "antonym_single_token": a_ok,
            "both_single_token": both,
            "word_token_ids": w_ids,
            "antonym_token_ids": a_ids,
        })
    n_valid = sum(r["both_single_token"] for r in rows)
    print(f"\n  FR antonym pairs valid: {n_valid}/{len(rows)}")
    return rows


def audit_fr_synonyms(tokenizer, concepts, candidates) -> list:
    print("\n" + "=" * 60)
    print("FR SYNONYM AUDIT (stretch goal)")
    rows = []
    for cidx, en_word, en_ant, fr_word, fr_ant in concepts:
        if cidx not in candidates:
            continue
        syn_w, syn_a = candidates[cidx]
        sw_ok, sw_ids = check_single_token(tokenizer, syn_w)
        sa_ok, sa_ids = check_single_token(tokenizer, syn_a)
        both = sw_ok and sa_ok
        flag = "✓" if both else "✗"
        print(f"  {cidx:>3}  {fr_word:<12} {syn_w:<14} {str(sw_ids):<12} {str(sa_ids):<12} {flag}")
        rows.append({
            "role": "synonym_fr",
            "concept_index": cidx,
            "language": "fr",
            "word": fr_word,
            "antonym": fr_ant,
            "synonym_word": syn_w,
            "synonym_antonym": syn_a,
            "syn_word_single_token": sw_ok,
            "syn_ant_single_token": sa_ok,
            "both_single_token": both,
        })
    n_valid = sum(r["both_single_token"] for r in rows)
    print(f"\n  FR synonyms valid: {n_valid}/{len(rows)}")
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Tokenization audit for multilingual_antonym synonyms and FR antonym pairs"
    )
    parser.add_argument(
        "--model_name",
        default="Qwen/Qwen3-4B-Instruct",
        help="HuggingFace model (tokenizer only, no GPU needed)",
    )
    parser.add_argument(
        "--output_jsonl",
        default=None,
        help="Path to write audit results JSONL",
    )
    args = parser.parse_args()

    from transformers import AutoTokenizer
    print(f"Loading tokenizer: {args.model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    print(f"Vocab size: {tokenizer.vocab_size}")

    en_syn_rows = audit_en_synonyms(tokenizer, _CONCEPTS, _EN_SYNONYM_CANDIDATES)
    fr_ant_rows = audit_fr_antonyms(tokenizer, _CONCEPTS)
    fr_syn_rows = audit_fr_synonyms(tokenizer, _CONCEPTS, _FR_SYNONYM_CANDIDATES)

    all_rows = en_syn_rows + fr_ant_rows + fr_syn_rows

    print("\n" + "=" * 60)
    print("SUMMARY FOR multilingual_antonym GENERATOR")
    print("=" * 60)

    valid_en_syn = [r for r in en_syn_rows if r["both_single_token"]]
    valid_fr_ant = [r for r in fr_ant_rows if r["both_single_token"]]
    valid_fr_syn = [r for r in fr_syn_rows if r["both_single_token"]]

    print(f"\nEN synonym pairs passing audit: {len(valid_en_syn)}/{len(en_syn_rows)}")
    for r in valid_en_syn:
        print(f"  concept {r['concept_index']:>2} ({r['word']:>6} → {r['synonym_word']:<8},"
              f" {r['antonym']:>10} → {r['synonym_antonym']:<8})")

    print(f"\nFR antonym pairs passing audit: {len(valid_fr_ant)}/{len(fr_ant_rows)}")
    for r in valid_fr_ant:
        print(f"  concept {r['concept_index']:>2} ({r['word']:>8} → {r['antonym']:<12})")

    print(f"\nFR synonym pairs passing audit: {len(valid_fr_syn)}/{len(fr_syn_rows)}")

    print("\nCopy-paste for _ML_VALID_SYNONYMS in 01_generate_prompts.py:")
    print("    # (concept_index, syn_of_word, syn_of_antonym)")
    for r in valid_en_syn:
        print(f"    ({r['concept_index']}, \"{r['synonym_word']}\", \"{r['synonym_antonym']}\"),  "
              f"# {r['word']} / {r['antonym']}")

    print("\nFR concepts with valid antonym pairs (for cross-lang C3):")
    for r in valid_fr_ant:
        print(f"  concept {r['concept_index']:>2}: {r['word']} / {r['antonym']}")

    if args.output_jsonl:
        out = Path(args.output_jsonl)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            for row in all_rows:
                json.dump(row, f)
                f.write("\n")
        print(f"\nWrote {len(all_rows)} rows to {out}")


if __name__ == "__main__":
    main()
