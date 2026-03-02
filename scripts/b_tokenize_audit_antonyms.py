#!/usr/bin/env python3
"""
Tokenization audit for antonym_operation behaviour.

Checks which (word, antonym) pairs yield exactly one token for both elements
under the Qwen3 tokenizer, in each candidate language.

Usage (no GPU needed):
    python scripts/b_tokenize_audit_antonyms.py
    python scripts/b_tokenize_audit_antonyms.py --model_name Qwen/Qwen3-4B-Instruct
    python scripts/b_tokenize_audit_antonyms.py --output_jsonl data/prompts/antonym_operation_audit.jsonl

Output:
    - Console table: (language, word, antonym, both_single_token)
    - JSONL with only the valid pairs (if --output_jsonl is given)
"""

import argparse
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Candidate word pairs.
# Format: {language_code: [(word, antonym), ...]}
# These cover simple adjectives likely to be single-token in BPE tokenizers.
# ---------------------------------------------------------------------------
CANDIDATES = {
    "en": [
        ("small", "large"),
        ("hot", "cold"),
        ("fast", "slow"),
        ("dark", "light"),
        ("hard", "soft"),
        ("rich", "poor"),
        ("new", "old"),
        ("good", "bad"),
        ("strong", "weak"),
        ("happy", "sad"),
        ("open", "closed"),
        ("empty", "full"),
        ("early", "late"),
        ("high", "low"),
        ("long", "short"),
        ("wide", "narrow"),
        ("clean", "dirty"),
        ("loud", "quiet"),
        ("cheap", "expensive"),
        ("easy", "hard"),
        ("bright", "dark"),
        ("deep", "shallow"),
        ("heavy", "light"),
        ("sharp", "dull"),
        ("thick", "thin"),
    ],
    "fr": [
        ("petit", "grand"),
        ("chaud", "froid"),
        ("vite", "lent"),
        ("sombre", "clair"),
        ("dur", "doux"),
        ("riche", "pauvre"),
        ("nouveau", "vieux"),
        ("bon", "mauvais"),
        ("fort", "faible"),
        ("heureux", "triste"),
        ("ouvert", "fermé"),
        ("vide", "plein"),
        ("tôt", "tard"),
        ("haut", "bas"),
        ("long", "court"),
        ("large", "étroit"),
        ("propre", "sale"),
        ("fort", "calme"),
        ("léger", "lourd"),
        ("facile", "difficile"),
    ],
    "de": [
        ("klein", "groß"),
        ("heiß", "kalt"),
        ("schnell", "langsam"),
        ("dunkel", "hell"),
        ("hart", "weich"),
        ("reich", "arm"),
        ("neu", "alt"),
        ("gut", "schlecht"),
        ("stark", "schwach"),
        ("glücklich", "traurig"),
        ("offen", "geschlossen"),
        ("leer", "voll"),
        ("früh", "spät"),
        ("hoch", "niedrig"),
        ("lang", "kurz"),
        ("breit", "schmal"),
        ("sauber", "schmutzig"),
        ("laut", "leise"),
        ("leicht", "schwer"),
        ("einfach", "schwierig"),
    ],
    "es": [
        ("pequeño", "grande"),
        ("caliente", "frío"),
        ("rápido", "lento"),
        ("oscuro", "claro"),
        ("duro", "suave"),
        ("rico", "pobre"),
        ("nuevo", "viejo"),
        ("bueno", "malo"),
        ("fuerte", "débil"),
        ("feliz", "triste"),
        ("abierto", "cerrado"),
        ("vacío", "lleno"),
        ("temprano", "tarde"),
        ("alto", "bajo"),
        ("largo", "corto"),
        ("ancho", "estrecho"),
        ("limpio", "sucio"),
        ("fuerte", "suave"),
        ("pesado", "ligero"),
        ("fácil", "difícil"),
    ],
    "zh": [
        ("小", "大"),
        ("热", "冷"),
        ("快", "慢"),
        ("暗", "亮"),
        ("硬", "软"),
        ("富", "穷"),
        ("新", "旧"),
        ("好", "坏"),
        ("强", "弱"),
        ("快乐", "悲伤"),
        ("开", "关"),
        ("空", "满"),
        ("早", "晚"),
        ("高", "低"),
        ("长", "短"),
        ("宽", "窄"),
        ("干净", "脏"),
        ("响", "静"),
        ("轻", "重"),
        ("容易", "难"),
    ],
}

# Prompt templates per language.
# Tokens are what comes RIGHT AFTER the template — the model predicts the antonym.
TEMPLATES = {
    "en": 'The opposite of "{word}" is "',
    "fr": 'Le contraire de "{word}" est "',
    "de": 'Das Gegenteil von "{word}" ist "',
    "es": 'Lo contrario de "{word}" es "',
    "zh": '"{word}"的反义词是"',
}


def check_single_token(tokenizer, token_str: str, add_space: bool = True) -> bool:
    """Return True iff token_str encodes to exactly 1 BPE token."""
    s = (" " + token_str) if add_space else token_str
    ids = tokenizer.encode(s, add_special_tokens=False)
    return len(ids) == 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3-4B-Instruct",
        help="HuggingFace model name (tokenizer only, no GPU needed)",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=["en", "fr"],
        help="Languages to audit (default: en fr)",
    )
    parser.add_argument(
        "--output_jsonl",
        type=str,
        default=None,
        help="If set, write valid pairs as JSONL prompt file",
    )
    parser.add_argument(
        "--min_languages",
        type=int,
        default=1,
        help="Minimum number of languages a word pair must pass in (cross-language analysis needs >= 2)",
    )
    args = parser.parse_args()

    from transformers import AutoTokenizer
    print(f"Loading tokenizer: {args.model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    print(f"Vocab size: {tokenizer.vocab_size}\n")

    valid_rows = []
    all_rows = []

    for lang in args.languages:
        if lang not in CANDIDATES:
            print(f"WARNING: no candidates for language {lang!r}")
            continue
        tmpl = TEMPLATES[lang]
        print(f"{'='*60}")
        print(f"Language: {lang}")
        print(f"Template: {tmpl}")
        print(f"{'word':<15} {'antonym':<15} {'word_tok':<12} {'ant_tok':<12} valid")
        print(f"{'-'*65}")
        for word, antonym in CANDIDATES[lang]:
            # Check both with leading space (BPE-friendly) and without
            # Use leading-space check first; if fails, try without space
            word_ok_space = check_single_token(tokenizer, word, add_space=True)
            ant_ok_space = check_single_token(tokenizer, antonym, add_space=True)
            word_ok_nospace = check_single_token(tokenizer, word, add_space=False)
            ant_ok_nospace = check_single_token(tokenizer, antonym, add_space=False)

            # Use whichever works; prefer with-space
            word_tok = " " + word if word_ok_space else word
            ant_tok = " " + antonym if ant_ok_space else antonym
            word_ok = word_ok_space or word_ok_nospace
            ant_ok = ant_ok_space or ant_ok_nospace
            both_ok = word_ok and ant_ok
            flag = "✓" if both_ok else "✗"

            word_ids = tokenizer.encode(word_tok, add_special_tokens=False)
            ant_ids = tokenizer.encode(ant_tok, add_special_tokens=False)
            print(f"  {word:<13} {antonym:<13} {str(word_ids):<12} {str(ant_ids):<12} {flag}")

            row = {
                "language": lang,
                "word": word,
                "antonym": antonym,
                "word_token": word_tok,
                "antonym_token": ant_tok,
                "word_token_ids": word_ids,
                "antonym_token_ids": ant_ids,
                "word_single_token": word_ok,
                "antonym_single_token": ant_ok,
                "both_single_token": both_ok,
                "template": tmpl,
                "prompt": tmpl.replace("{word}", word),
                "correct_answer": ant_tok.strip(),
                "incorrect_answer": word_tok.strip(),  # original word = "incorrect" (no antonym)
            }
            all_rows.append(row)
            if both_ok:
                valid_rows.append(row)

        lang_valid = [r for r in valid_rows if r["language"] == lang]
        print(f"\n  Valid pairs for {lang}: {len(lang_valid)}/{len(CANDIDATES[lang])}")

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    by_lang = {}
    for r in valid_rows:
        by_lang.setdefault(r["language"], []).append(r)
    for lang, rows in by_lang.items():
        print(f"  {lang}: {len(rows)} valid pairs")

    # Cross-language pairs: same (word concept) valid in multiple languages
    # NOTE: for cross-language analysis, we want the SAME ENGLISH word to have
    # valid translations across all requested languages.
    # We define cross-lang pairs by the English word.
    if len(args.languages) >= 2:
        en_valid_words = {r["word"] for r in valid_rows if r["language"] == "en"}
        # Find pairs where the EN word exists in the candidates and is valid in all languages
        # This requires alignment in CANDIDATES (same index = same concept)
        en_pairs = {r["word"]: r["antonym"] for r in valid_rows if r["language"] == "en"}
        cross_lang_valid = []
        # Build a concept map
        concept_map = {}
        for lang in args.languages:
            en_cands = CANDIDATES.get("en", [])
            lang_cands = CANDIDATES.get(lang, [])
            if lang == "en":
                continue
            for i, (en_w, en_a) in enumerate(en_cands):
                if i < len(lang_cands):
                    concept_map.setdefault(en_w, {})[lang] = lang_cands[i]

        print(f"\nCross-language valid pairs (valid in EN + all of {[l for l in args.languages if l!='en']}):")
        cross_count = 0
        for en_word, en_ant in en_pairs.items():
            langs_valid = ["en"]
            for lang in args.languages:
                if lang == "en":
                    continue
                lang_pair = concept_map.get(en_word, {}).get(lang)
                if lang_pair is None:
                    break
                lang_word, lang_ant = lang_pair[0] if isinstance(lang_pair, tuple) else (None, None)
                # Check in valid_rows
                lang_valid_match = any(
                    r["language"] == lang and r["antonym"] == concept_map[en_word].get(lang, (None,None))[1 if isinstance(concept_map[en_word].get(lang), tuple) else 0]
                    for r in valid_rows
                ) if isinstance(concept_map.get(en_word, {}).get(lang), tuple) else False
                if lang_valid_match:
                    langs_valid.append(lang)
            if len(langs_valid) >= args.min_languages:
                cross_count += 1
        print(f"  Pairs valid in >= {args.min_languages} languages: {cross_count}")

    if args.output_jsonl:
        out_path = Path(args.output_jsonl)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            for r in valid_rows:
                json.dump(r, f)
                f.write("\n")
        print(f"\nWrote {len(valid_rows)} valid prompt pairs to {out_path}")
        print("These can be used directly with scripts/02_run_baseline.py --prompts_file ...")

    print(f"\nTotal valid pairs: {len(valid_rows)}/{len(all_rows)}")
    return valid_rows


if __name__ == "__main__":
    main()
