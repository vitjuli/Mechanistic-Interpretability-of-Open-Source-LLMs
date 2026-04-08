"""
Tokenization audit for physics behaviour answer tokens.
Run before writing any new prompts (D2 decision rule).

Results (Qwen3-4B tokenizer, 2026-04-08):
  ' alpha'       [8287]          — OK
  ' beta'        [13440]         — OK
  ' classical'   [28824]         — OK
  ' relativistic' [58325, 4532]  — MULTI-TOKEN ⚠  → B4 uses ' valid'/' invalid'
  ' allowed'     [5420]          — OK
  ' forbidden'   [36813]         — OK
  ' on'          [389]           — OK
  ' by'          [553]           — OK
  ' Yes'         [7414]          — OK
  ' No'          [2308]          — OK
  ' intensive'   [36195]         — OK
  ' extensive'   [16376]         — OK
  ' valid'       [2697]          — OK
  ' invalid'     [8318]          — OK
  ' same'        [1852]          — OK
  ' different'   [2155]          — OK
  ' possible'    [3204]          — OK
  ' impossible'  [11997]         — OK
  ' positive'    [6785]          — OK
  ' zero'        [7168]          — OK

Decisions:
  B1 (decay type):          ' alpha'  /  ' beta'       — x_or_y framing
  B3 (selection rule):      ' allowed'/ ' forbidden'   — x_or_y framing
  B4 (approx regime):       ' valid'  /  ' invalid'    — x_or_y framing (relativistic is multi-token)
  B5 (gauge equivalence):   ' same'   /  ' different'  — x_or_y framing
  B6 (intensive/extensive): ' intensive'/ ' extensive' — x_or_y framing
  B7 (entropy):             ' possible'/ ' impossible' — x_or_y framing
  B8 (isothermal):          ' on'     /  ' by'         — x_or_y framing
  B2 (decay chain):         DEFERRED — element symbol tokenization unpredictable
"""

from transformers import AutoTokenizer


def run_audit(tokens: list[str], model_name: str = "Qwen/Qwen3-4B") -> None:
    tok = AutoTokenizer.from_pretrained(model_name)
    print(f"Tokenizer: {model_name}\n")
    any_multi = False
    for word in tokens:
        ids = tok.encode(word, add_special_tokens=False)
        status = "OK" if len(ids) == 1 else "MULTI-TOKEN ⚠️"
        if len(ids) > 1:
            any_multi = True
        print(f"  {repr(word):25s} {str(ids):20s} — {status}")
    print()
    if any_multi:
        print("⚠  Some tokens are multi-token. Use alternative single-token answers for those.")
    else:
        print("✓  All tokens are single-token.")


if __name__ == "__main__":
    PLANNED_TOKENS = [
        " alpha", " beta",
        " classical", " relativistic",
        " allowed", " forbidden",
        " on", " by",
        " Yes", " No",
        " intensive", " extensive",
        " valid", " invalid",
        " same", " different",
        " possible", " impossible",
        " positive", " zero",
    ]
    run_audit(PLANNED_TOKENS)
