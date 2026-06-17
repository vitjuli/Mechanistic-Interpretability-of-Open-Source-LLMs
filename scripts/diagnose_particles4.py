"""
Diagnostic for particles4 raw capture.
Reads meta.npz and prints what the model is actually predicting per class.
Usage:
  python scripts/diagnose_particles4.py
  python scripts/diagnose_particles4.py --dump data/analysis/runD_v2/particles4_raw/field_dump
"""
import argparse
from collections import Counter
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="data/analysis/runD_v2/particles4_raw/field_dump")
    ap.add_argument("--model_name", default="Qwen/Qwen3-4B")
    ap.add_argument("--top", type=int, default=15, help="how many top-1 tokens to list")
    ap.add_argument("--per_class_wrong", type=int, default=5, help="top wrong answers per class")
    args = ap.parse_args()

    dump = Path(args.dump)
    meta = np.load(dump / "meta.npz", allow_pickle=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    top1 = meta["baseline_top1"]
    class_tokens = [int(t) for t in meta["class_tokens"]]
    class_names = [str(n) for n in meta["class_names"]]
    y = meta["y"]
    n = len(y)

    print("=" * 80)
    print(f"Diagnostic — particles4 raw capture  (n={n})")
    print("=" * 80)

    print("\nclass_tokens (correct answers):")
    for c, t in zip(class_names, class_tokens):
        print(f"  {c:12s}  id={t:6d}  decoded={tok.decode([t])!r}")

    print(f"\nbaseline top-1 distribution (top {args.top} most common):")
    for tid, cnt in Counter(int(t) for t in top1).most_common(args.top):
        marker = "  <-- correct class" if tid in class_tokens else ""
        print(f"  {cnt:3d}×  id={tid:6d}  {tok.decode([tid])!r}{marker}")

    print(f"\nclean_accuracy (from meta): {float(meta['clean_accuracy']):.3f}")
    print(f"\nper-class breakdown:")
    for c, name in enumerate(class_names):
        sub = top1[y == c]
        correct_tok = class_tokens[c]
        n_correct = int((sub == correct_tok).sum())
        wrong = [int(t) for t in sub if int(t) != correct_tok]
        print(f"\n  [{name}] (n={len(sub)}, correct token={correct_tok!r}={tok.decode([correct_tok])!r}):")
        print(f"     n_correct = {n_correct}/{len(sub)}  ({n_correct/len(sub)*100:.1f}%)")
        if wrong:
            print(f"     top wrong answers:")
            for tid, cnt in Counter(wrong).most_common(args.per_class_wrong):
                tag = "  (other class!)" if tid in class_tokens else ""
                print(f"       {cnt:3d}×  id={tid:6d}  {tok.decode([tid])!r}{tag}")

    print("\n" + "=" * 80)
    print("READING:")
    print("  - if top-1 dominated by ' The', ' In', ' This' etc:")
    print("    model continues the prompt as TEXT, doesn't answer with a class token.")
    print("    -> Need few-shot scaffold so the model commits to a single-token class answer.")
    print("  - if top-1 dominated by ' Photon', ' Electron' (capitalized):")
    print("    model knows the answer but writes wrong CASE.")
    print("    -> Switch class tokens to capitalized variants OR add 'lowercase' few-shot.")
    print("  - if top-1 dominated by ' photons', ' electrons' (plural):")
    print("    -> Use plural single-tokens OR force singular via few-shot.")
    print("  - if top-1 is one of the correct 4 tokens but wrong class on each prompt:")
    print("    model classifies but answers the wrong particle.")
    print("    -> Real model failure, but our setup is OK; need to re-think task framing.")
    print("=" * 80)


if __name__ == "__main__":
    main()
