"""
76_recall_baseline_precheck.py   [CSD3 / GPU, light: one forward pass per prompt]
===================================================================
Cheap PRE-CHECK before building the inferential-depth gradient (D0->D1->D2).
The gradient's RECALL end (D0) is only valid if the model actually KNOWS the decay
mode of named isotopes from pretraining, WITHOUT any physics cue in the prompt. This
script tests exactly that -- and, crucially, separates genuine recall from a cheap
heuristic.

THE CONFOUND: "famous isotope -> decay mode" overlaps with the heuristic
"heavy nucleus -> alpha, light -> beta". Most famous cases obey it (Po-210 heavy->a,
C-14 light->b). So high accuracy could be the heuristic, NOT recall of that isotope.

THE FIX: include HEURISTIC-BREAKING isotopes --
  * heavy beta-emitters (Bi-210, Pb-210, Pb-214, Ac-228): heuristic says alpha, truth beta
  * medium-mass alpha-emitters (Sm-147, Gd-152, Nd-144): heuristic says beta, truth alpha
If the model gets the BREAKING set right, it is doing real recall. If it only gets the
heuristic-CONSISTENT set right, D0 is heuristic, not recall -> the gradient's recall
end is invalid and must be rethought.

NO PHYSICS CUE: prompts name the isotope only ("Polonium-210 undergoes ___ decay"),
never mention emitted particle / charge / mass-number reasoning. Otherwise the model
could DERIVE the answer (that would be D1/D2, not recall).

OUTPUT: overall accuracy; accuracy on heuristic-consistent vs heuristic-breaking;
the heuristic's own accuracy (reference); per-isotope table; verdict on D0 validity.
Also writes the prompts as JSONL so a valid D0 can feed the battery directly.

SELF-TEST: python 76_recall_baseline_precheck.py --self_test
"""

from __future__ import annotations
import argparse, json, logging
from pathlib import Path
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("recall_precheck")

# (element, full_name, mass, true_decay, heuristic_class)
#   heuristic "mass >= 210 -> alpha, else beta"
#   class: "consistent" if heuristic == truth, "breaking" if heuristic != truth
# Physics verified; only confident, well-attested decay modes included.
ISOTOPES = [
    # --- heuristic-consistent ALPHA (heavy, famous) ---
    ("Po", "Polonium", 210, "alpha", "consistent"),
    ("U",  "Uranium",  238, "alpha", "consistent"),
    ("U",  "Uranium",  235, "alpha", "consistent"),
    ("Ra", "Radium",   226, "alpha", "consistent"),
    ("Rn", "Radon",    222, "alpha", "consistent"),
    ("Am", "Americium",241, "alpha", "consistent"),
    ("Pu", "Plutonium",239, "alpha", "consistent"),
    ("Th", "Thorium",  232, "alpha", "consistent"),
    # --- heuristic-consistent BETA (light, famous) ---
    ("C",  "Carbon",    14, "beta", "consistent"),
    ("H",  "Hydrogen",   3, "beta", "consistent"),
    ("Sr", "Strontium", 90, "beta", "consistent"),
    ("I",  "Iodine",   131, "beta", "consistent"),
    ("Cs", "Cesium",   137, "beta", "consistent"),
    ("P",  "Phosphorus",32, "beta", "consistent"),
    ("Co", "Cobalt",    60, "beta", "consistent"),
    # --- heuristic-BREAKING: heavy BETA (heuristic says alpha, truth beta) ---
    ("Bi", "Bismuth",  210, "beta", "breaking"),
    ("Pb", "Lead",     210, "beta", "breaking"),
    ("Pb", "Lead",     214, "beta", "breaking"),
    ("Ac", "Actinium", 228, "beta", "breaking"),
    # --- heuristic-BREAKING: medium-mass ALPHA (heuristic says beta, truth alpha) ---
    ("Sm", "Samarium", 147, "alpha", "breaking"),
    ("Gd", "Gadolinium",152,"alpha", "breaking"),
    ("Nd", "Neodymium",144, "alpha", "breaking"),
]

SURFACE_TEMPLATES = [
    "{full}-{m} undergoes ___ decay. The decay type is",
    "The radioactive isotope {sym}-{m} decays by emitting a particle. Its decay type is",
    "{full}-{m} is a radioactive nuclide. Its mode of radioactive decay is",
    "Question: what is the decay type of {sym}-{m}? Answer:",
    "Nuclide {full}-{m}; radioactive decay classification:",
]


def heuristic_pred(mass):
    return "alpha" if mass >= 210 else "beta"


def build_prompts():
    rows = []
    for (sym, full, m, truth, cls) in ISOTOPES:
        fam = f"{sym}-{m}"
        for ti, tpl in enumerate(SURFACE_TEMPLATES):
            text = tpl.format(full=full, sym=sym, m=m)
            rows.append({
                "prompt": text,
                "correct_answer": " " + truth,            # leading space (match 02)
                "incorrect_answer": " " + ("beta" if truth == "alpha" else "alpha"),
                "surface_family": fam, "isotope": fam, "mass": m,
                "heuristic_class": cls, "heuristic_pred": heuristic_pred(m),
                "level": "D0_recall",
            })
    return rows


# =====================================================================
# Self-test: simulate a RECALL model (knows all) vs a HEURISTIC model
# (mass rule only); confirm the breakdown separates them.
# =====================================================================
def self_test():
    rows = build_prompts()
    truth = np.array([r["correct_answer"].strip() for r in rows])
    cls = np.array([r["heuristic_class"] for r in rows])
    heur = np.array([r["heuristic_pred"] for r in rows])

    def acc(pred, mask=None):
        m = np.ones(len(pred), bool) if mask is None else mask
        return float((pred[m] == truth[m]).mean())

    recall_pred = truth.copy()                       # perfect recall
    heuristic_pred_arr = heur.copy()                 # mass rule only

    print("\n--- SELF TEST -------------------------------------------------")
    print(f"  isotopes={len({r['isotope'] for r in rows})}  prompts={len(rows)}  "
          f"(consistent={int((cls=='consistent').sum())}, breaking={int((cls=='breaking').sum())})")
    print(f"  RECALL model    : overall={acc(recall_pred):.2f}  "
          f"breaking={acc(recall_pred, cls=='breaking'):.2f}")
    print(f"  HEURISTIC model : overall={acc(heuristic_pred_arr):.2f}  "
          f"breaking={acc(heuristic_pred_arr, cls=='breaking'):.2f}")
    # the discriminator: accuracy on the BREAKING set
    assert acc(recall_pred, cls == "breaking") > 0.9, "recall model must ace breaking set"
    assert acc(heuristic_pred_arr, cls == "breaking") < 0.1, "heuristic model must fail breaking set"
    # heuristic does well overall (confound) but fails breaking -> that's the whole point
    assert acc(heuristic_pred_arr) > 0.6, "heuristic should look OK overall (the confound)"
    print("\nALL SELF-TEST ASSERTIONS PASSED ")
    print("(accuracy on the heuristic-BREAKING set is what distinguishes genuine recall")
    print(" from the heavy=alpha heuristic; overall accuracy alone cannot)")
    print("---------------------------------------------------------------\n")


# =====================================================================
# Real run
# =====================================================================
def run_real(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    rows = build_prompts()
    # save prompts (a valid D0 feeds the battery)
    with open(out / "physics_decay_D0_recall.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    logger.info("built %d recall prompts over %d isotopes", len(rows), len({r["isotope"] for r in rows}))

    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.float32, low_cpu_mem_usage=True, trust_remote_code=True,
    ).to(args.device).eval()

    def score(ptext, a_tok, b_tok):
        inp = tok([ptext], return_tensors="pt").to(args.device)
        with torch.no_grad():
            o = model(**inp, use_cache=False)
        lp = torch.log_softmax(o.logits[0, -1, :].float(), 0)
        ia = tok.encode(a_tok, add_special_tokens=False)[0]
        ib = tok.encode(b_tok, add_special_tokens=False)[0]
        return float(lp[ia]), float(lp[ib])

    preds, conf = [], []
    for i, r in enumerate(rows):
        la, lb = score(r["prompt"], " alpha", " beta")
        preds.append("alpha" if la > lb else "beta")
        conf.append(abs(la - lb))
        if (i + 1) % 25 == 0:
            logger.info("  scored %d/%d", i + 1, len(rows))
    preds = np.array(preds)
    truth = np.array([r["correct_answer"].strip() for r in rows])
    cls = np.array([r["heuristic_class"] for r in rows])
    heur = np.array([r["heuristic_pred"] for r in rows])
    mass = np.array([r["mass"] for r in rows])

    def acc(p, mask=None):
        m = np.ones(len(p), bool) if mask is None else mask
        return float((p[m] == truth[m]).mean()) if m.any() else float("nan")

    overall = acc(preds)
    acc_cons = acc(preds, cls == "consistent")
    acc_break = acc(preds, cls == "breaking")
    heur_overall = acc(heur)
    heur_break = acc(heur, cls == "breaking")
    # model-vs-heuristic agreement on breaking set (high agreement => model using heuristic)
    agree_break = float((preds[cls == "breaking"] == heur[cls == "breaking"]).mean())

    # per-isotope accuracy
    per_iso = {}
    for iso in sorted({r["isotope"] for r in rows}):
        m = np.array([r["isotope"] == iso for r in rows])
        per_iso[iso] = {"acc": acc(preds, m), "truth": truth[m][0], "class": cls[m][0],
                        "mass": int(mass[m][0])}

    valid = bool(acc_break >= args.recall_thr)
    res = {
        "overall_acc": overall, "acc_heuristic_consistent": acc_cons,
        "acc_heuristic_breaking": acc_break,
        "heuristic_own_acc_overall": heur_overall, "heuristic_own_acc_breaking": heur_break,
        "model_heuristic_agreement_on_breaking": agree_break,
        "n_isotopes": len(per_iso), "n_prompts": len(rows), "per_isotope": per_iso,
        "verdict": (
            f"D0 RECALL VALID: accuracy on heuristic-breaking isotopes is {acc_break:.2f} "
            f"(>= {args.recall_thr}), well above the heuristic's own {heur_break:.2f} on that set. "
            "The model genuinely recalls decay modes, not just the heavy=alpha rule. The recall end "
            "of the gradient is sound; build D0->D1->D2."
            if valid else
            f"D0 RECALL INVALID/WEAK: accuracy on heuristic-breaking isotopes is only {acc_break:.2f}, "
            f"near the heuristic's {heur_break:.2f} (agreement {agree_break:.2f}). The model appears to "
            "use the heavy=alpha heuristic rather than recall specific isotopes. The 'recall' end is not "
            "clean -- rethink D0 (e.g. use only the most famous isotopes, or drop the recall level and "
            "anchor the control with sentiment instead).")}

    with open(out / "recall_precheck.json", "w") as fh:
        json.dump(res, fh, indent=2, default=float)

    print("\n" + "=" * 80)
    print("D0 RECALL PRE-CHECK  --  does the model recall decay modes (vs heavy=alpha)?")
    print("=" * 80)
    print(f"  overall accuracy                 : {overall:.3f}  ({len(rows)} prompts)")
    print(f"  heuristic-consistent subset      : {acc_cons:.3f}")
    print(f"  heuristic-BREAKING subset        : {acc_break:.3f}   <-- the decisive number")
    print(f"  (heuristic's own acc on breaking : {heur_break:.3f})")
    print(f"  model agrees w/ heuristic break  : {agree_break:.3f}  (high => using heuristic)")
    print("\n  per-isotope (breaking marked *):")
    for iso, d in per_iso.items():
        star = "*" if d["class"] == "breaking" else " "
        print(f"    {star}{iso:<8} m={d['mass']:<4} truth={d['truth']:<6} acc={d['acc']:.2f}")
    print("\nVERDICT: " + res["verdict"])
    print(f"\nwrote: {out}/recall_precheck.json + physics_decay_D0_recall.jsonl")
    print("=" * 80)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self_test", action="store_true")
    p.add_argument("--model_name", default="Qwen/Qwen3-4B")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out_dir", default="data/prompts")
    p.add_argument("--recall_thr", type=float, default=0.75,
                   help="min accuracy on heuristic-breaking set to call D0 valid recall")
    return p


def main():
    a = build_parser().parse_args()
    if a.self_test:
        self_test(); return
    run_real(a)


if __name__ == "__main__":
    main()
