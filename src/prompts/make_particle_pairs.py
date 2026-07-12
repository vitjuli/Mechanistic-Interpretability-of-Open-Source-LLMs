"""
make_particle_pairs.py  —  slice 6 binary pairs from the 4-particle multiclass corpus
=====================================================================================
From the multiclass particle corpus (electron/neutron/photon/proton, 80 each, 320 balanced),
write 6 balanced binary JSONL files (one per pair) for 154/119 dumps. No generation -- pure filter.

Class is read from --class_key (default: correct_answer, which holds the particle name, e.g.
" electron"). Particles are matched case-insensitively by substring so leading spaces / formatting
don't matter. Only the 4 main particles are kept (positron/muon dropped, as in the K=4 analysis).

USAGE:
  python make_particle_pairs.py \
    --corpus data/prompts/physics_internal_candidate_selection_v2_train.jsonl \
    --class_key correct_answer \
    --out_dir data/prompts/particle_pairs
SELF-TEST: python make_particle_pairs.py --self_test
"""
from __future__ import annotations
import argparse, json, logging
from pathlib import Path
from itertools import combinations

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("pairs")

PARTICLES = ["electron", "neutron", "photon", "proton"]

def particle_of(p, class_key):
    v = str(p.get(class_key, "")).strip().lower()
    for part in PARTICLES:
        if part in v:
            return part
    return None

def run(args):
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    prompts = [json.loads(l) for l in open(args.corpus)]
    # bucket by particle
    by_part = {p: [] for p in PARTICLES}
    dropped = 0
    for pr in prompts:
        part = particle_of(pr, args.class_key)
        if part is None:
            dropped += 1; continue
        by_part[part].append(pr)
    logger.info("corpus %d | per particle: %s | dropped(non-4particle): %d",
                len(prompts), {k: len(v) for k, v in by_part.items()}, dropped)
    if dropped == len(prompts):
        logger.warning("ALL dropped -- check --class_key (got 0 matches for %s)", PARTICLES)

    # write 6 pairs, class0 = first particle (alpha-role), class1 = second (beta-role) by convention
    manifest = {}
    for a, b in combinations(PARTICLES, 2):
        pa, pb = by_part[a], by_part[b]
        n = min(len(pa), len(pb))
        if n == 0:
            logger.warning("pair %s_vs_%s: one side empty, skipping", a, b); continue
        # balance: take min(n) from each (already 80/80 if balanced corpus)
        rows = []
        for pr in pa[:n]:
            pr = dict(pr); pr["_pair_class"] = 0; pr["_pair"] = f"{a}_vs_{b}"; pr["_particle"] = a
            rows.append(pr)
        for pr in pb[:n]:
            pr = dict(pr); pr["_pair_class"] = 1; pr["_pair"] = f"{a}_vs_{b}"; pr["_particle"] = b
            rows.append(pr)
        fn = out / f"particles_{a}_vs_{b}.jsonl"
        with open(fn, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        manifest[f"{a}_vs_{b}"] = {"file": str(fn), "n_class0": n, "n_class1": n,
                                   "class0": a, "class1": b}
        logger.info("wrote %s  (%d/%d, class0=%s class1=%s)", fn.name, n, n, a, b)
    json.dump(manifest, open(out / "pairs_manifest.json", "w"), indent=2)
    logger.info("manifest -> %s | %d pairs", out / "pairs_manifest.json", len(manifest))
    print("\nNEXT: use _pair_class as the binary label (class1=beta-role) in 154/119:")
    print("      --label_key _pair_class --pos_value 1   (or your dump's native label flag)")

def self_test():
    import tempfile, os
    d = tempfile.mkdtemp()
    corpus = os.path.join(d, "c.jsonl")
    rows = []
    for part in PARTICLES:
        for i in range(5):
            rows.append({"prompt": f"q {part} {i}", "correct_answer": f" {part}"})
    rows.append({"prompt": "muon q", "correct_answer": " muon"})  # should be dropped
    open(corpus, "w").write("\n".join(json.dumps(r) for r in rows))
    from types import SimpleNamespace
    run(SimpleNamespace(corpus=corpus, class_key="correct_answer", out_dir=os.path.join(d, "out")))
    man = json.load(open(os.path.join(d, "out", "pairs_manifest.json")))
    assert len(man) == 6, f"expected 6 pairs, got {len(man)}"
    assert man["electron_vs_photon"]["n_class0"] == 5, "balanced 5/5"
    # verify muon dropped: each pair has only its 2 particles
    ep = [json.loads(l) for l in open(man["electron_vs_photon"]["file"])]
    parts = set(r["_particle"] for r in ep)
    assert parts == {"electron", "photon"}, f"pair must contain only its 2 particles, got {parts}"
    assert all(r["_pair_class"] in (0, 1) for r in ep), "binary label present"
    print("[self_test] OK - 6 balanced pairs, muon dropped, binary _pair_class label. pass.")

def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self_test", action="store_true")
    p.add_argument("--corpus"); p.add_argument("--class_key", default="correct_answer")
    p.add_argument("--out_dir", default="particle_pairs")
    return p

if __name__ == "__main__":
    a = build_parser().parse_args()
    if a.self_test: self_test()
    elif a.corpus: run(a)
    else: raise SystemExit("need --corpus (or --self_test)")
