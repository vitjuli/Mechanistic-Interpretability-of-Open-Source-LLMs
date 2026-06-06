"""
81_cluster_union_cue_specificity.py   [STEP 2 — do cluster UNIONS realize cue-FAMILIES?]
=========================================================================================
Step 1 (cluster_cue_* matrices) showed single clusters are BROAD: their effect spreads
across 11-13 of 14 same-class cues (PR), with only a weak family gradient (quark/weak
dominant for C20/C18/C19). This script tests whether UNIONS sharpen onto a cue-FAMILY,
i.e. whether ablating a union breaks the model on ONE physical computation (charge
conservation, lepton number, mass/A, quark/weak, ...) and not the others.

DESIGN PRINCIPLE (sign-agnostic, per the "a cluster can carry charge-conservation and
be active on BOTH alpha and beta" insight): we DO NOT care which class a union supports.
A "flip" = the model's decision breaks after ablation (sign of correct-incorrect margin
changes), regardless of direction. Cue-family SPECIFICITY = flip-rate on the target
family's prompts vs flip-rate on OTHER families' prompts (same machinery for every cue).
A union "realizes" family X iff it flips family-X prompts much more than other families.
This is the decisive test that separates "implements cue X" from "default-and-suppress"
(which flips everything near the boundary uniformly).

REAL joint ablation (NOT the additive upper bound): reuses script 27's mechanism exactly
(encode post_attention_layernorm output -> zero the union's features -> decode -> patch
all layers simultaneously via ExitStack). The additive SFR CSVs are an upper bound for
RANKING only; antagonism (L18<->L24) can deflate the real effect, so every candidate is
measured, not estimated.

OPTIMISATION that makes "all pairs" feasible: the clean MLP input at each layer does NOT
depend on the union, so we cache clean_mlp[prompt][layer] and clean margins ONCE. Each
union then costs encode/zero/decode (no forward) + ONE patched forward per prompt.

MODES (run the two jobs in parallel as two SLURM jobs):
  --mode pairs     JOB A: all C(30,2) pairs, real ablation, per-family flip vector each.
                   Feeds offline agglomeration (combine pairs strong on the SAME family).
  --mode family    JOB B: for each cue-family, build the union of the top-m clusters that
                   TARGET that family (sign-agnostic |effect| from Step-1 matrix), ablate
                   it AND agglomeratively extend (add next-best family cluster up to k=5),
                   vs a random-union null on the SPECIFICITY metric (cheap, secondary).
  --mode candidates --candidates "20+18+19,2+5+18+19+32"
                   Ablate explicit unions (validate top-additive / agglomerated triples).

NULL (re-aimed per discussion): NOT "beat random union on overall SFR" (wrong target,
search artifact). The null is on cue-family SPECIFICITY only: does the target union's
family-specificity exceed that of a random union of the same size? Cheap and secondary;
the primary signal is the specificity contrast itself.

EXPECTATION (honest, from Step 1): single clusters are broad, so the likely outcome is
"distributed carriers, broad over the beta-state" (-> confirms the distributed thesis),
with quark/weak the only real shot at a family-specific positive. Both outcomes are clean.

SELF-TEST (no torch / no repo):  python 81_cluster_union_cue_specificity.py --self_test
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("union_cue")

# default cue-family grouping (override with --family_json)
DEFAULT_FAMILIES = {
    "charge_Z": ["daughter_z_minus2", "daughter_z_plus1", "emitted_charge_minus1",
                 "emitted_charge_plus2", "element_shift_minus2", "element_shift_plus1"],
    "mass_A": ["daughter_a_minus4", "daughter_a_unchanged", "emitted_mass4", "emitted_mass_negligible"],
    "quark_weak": ["quark_flavour_change", "quark_change_z_plus1", "no_quark_flavour_change",
                   "w_boson_mediation", "not_weak_force", "daughter_n_minus1"],
    "lepton": ["lepton_number_increases", "antineutrino_emitted", "lightest_charged_elementary"],
    "energy": ["continuous_energy_spectrum", "discrete_energy_spectrum"],
    "emission": ["emitted_2neutrons", "emitted_2protons", "cluster_ejection",
                 "cluster_ejection_daughter", "cluster_no_creation", "new_particles_created", "no_new_particles"],
}


# =====================================================================
# Pure-numpy analysis (exercised by --self_test)
# =====================================================================
def is_flip(base_margin: float, joint_margin: float) -> int:
    """Sign-agnostic: decision flips iff the sign of (correct-incorrect) margin changes."""
    return int((base_margin > 0) != (joint_margin > 0))


def family_flip_rates(flips: np.ndarray, fams: List[str], families: Dict[str, List[str]],
                      cue_fam: Dict[str, str], base_correct: np.ndarray) -> Dict[str, float]:
    """flips: per-prompt 0/1; fams: per-prompt cue_type; returns family -> flip-rate over
    baseline-correct prompts of that family."""
    out = {}
    for fam in families:
        mask = np.array([(cue_fam.get(c) == fam) and bc for c, bc in zip(fams, base_correct)])
        out[fam] = float(flips[mask].mean()) if mask.any() else float("nan")
    return out


def cue_specificity(fam_flips: Dict[str, float], target: str) -> Tuple[float, float]:
    """Returns (target flip-rate, contrast = target - max(other families)).
    Positive contrast = breaks the target family MORE than any other family."""
    t = fam_flips.get(target, float("nan"))
    others = [v for k, v in fam_flips.items() if k != target and not np.isnan(v)]
    if np.isnan(t) or not others:
        return t, float("nan")
    return t, float(t - max(others))


def family_strength_from_matrix(S: np.ndarray, cues: List[str], cluster_ids: List[str],
                                families: Dict[str, List[str]]) -> Dict[str, np.ndarray]:
    """Sign-agnostic per-cluster strength per family = mean |signed effect| over the
    family's cues. Returns family -> array over clusters."""
    ci = {c: j for j, c in enumerate(cues)}
    out = {}
    for fam, members in families.items():
        idx = [ci[c] for c in members if c in ci]
        out[fam] = np.abs(S[:, idx]).mean(1) if idx else np.zeros(len(cluster_ids))
    return out


def agglomerative_family_unions(strength: np.ndarray, k_max: int) -> List[List[int]]:
    """Given per-cluster strength for ONE family, return nested unions top-2..top-k_max
    (each adds the next-strongest family-targeting cluster). Agglomerative growth toward
    the family, NOT greedy-on-overall-SFR."""
    order = list(np.argsort(-strength))
    return [order[:k] for k in range(2, k_max + 1)]


def percentile_of(value: float, null: np.ndarray) -> float:
    null = np.asarray(null, float); null = null[~np.isnan(null)]
    return float(100.0 * np.mean(null <= value)) if null.size else float("nan")


# =====================================================================
# Self-test (synthetic; no torch)
# =====================================================================
def self_test() -> None:
    rng = np.random.default_rng(0)
    # synthetic: 8 clusters, families A/B/C; clusters 0,1 target family A strongly
    cues = ["a1", "a2", "b1", "b2", "c1", "c2"]
    fams = {"A": ["a1", "a2"], "B": ["b1", "b2"], "C": ["c1", "c2"]}
    cluster_ids = [str(i) for i in range(8)]
    S = 0.1 * rng.standard_normal((8, 6))
    S[0, 0:2] += 1.0; S[1, 0:2] += 0.9         # clusters 0,1 strong on family A
    S[2, 2:4] += 1.0                            # cluster 2 strong on family B
    strength = family_strength_from_matrix(S, cues, cluster_ids, fams)
    assert set(np.argsort(-strength["A"])[:2].tolist()) == {0, 1}, "family-A strength should rank clusters 0,1"
    unions = agglomerative_family_unions(strength["A"], k_max=4)
    assert unions[0] == list(np.argsort(-strength["A"])[:2]), "agglomeration should start from top-2"
    assert len(unions[-1]) == 4 and unions[0] == unions[-1][:2], "nested growth"

    # specificity: a union that flips family-A prompts but not others
    cue_fam = {c: f for f, members in fams.items() for c in members}
    prompt_cues = ["a1", "a1", "a2", "b1", "b2", "c1", "c2"]
    base = np.ones(len(prompt_cues))                      # all correct at baseline
    # union breaks only family-A prompts
    joint = np.array([-1, -1, -1, 1, 1, 1, 1.0])          # margin sign flips for a1,a1,a2
    flips = np.array([is_flip(b, j) for b, j in zip(base, joint)])
    ff = family_flip_rates(flips, prompt_cues, fams, cue_fam, base > 0)
    assert ff["A"] == 1.0 and ff["B"] == 0.0 and ff["C"] == 0.0, f"family-A specific, got {ff}"
    t, contrast = cue_specificity(ff, "A")
    assert t == 1.0 and contrast == 1.0, "perfect family-A specificity -> contrast 1.0"

    # a default-and-suppress union flips everything -> low contrast
    joint_all = -np.ones(len(prompt_cues))
    flips_all = np.array([is_flip(b, j) for b, j in zip(base, joint_all)])
    ff_all = family_flip_rates(flips_all, prompt_cues, fams, cue_fam, base > 0)
    _, contrast_all = cue_specificity(ff_all, "A")
    assert contrast_all == 0.0, "default-and-suppress -> zero specificity contrast"
    assert contrast > contrast_all, "family-specific union must beat default-and-suppress on contrast"

    assert percentile_of(0.9, np.array([0.1, 0.2, 0.3])) == 100.0
    print("[self_test] OK — family strength, agglomeration, flip, specificity, default-suppress contrast pass.")


# =====================================================================
# Real run
# =====================================================================
def run_real(args):
    import torch, yaml, contextlib
    from src.model_utils import ModelWrapper
    from src.transcoder import load_transcoder_set

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    families = json.loads(Path(args.family_json).read_text()) if args.family_json else DEFAULT_FAMILIES
    cue_fam = {c: f for f, members in families.items() for c in members}

    # ---- machinery copied from script 27 (kept identical for compatibility) ----
    def get_mlp_input(model, inputs, layer_idx, token_pos=-1):
        blocks = model.model.model.layers
        cap = {}
        def hook(m, i, o):
            t = o[0] if isinstance(o, tuple) else o; cap["x"] = t.detach()
        h = blocks[layer_idx].post_attention_layernorm.register_forward_hook(hook)
        try:
            with torch.no_grad():
                model.model(**inputs, use_cache=False)
        finally:
            h.remove()
        return cap["x"][:, token_pos, :]

    @contextlib.contextmanager
    def patch_mlp_layer(model_hf, layer_idx, token_pos, new_in):
        block = model_hf.model.layers[layer_idx]
        called = {"n": 0}
        def hook(m, i, o):
            called["n"] += 1
            t = o[0] if isinstance(o, tuple) else o
            t = t.clone(); t[:, token_pos, :] = new_in.to(t.dtype).to(t.device)
            return (t,) + o[1:] if isinstance(o, tuple) else t
        h = block.post_attention_layernorm.register_forward_hook(hook)
        try:
            yield
        finally:
            h.remove()
        assert called["n"] > 0, f"patch hook at layer {layer_idx} never fired"

    def margin(logits, cid, iid):
        lp = torch.log_softmax(logits, dim=0)
        return float(lp[cid] - lp[iid])

    # ---- load clusters ----
    clu_dir = Path(args.clustering_dir) if Path(args.clustering_dir).is_absolute() else ROOT / args.clustering_dir
    import csv as _csv
    with open(clu_dir / "cluster_labels.csv") as f:
        rows = list(_csv.DictReader(f))
    if args.cluster_col not in rows[0]:
        raise SystemExit(f"column '{args.cluster_col}' not found in {clu_dir}/cluster_labels.csv. "
                         f"Available columns: {sorted(rows[0].keys())[:20]}...")
    coimp = {r["feature_id"]: int(r[args.cluster_col]) for r in rows}
    from collections import defaultdict
    clusters = defaultdict(list)
    for r in rows:
        clusters[int(r[args.cluster_col])].append(r["feature_id"])

    def parse_lf(fid):
        return int(fid.split("_")[0][1:]), int(fid.split("_F")[1])

    def cluster_by_layer(cid):
        d = defaultdict(list)
        for fid in clusters[cid]:
            l, ff = parse_lf(fid); d[l].append(ff)
        return dict(d)

    def union_by_layer(cids):
        d = defaultdict(list)
        for cid in cids:
            for l, fs in cluster_by_layer(cid).items():
                d[l].extend(fs)
        return {l: sorted(set(v)) for l, v in d.items()}

    all_cluster_ids = sorted(clusters.keys())

    # ---- load Step-1 matrix for family grouping (sign-agnostic strength) ----
    def load_wide(fn):
        with open(fn) as f:
            r = _csv.reader(f); hdr = next(r); rr = list(r)
        return [x[0] for x in rr], hdr[1:], np.array([[float(v) for v in x[1:]] for x in rr])
    s1_ids, s1_cues, S1 = load_wide(args.step1_signed_effect)
    fam_strength = family_strength_from_matrix(S1, s1_cues, s1_ids, families)  # family -> array over s1_ids
    s1_id_to_cluster = [int(x) for x in s1_ids]

    # ---- load prompts ----
    ppath = ROOT / "data/prompts" / f"{args.behaviour}_{args.split}.jsonl"
    prompts = [json.loads(l) for l in open(ppath)]
    if args.n_prompts:
        prompts = prompts[: args.n_prompts]
    logger.info("Loaded %d prompts; %d clusters; families: %s", len(prompts), len(all_cluster_ids), list(families))

    # ---- model + transcoders ----
    with open(ROOT / "configs/transcoder_config.yaml") as f:
        tc_cfg = yaml.safe_load(f)
    model_size = tc_cfg.get("model_size", "4b")
    model_name = tc_cfg["transcoders"][model_size]["model_name"]
    layers_needed = sorted({l for cid in all_cluster_ids for l in cluster_by_layer(cid)})
    model = ModelWrapper(model_name=model_name, dtype="bfloat16", device="auto", trust_remote_code=True)
    device = next(model.model.parameters()).device
    transcoder_set = load_transcoder_set(model_size=model_size, device=device, dtype=torch.bfloat16,
                                         lazy_load=True, layers=layers_needed)
    logger.info("model on %s; transcoder layers %s", device, layers_needed)

    def tok_ids(p):
        cid = model.tokenizer.encode(p[args.correct_key], add_special_tokens=False)[0]
        iid = model.tokenizer.encode(p[args.incorrect_key], add_special_tokens=False)[0]
        return cid, iid

    # ---- PRECOMPUTE clean MLP inputs (per prompt, per layer) + clean margins ONCE ----
    logger.info("Precomputing clean MLP inputs for %d prompts x %d layers...", len(prompts), len(layers_needed))
    clean_mlp = []      # list over prompts: {layer: tensor(H,)}
    clean_margin = []; cue_types = []; base_correct = []
    for i, p in enumerate(prompts):
        inputs = model.tokenize([p["prompt"]]); inputs = {k: v.to(device) for k, v in inputs.items()}
        per = {}
        # one forward with hooks on all needed layers
        caps = {}
        hooks = []
        blocks = model.model.model.layers
        for L in layers_needed:
            def mk(L=L):
                def hook(m, ii, o):
                    t = o[0] if isinstance(o, tuple) else o; caps[L] = t[:, -1, :].detach()
                return hook
            hooks.append(blocks[L].post_attention_layernorm.register_forward_hook(mk()))
        with torch.no_grad():
            o = model.model(**inputs, use_cache=False)
        for h in hooks:
            h.remove()
        for L in layers_needed:
            per[L] = caps[L].squeeze(0)
        clean_mlp.append(per)
        cid, iid = tok_ids(p)
        m = margin(o.logits[0, -1, :], cid, iid)
        clean_margin.append(m); base_correct.append(m > 0)
        cue_types.append(p.get(args.cue_key, "?"))
        if (i + 1) % 100 == 0:
            logger.info("  precompute %d/%d", i + 1, len(prompts))
    clean_margin = np.array(clean_margin); base_correct = np.array(base_correct)

    def ablate_union_flips(cids):
        """Real joint ablation of the union; returns per-prompt flip array (sign-agnostic)."""
        ubl = union_by_layer(cids)
        flips = np.zeros(len(prompts), int)
        for i, p in enumerate(prompts):
            inputs = model.tokenize([p["prompt"]]); inputs = {k: v.to(device) for k, v in inputs.items()}
            mod_per = {}
            for L, feat_idx in ubl.items():
                act = clean_mlp[i][L].unsqueeze(0)
                tc = transcoder_set[L]
                with torch.no_grad():
                    fe = tc.encode(act.to(tc.dtype)); fe[:, feat_idx] = 0.0
                    mod_per[L] = tc.decode(fe).to(act.dtype).squeeze(0)
            with contextlib.ExitStack() as stack:
                for L, mi in mod_per.items():
                    stack.enter_context(patch_mlp_layer(model.model, L, -1, mi))
                with torch.no_grad():
                    jo = model.model(**inputs, use_cache=False)
            cid, iid = tok_ids(p)
            flips[i] = is_flip(clean_margin[i], margin(jo.logits[0, -1, :], cid, iid))
        return flips

    def specificity_record(cids):
        flips = ablate_union_flips(cids)
        ff = family_flip_rates(flips, cue_types, families, cue_fam, base_correct)
        overall = float(flips[base_correct].mean()) if base_correct.any() else float("nan")
        best_fam = max((f for f in ff if not np.isnan(ff[f])), key=lambda f: ff[f], default=None)
        _, contrast = cue_specificity(ff, best_fam) if best_fam else (np.nan, np.nan)
        return {"ids": "+".join(str(c) for c in cids), "k": len(cids), "overall_flip": overall,
                "best_family": best_fam, "best_family_contrast": contrast,
                **{f"flip_{f}": ff[f] for f in families}}

    records = []

    # ---------- MODE: pairs ----------
    if args.mode == "pairs":
        pairs = list(combinations(all_cluster_ids, 2))
        # optional slicing for parallelisation: --pair_start / --pair_end
        a = max(0, args.pair_start) if args.pair_start is not None else 0
        b = min(len(pairs), args.pair_end) if args.pair_end is not None else len(pairs)
        pairs_slice = pairs[a:b]
        logger.info("JOB A: real ablation of %d pairs (slice %d..%d of %d total)...",
                    len(pairs_slice), a, b, len(pairs))
        for j, pr in enumerate(pairs_slice):
            records.append(specificity_record(list(pr)))
            if (j + 1) % 20 == 0:
                logger.info("  pair %d/%d (global %d)", j + 1, len(pairs_slice), a + j + 1)

    # ---------- MODE: family (agglomerative + null) ----------
    elif args.mode == "family":
        if args.families:
            fams_run = [f for f in families if f in args.families]
            if not fams_run:
                raise SystemExit(f"--families {args.families} not in {list(families)}")
        else:
            fams_run = list(families)
        logger.info("JOB B: family-grouped agglomerative unions + random-union null... (families=%s)", fams_run)
        for fam in fams_run:
            strength = fam_strength[fam]                      # over s1_ids order
            nested_local = agglomerative_family_unions(strength, args.k_max)
            for local_idx in nested_local:
                cids = [s1_id_to_cluster[i] for i in local_idx]
                rec = specificity_record(cids); rec["target_family"] = fam
                # random-union null on specificity contrast for THIS family, same size
                null_contr = []
                for _ in range(args.n_random_union):
                    rc = list(rng.choice(all_cluster_ids, size=len(cids), replace=False))
                    fl = ablate_union_flips(rc)
                    ff = family_flip_rates(fl, cue_types, families, cue_fam, base_correct)
                    _, c = cue_specificity(ff, fam); null_contr.append(c)
                rec["null_contrast_mean"] = float(np.nanmean(null_contr))
                rec["null_contrast_p95"] = float(np.nanpercentile(null_contr, 95))
                rec["target_pct_vs_null"] = percentile_of(rec["best_family_contrast"], np.array(null_contr)) \
                    if rec["best_family"] == fam else float("nan")
                records.append(rec)
                logger.info("  %s k=%d: flip_%s=%.2f contrast=%.2f (null p95=%.2f)", fam, len(cids), fam,
                            rec.get(f"flip_{fam}", float('nan')), rec["best_family_contrast"], rec["null_contrast_p95"])

    # ---------- MODE: candidates ----------
    elif args.mode == "candidates":
        cands = [[int(x) for x in c.split("+")] for c in args.candidates.split(",")] if args.candidates else []
        logger.info("Validation: %d explicit unions...", len(cands))
        for cids in cands:
            records.append(specificity_record(cids))

    # ---- save ----
    import csv as _csv2
    keys = sorted({k for r in records for k in r})
    fn = out / f"union_cue_{args.mode}.csv"
    with open(fn, "w", newline="") as f:
        w = _csv2.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in records:
            w.writerow(r)
    logger.info("wrote %s (%d rows)", fn, len(records))


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self_test", action="store_true")
    p.add_argument("--mode", choices=["pairs", "family", "candidates"], default="family")
    p.add_argument("--behaviour", default="physics_decay_type_probe_v2")
    p.add_argument("--split", default="train")
    p.add_argument("--out_dir", default="data/analysis/runD_v2/union_cue")
    p.add_argument("--clustering_dir", default="data/analysis/runD_v2/clustering_full",
                   help="directory with cluster_labels.csv (default: 30-subgroup labels)")
    p.add_argument("--cluster_col", default="agglo_coimp_subgroup_k30",
                   help="column in cluster_labels.csv that gives cluster IDs (default: 30 subgroups)")
    p.add_argument("--step1_signed_effect",
                   default="data/analysis/runD_v2/cluster_joint_ablation_subgroup/cluster_cue_signed_effect.csv")
    p.add_argument("--family_json", default=None, help="optional JSON override of cue-family grouping")
    p.add_argument("--correct_key", default="correct_answer")
    p.add_argument("--incorrect_key", default="incorrect_answer")
    p.add_argument("--cue_key", default="cue_type")
    p.add_argument("--k_max", type=int, default=5, help="max union size in agglomerative family growth")
    p.add_argument("--n_random_union", type=int, default=30, help="random-union null draws (on specificity)")
    p.add_argument("--candidates", default=None, help="for --mode candidates: '20+18+19,2+5+18+19+32'")
    p.add_argument("--pair_start", type=int, default=None, help="for --mode pairs: index of first pair to process (parallelisation)")
    p.add_argument("--pair_end",   type=int, default=None, help="for --mode pairs: index after last pair (exclusive)")
    p.add_argument("--families", type=str, nargs="*", default=None, help="for --mode family: restrict to these family names")
    p.add_argument("--n_prompts", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    return p


def main():
    args = build_parser().parse_args()
    if args.self_test:
        self_test(); return
    run_real(args)


if __name__ == "__main__":
    main()
