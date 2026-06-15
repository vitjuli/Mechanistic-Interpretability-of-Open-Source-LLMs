"""
128b_carrier_direction_resolved.py   [is the Q1 ablation effect SELECTIVE for u?]
==================================================================================
128 found: ablating the Q1 attribution-carrier features moves the margin ABOVE the
frequency-matched null (|dm| ~ 0.3-0.5 vs null ~0.2-0.38) but almost never flips
(intact-flip ~ 0). Two readings, and only a direction-resolved test separates them:

  (P) PARTIAL LOCALIZATION: Q1 features are SELECTIVE for the usage axis u -- their
      ablation delta lands disproportionately ALONG u (the causal coordinate),
      more than a frequency-matched random set does. Subcritical but real and
      direction-specific.
  (N) NO LOCALIZATION: Q1's larger |dm| is just because Q1 features are higher-norm
      / more active, so they perturb the residual more in ALL directions, including
      a bit along u -- no special alignment with the causal axis. The above-null
      |dm| is an activation-magnitude artifact, not localization.

The 128 run already stored, per layer, the ablation margin delta. To resolve
direction we need the ablation DELTA VECTOR in the residual, not just its margin
projection. This script recomputes the ablation on the SAME Q1 sets and the SAME
freq-matched nulls, but records the decoder-output delta vector at the decision
position, then measures:

   along_u      = | <delta_vec, u_hat> |                 (component on the causal axis)
   total_norm   = || delta_vec ||
   u_fraction   = along_u / total_norm                   (selectivity for u)
   margin_delta = the logit-margin change (cross-check vs 128)

and compares Q1's u_fraction and along_u against the null distribution. The verdict:
  - Q1 u_fraction AND along_u both exceed null p95 -> (P) partial localization:
    the carriers are direction-selective for u. State "partially concentrates".
  - Q1 along_u exceeds null but u_fraction does NOT (Q1 just has bigger total_norm)
    -> (N): above-null margin effect is magnitude, not direction. State "not
    localizable; the apparent effect is activation magnitude, not alignment".

Needs transcoders + model (GPU). Mirrors 128 exactly; only the measurement differs.

SELF-TEST (no torch / no repo):  python 128b_carrier_direction_resolved.py --self_test
"""

from __future__ import annotations

import argparse
import csv as _csv
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("carrier128b")


# =====================================================================
# Pure-numpy helpers (exercised by --self_test)
# =====================================================================
def unit_raw(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-30 else v


def decompose_delta(delta_vec, u_hat):
    """component of delta along u, total norm, and the fraction (selectivity)."""
    delta_vec = np.asarray(delta_vec, float); u_hat = unit_raw(np.asarray(u_hat, float))
    along = float(abs(delta_vec @ u_hat))
    tot = float(np.linalg.norm(delta_vec))
    return along, tot, (along / tot if tot > 1e-30 else 0.0)


def freq_matched_null_sets(all_feats, fire_rates, target_feats, n_sets, rng, tol=0.05):
    fr = dict(zip(all_feats, fire_rates))
    pool = [f for f in all_feats if f not in set(target_feats)]
    pool_fr = np.array([fr[f] for f in pool])
    sets = []
    for _ in range(n_sets):
        chosen, used = [], set()
        for tf in target_feats:
            tfr = fr[tf]
            cand = [i for i, f in enumerate(pool) if f not in used and abs(pool_fr[i] - tfr) <= tol]
            if not cand:
                order = np.argsort(np.abs(pool_fr - tfr))
                cand = [i for i in order if pool[i] not in used][:50]
            pick = pool[int(rng.choice(cand))]
            chosen.append(pick); used.add(pick)
        sets.append(chosen)
    return sets


# =====================================================================
# Self-test
# =====================================================================
def self_test():
    rng = np.random.default_rng(0)
    d = 60
    u = unit_raw(rng.standard_normal(d))

    # (P) a delta that is SELECTIVE for u: mostly along u
    dp = 1.0 * u + 0.1 * unit_raw(rng.standard_normal(d))
    a, t, f = decompose_delta(dp, u)
    assert f > 0.9, f"u-selective delta must have high u_fraction: {f}"

    # (N) a delta that is LARGE but isotropic: big norm, small u_fraction
    dn = 5.0 * unit_raw(rng.standard_normal(d))
    dn -= (dn @ u) * u                                  # remove u component -> isotropic-ish
    dn = dn + 0.3 * u
    an, tn, fn = decompose_delta(dn, u)
    assert tn > t and fn < 0.3, f"large isotropic delta: bigger norm, low u_fraction ({tn},{fn})"
    # along_u can be comparable while u_fraction separates them:
    assert fn < f, "u_fraction must distinguish selective from isotropic"

    # null sets keep size, exclude targets
    feats = list(range(150)); fire = rng.uniform(0, 1, 150); tgt = [3, 40, 90]
    sets = freq_matched_null_sets(feats, fire, tgt, 10, rng)
    assert all(len(s) == 3 and not (set(s) & set(tgt)) for s in sets)
    print("[self_test] OK — delta decomposition (selective vs isotropic), u_fraction separation, "
          "null sets pass.")


# =====================================================================
# Real run
# =====================================================================
def _chain(o, p):
    for x in p.split("."):
        o = getattr(o, x)
    return o


def reconstruct_split(fams, seed, train_frac):
    rng = np.random.default_rng(seed)
    fl = sorted(set(fams)); rng.shuffle(fl)
    train = set(fl[: int(round(len(fl) * train_frac))])
    return np.array([f in train for f in fams], bool)


def run_real(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    sys.path.insert(0, str(Path(args.repo_root)))
    from transcoder_loader import load_transcoder_set

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    feat_rows = list(_csv.DictReader(open(args.feature_metrics)))
    q1_by_layer = defaultdict(list); fire_by_layer = defaultdict(dict)
    for r in feat_rows:
        L = int(r["layer"]); f = int(r["feature"]); fr = float(r["fire_rate"])
        fire_by_layer[L][f] = fr
        if r.get("is_attr") == "1":
            q1_by_layer[L].append(f)
    layers = [L for L in args.layers if q1_by_layer.get(L)]

    dump = Path(args.dump_dir)
    meta = np.load(dump / "meta.npz", allow_pickle=True)
    fams = [json.loads(l)["surface_family"] for l in open(args.corpus)]
    prompts = [json.loads(l) for l in open(args.corpus)]
    y = meta["y"].astype(int); m0_all = meta["clean_margin"].astype(np.float64)
    id0 = meta["id_class0"].astype(int) if "id_class0" in meta else np.full(len(y), int(meta["alpha_id"]))
    id1 = meta["id_class1"].astype(int) if "id_class1" in meta else np.full(len(y), int(meta["beta_id"]))
    trm = reconstruct_split(fams, args.split_seed, args.train_frac)
    correct = ((y == 1) & (m0_all > 0)) | ((y == 0) & (m0_all < 0))
    held = np.where(~trm)[0]
    targets = [int(i) for i in held if correct[i]][: args.max_targets or None]

    # per-layer usage axis from the dump (u_hat)
    def u_hat_layer(L):
        G = np.load(dump / f"grad_L{L:02d}.npy").astype(np.float64)
        return unit_raw(G.mean(0))

    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.float32, low_cpu_mem_usage=True, trust_remote_code=True,
    ).to(args.device).eval()
    blocks = _chain(model, "model.layers"); mlp_of = lambda L: blocks[L].mlp
    TCSET = load_transcoder_set("4b", repo_id=args.transcoder_repo, device=args.device,
                                dtype=torch.float32, layers=layers)

    def ablation_delta_vec(i, L, feats, u_t):
        """returns (along_u, total_norm, margin_delta) for ablating `feats` at layer L
        on prompt i. delta_vec is the decoder-output change at the decision position."""
        tc = TCSET[L]
        inp = tok([prompts[i]["prompt"]], return_tensors="pt").to(args.device)
        feats_t = torch.tensor(sorted(feats), dtype=torch.long, device=args.device)
        store = {}
        def hook(mod, mlp_in, mlp_out):
            x = mlp_in[0]
            acts = tc.encode(x)
            recon_full = tc.decode(acts, x)
            if len(feats_t) > 0:
                acts_ab = acts.clone(); acts_ab[..., feats_t] = 0.0
                recon_ab = tc.decode(acts_ab, x)
                delta = recon_ab - recon_full
            else:
                delta = torch.zeros_like(recon_full)
            store["delta_last"] = delta[0, -1, :].detach().float().cpu().numpy()
            return mlp_out + delta
        h = mlp_of(L).register_forward_hook(hook)
        try:
            with torch.no_grad():
                row = model(**inp, use_cache=False).logits[0, -1, :].float()
            margin = float(row[int(id1[i])] - row[int(id0[i])])
        finally:
            h.remove()
        dv = store["delta_last"]
        along, tot, _ = decompose_delta(dv, u_t)
        return along, tot, margin

    rows = []
    for L in layers:
        u_t = u_hat_layer(L)
        q1 = q1_by_layer[L]
        # baseline margin (ablate nothing) to get the margin delta
        base = {i: ablation_delta_vec(i, L, [], u_t)[2] for i in targets}

        # Q1: per-prompt decomposition
        q1_along, q1_norm, q1_mdelta = [], [], []
        for i in targets:
            a, t, m = ablation_delta_vec(i, L, q1, u_t)
            q1_along.append(a); q1_norm.append(t); q1_mdelta.append(m - base[i])
        Q = {"along_u": float(np.mean(q1_along)), "total_norm": float(np.mean(q1_norm)),
             "u_fraction": float(np.mean(np.array(q1_along) / (np.array(q1_norm) + 1e-30))),
             "abs_mdelta": float(np.mean(np.abs(q1_mdelta)))}

        # frequency-matched null
        all_f = list(fire_by_layer[L].keys()); all_fr = [fire_by_layer[L][f] for f in all_f]
        null_sets = freq_matched_null_sets(all_f, all_fr, q1, args.n_null, rng, args.freq_tol)
        n_along, n_norm, n_frac = [], [], []
        for ns in null_sets:
            a_s, t_s = [], []
            for i in targets:
                a, t, _ = ablation_delta_vec(i, L, ns, u_t)
                a_s.append(a); t_s.append(t)
            n_along.append(float(np.mean(a_s))); n_norm.append(float(np.mean(t_s)))
            n_frac.append(float(np.mean(np.array(a_s) / (np.array(t_s) + 1e-30))))
        def p95(a): return float(np.quantile(a, 0.95))
        def mn(a): return float(np.mean(a))

        row = {"layer": L, "n_q1": len(q1),
               "q1_along_u": Q["along_u"], "null_along_u_p95": p95(n_along), "null_along_u_mean": mn(n_along),
               "q1_total_norm": Q["total_norm"], "null_total_norm_mean": mn(n_norm),
               "q1_u_fraction": Q["u_fraction"], "null_u_fraction_p95": p95(n_frac), "null_u_fraction_mean": mn(n_frac),
               "q1_abs_mdelta": Q["abs_mdelta"],
               "along_above_null": int(Q["along_u"] > p95(n_along)),
               "fraction_above_null": int(Q["u_fraction"] > p95(n_frac)),
               "norm_ratio_q1_vs_null": Q["total_norm"] / (mn(n_norm) + 1e-30)}
        rows.append(row)
        logger.info("L%02d: along_u Q1=%.4f (null p95 %.4f) | u_frac Q1=%.3f (null p95 %.3f) | "
                    "norm Q1/null=%.2f | along>null=%s frac>null=%s",
                    L, row["q1_along_u"], row["null_along_u_p95"], row["q1_u_fraction"],
                    row["null_u_fraction_p95"], row["norm_ratio_q1_vs_null"],
                    bool(row["along_above_null"]), bool(row["fraction_above_null"]))

    with open(out / "carrier_direction_resolved.csv", "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
        [w.writerow(r) for r in rows]

    n_along_above = sum(r["along_above_null"] for r in rows)
    n_frac_above = sum(r["fraction_above_null"] for r in rows)
    print("\n" + "=" * 96)
    print("CARRIER DIRECTION-RESOLVED — is the Q1 effect SELECTIVE for u, or just bigger norm?")
    print("=" * 96)
    for r in rows:
        verdict = ("u-SELECTIVE" if r["fraction_above_null"] else
                   ("magnitude-only" if r["along_above_null"] else "at null"))
        print(f"  L{r['layer']:02d}: along_u={r['q1_along_u']:.4f} (null p95 {r['null_along_u_p95']:.4f}) | "
              f"u_frac={r['q1_u_fraction']:.3f} (null p95 {r['null_u_fraction_p95']:.3f}) | "
              f"norm Q1/null={r['norm_ratio_q1_vs_null']:.2f}  -> {verdict}")
    print(f"\nlayers with along_u above null: {n_along_above}/{len(rows)} | "
          f"with u_FRACTION above null: {n_frac_above}/{len(rows)}")
    print("VERDICT:")
    print("  - u_fraction above null on most layers -> (P) PARTIAL LOCALIZATION: Q1 carriers are")
    print("    direction-selective for the causal axis. Write 'partially concentrates in the")
    print("    attribution-selected subset' (subcritical but real).")
    print("  - along_u above null but u_fraction NOT (norm ratio >1) -> (N) NO LOCALIZATION: the")
    print("    above-null margin effect is activation MAGNITUDE, not alignment. Write 'not")
    print("    localizable; apparent effect is bigger perturbation norm, not u-selectivity'.")
    print(f"saved: {out/'carrier_direction_resolved.csv'}")
    print("=" * 96 + "\n")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self_test", action="store_true")
    p.add_argument("--corpus", default="data/prompts/B1_alpha_beta.jsonl")
    p.add_argument("--dump_dir", default="data/analysis/runD_v2/B1_alpha_beta/field_dump")
    p.add_argument("--feature_metrics", default="data/analysis/feature_metrics_full.csv")
    p.add_argument("--out_dir", default="data/analysis/runD_v2/B1_alpha_beta/carrier_ablation")
    p.add_argument("--repo_root", default=".")
    p.add_argument("--transcoder_repo", default="mwhanna/qwen3-4b-transcoders")
    p.add_argument("--layers", type=int, nargs="*", default=[19, 20, 21, 22, 23, 24])
    p.add_argument("--n_null", type=int, default=20)
    p.add_argument("--freq_tol", type=float, default=0.05)
    p.add_argument("--max_targets", type=int, default=0)
    p.add_argument("--model_name", default="Qwen/Qwen3-4B")
    p.add_argument("--device", default="cuda")
    p.add_argument("--train_frac", type=float, default=0.6)
    p.add_argument("--split_seed", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    return p


def main():
    args = build_parser().parse_args()
    if args.self_test:
        self_test(); return
    run_real(args)


if __name__ == "__main__":
    main()
