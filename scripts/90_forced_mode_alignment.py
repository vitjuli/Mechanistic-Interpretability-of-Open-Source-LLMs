"""
90_forced_mode_alignment.py   [does the usage direction ROTATE when the shortcut is taken away?]
==============================================================================================
Exp 86 measured cos(u, w_res) ~ 0.03 in the base task regime: the model's readout ignores the
readable concept axis -- consistent with a surface shortcut (H2). This script tests H2
CONSTRUCTIVELY by changing the regime, not the direction:

(A) FORCED-ANSWER REGIME. Each prompt is wrapped in a balanced few-shot template whose next
    token is literally " alpha"/" beta". Two consequences: (i) the model is pushed to route
    the concept into the output; (ii) the intact-flip ceiling disappears (clean top-1 CAN be
    an answer token -- reported as clean_intact_rate; if it stays low the script says so and
    falls back to margin metrics). We re-capture residuals + margin gradients and re-measure,
    per layer: cos(u_forced, w_res_forced), cos(u_forced, u_base) (rotation of the usage
    itself), AUC along u_forced. HEADLINE: does median |cos(u, w_res)| jump from ~0.03?
      jump  -> the same readable axis BECOMES used when the bypass is closed (mechanistic
               proof of H2: causality of the representation is task-regime-conditional);
      no jump -> the representation is epiphenomenal even under forcing (strong claim too).

(B) INTACT STEERING WITH THE CEILING REMOVED. In the forced regime, steering along
    w_res_forced and u_forced at selected taps, metric = intact-flip (top-1 actually becomes
    the opposite answer token), vs shuffled-label-direction and random-direction nulls of
    equal amplitude. This is the behavioural test exp 65/75/85 could not run honestly.

(C) CUE-SUBSET ALIGNMENT (offline, base regime). Recompute u from gradients of only those
    prompts whose cue family is NOT heuristic-applicable (exclude mass_A, emission by
    default; cue types joined from a per-prompt CSV if the jsonl lacks a cue field). If
    cos(u_subset, w_res) rises on shortcut-free prompts, the bypass is cue-conditional.

SELF-TEST (no torch / no repo):  python 90_forced_mode_alignment.py --self_test
"""

from __future__ import annotations

import argparse
import csv as _csv
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("forced_mode")

FAMILY_MAP = {
    "charge_Z": ["daughter_z_minus2", "daughter_z_plus1", "emitted_charge_minus1", "emitted_charge_plus2",
                 "element_shift_minus2", "element_shift_plus1"],
    "mass_A": ["daughter_a_minus4", "daughter_a_unchanged", "emitted_mass4", "emitted_mass_negligible"],
    "quark_weak": ["quark_flavour_change", "quark_change_z_plus1", "no_quark_flavour_change",
                   "w_boson_mediation", "not_weak_force", "daughter_n_minus1"],
    "lepton": ["lepton_number_increases", "antineutrino_emitted", "lightest_charged_elementary"],
    "energy": ["continuous_energy_spectrum", "discrete_energy_spectrum"],
    "emission": ["emitted_2neutrons", "emitted_2protons", "cluster_ejection", "cluster_ejection_daughter",
                 "cluster_no_creation", "new_particles_created", "no_new_particles"],
}
CUE2FAM = {c: f for f, m in FAMILY_MAP.items() for c in m}


# =====================================================================
# Pure-numpy core (exercised by --self_test)
# =====================================================================
def unit_raw(v):
    n = np.linalg.norm(v); return v / n if n > 1e-30 else v


def cosine(a, b):
    return float(np.dot(unit_raw(np.asarray(a, float)), unit_raw(np.asarray(b, float))))


def fisher_axis(H, y, shrink=0.1):
    mu0, mu1 = H[y == 0].mean(0), H[y == 1].mean(0)
    X0, X1 = H[y == 0] - mu0, H[y == 1] - mu1
    Sw = (X0.T @ X0 + X1.T @ X1) / max(H.shape[0] - 2, 1)
    Sw = 0.5 * (Sw + Sw.T); Sw = (1 - shrink) * Sw + shrink * np.diag(np.diag(Sw))
    Sw += 1e-6 * float(np.mean(np.diag(Sw)) + 1e-12) * np.eye(Sw.shape[0])
    return unit_raw(np.linalg.solve(Sw, mu1 - mu0))


def auc_scalar(s, y):
    s = np.asarray(s, float); o = np.argsort(s); r = np.empty_like(o, float); r[o] = np.arange(1, len(s) + 1)
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 * n0 else float("nan")


def build_forced_prompt(target_prompt, ex_alpha, ex_beta, suffix, flip_order=False):
    """Balanced two-shot template; exemplars must come from other families than the target."""
    a_block = f"{ex_alpha}{suffix} alpha"
    b_block = f"{ex_beta}{suffix} beta"
    first, second = (b_block, a_block) if flip_order else (a_block, b_block)
    return f"{first}\n\n{second}\n\n{target_prompt}{suffix}"


def pick_exemplars(prompts, train_fams, target_family, rng):
    pool_a = [p for p in prompts if p["surface_family"] in train_fams
              and p["surface_family"] != target_family and p["correct_answer"].strip() == "alpha"]
    pool_b = [p for p in prompts if p["surface_family"] in train_fams
              and p["surface_family"] != target_family and p["correct_answer"].strip() == "beta"]
    return pool_a[int(rng.integers(len(pool_a)))]["prompt"], pool_b[int(rng.integers(len(pool_b)))]["prompt"]


def intact_flag(top1_id, alpha_id, beta_id):
    return int(top1_id in (alpha_id, beta_id))


def subset_mean_direction(G, keep_mask):
    G = np.asarray(G, np.float64)
    return unit_raw(G[keep_mask].mean(0))


# =====================================================================
# Self-test
# =====================================================================
def self_test():
    rng = np.random.default_rng(0)
    prompts = [{"prompt": f"p{i}", "surface_family": f"f{i % 5}",
                "correct_answer": "alpha" if i % 2 == 0 else "beta"} for i in range(20)]
    train_fams = {"f0", "f1", "f2"}
    ea, eb = pick_exemplars(prompts, train_fams, "f3", rng)
    fp = build_forced_prompt("TARGET", ea, eb, "\nAnswer (alpha or beta):")
    assert "TARGET" in fp and fp.count("Answer (alpha or beta):") == 3
    assert " alpha" in fp and " beta" in fp, "both exemplar answers must appear"
    assert fp.rstrip().endswith("Answer (alpha or beta):"), "target slot must be last and unanswered"
    fp2 = build_forced_prompt("TARGET", ea, eb, "\nA:", flip_order=True)
    assert fp2.index(" beta") < fp2.index(" alpha"), "flip_order must swap exemplar order"
    assert intact_flag(7, 7, 9) == 1 and intact_flag(8, 7, 9) == 0
    # rotation logic: forced usage closer to w than base usage
    d = 16; w = unit_raw(rng.standard_normal(d))
    g0 = rng.standard_normal(d)
    u_base = unit_raw(g0 - (g0 @ w) * w)
    u_forced = unit_raw(0.8 * w + 0.3 * u_base)
    assert abs(cosine(u_base, w)) < 0.5 and cosine(u_forced, w) > 0.7
    G = np.vstack([np.tile(w, (5, 1)), np.tile(u_base, (5, 1))])
    mask = np.array([True] * 5 + [False] * 5)
    assert cosine(subset_mean_direction(G, mask), w) > 0.99, "subset mean direction"
    assert CUE2FAM["antineutrino_emitted"] == "lepton"
    print("[self_test] OK — few-shot builder (balanced, leak-free, order-flippable), intact flag, rotation and subset logic pass.")


# =====================================================================
# Real run
# =====================================================================
def _chain(o, p):
    for x in p.split("."):
        o = getattr(o, x)
    return o


def run_real(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.float32, low_cpu_mem_usage=True, trust_remote_code=True,
    ).to(args.device).eval()
    blocks = _chain(model, "model.layers"); n_layers = len(blocks); norm_mod = _chain(model, "model.norm")
    d = model.config.hidden_size; last = n_layers - 1
    alpha_id = tok.encode(args.alpha_answer, add_special_tokens=False)[0]
    beta_id = tok.encode(args.beta_answer, add_special_tokens=False)[0]
    layers = sorted({L for L in (args.layers or list(range(n_layers))) if 0 <= L < n_layers})
    logger.info("model: %d layers; forced-mode alignment over %d taps", n_layers, len(layers))

    prompts = [json.loads(l) for l in open(args.prompts)]
    fams = sorted({p["surface_family"] for p in prompts}); rng.shuffle(fams)
    train_fams = set(fams[: int(round(len(fams) * args.train_frac))])
    nP = len(prompts)
    y = np.array([1 if p["correct_answer"].strip() == "beta" else 0 for p in prompts])
    trm = np.array([p["surface_family"] in train_fams for p in prompts])

    # cue families (field in jsonl, else join by row order with a csv)
    cue_fam = [None] * nP
    if "cue_type" in prompts[0]:
        cue_fam = [CUE2FAM.get(p.get("cue_type", ""), None) for p in prompts]
    elif args.cue_csv:
        try:
            with open(args.cue_csv) as f:
                rd = list(_csv.DictReader(f))
            for r in rd:
                i = int(r["prompt_idx"])
                if i < nP:
                    cue_fam[i] = CUE2FAM.get(r["cue_type"], None)
            logger.info("cue families joined from %s (%d rows)", args.cue_csv, len(rd))
        except Exception as e:
            logger.warning("cue csv join failed (%s) -> part C skipped", e)

    forced_text = {}
    for i, p in enumerate(prompts):
        ea, eb = pick_exemplars(prompts, train_fams, p["surface_family"], rng)
        forced_text[i] = build_forced_prompt(p["prompt"], ea, eb, args.suffix, flip_order=bool(i % 2))

    def tap(L):
        return blocks[L + 1] if L < last else norm_mod

    def capture(texts, label):
        res = {L: np.zeros((nP, d), np.float32) for L in layers}
        grad = {L: np.zeros((nP, d), np.float32) for L in layers}
        cm = np.zeros(nP); intact = np.zeros(nP, int)
        logger.info("capturing %s regime (%d prompts)...", label, nP)
        for p_ in model.parameters():
            p_.requires_grad_(True)
        for i in range(nP):
            inp = tok([texts[i]], return_tensors="pt").to(args.device)
            keep = {}; handles = []
            for L in layers:
                def mk(L=L):
                    def pre(m, a):
                        a[0].retain_grad(); keep[L] = a[0]; return None
                    return pre
                handles.append(tap(L).register_forward_pre_hook(mk(), with_kwargs=False))
            try:
                row = model(**inp, use_cache=False).logits[0, -1, :]
                (row[beta_id] - row[alpha_id]).backward()
                for L in layers:
                    t = keep[L]
                    res[L][i] = t.detach()[0, -1, :].float().cpu().numpy()
                    grad[L][i] = t.grad[0, -1, :].float().cpu().numpy() if t.grad is not None else 0.0
                lp = torch.log_softmax(row.detach().float(), 0)
                cm[i] = float(lp[beta_id] - lp[alpha_id])
                intact[i] = intact_flag(int(row.argmax().item()), alpha_id, beta_id)
            finally:
                for h in handles:
                    h.remove()
            model.zero_grad(set_to_none=True)
            if (i + 1) % 100 == 0:
                logger.info("  %s capture %d/%d", label, i + 1, nP)
        return res, grad, cm, intact

    base_texts = {i: p["prompt"] for i, p in enumerate(prompts)}
    res_b, grad_b, cm_b, int_b = capture(base_texts, "BASE")
    res_f, grad_f, cm_f, int_f = capture(forced_text, "FORCED")
    acc_b = float(np.mean(((cm_b[~trm] > 0).astype(int)) == y[~trm]))
    acc_f = float(np.mean(((cm_f[~trm] > 0).astype(int)) == y[~trm]))
    logger.info("clean: base intact-rate=%.3f acc=%.3f | FORCED intact-rate=%.3f acc=%.3f",
                int_b.mean(), acc_b, int_f.mean(), acc_f)
    if int_f.mean() < 0.3:
        logger.warning("forced regime intact-rate < 0.3 -> top-1 still not an answer token; intact metrics weak, margin metrics remain valid")

    # ---------- (A) geometry per layer ----------
    geo = []
    for L in layers:
        Hb = res_b[L].astype(np.float64); Hf = res_f[L].astype(np.float64)
        wb = fisher_axis(Hb[trm], y[trm], args.shrink); wf = fisher_axis(Hf[trm], y[trm], args.shrink)
        ub = unit_raw(grad_b[L].astype(np.float64).mean(0)); uf = unit_raw(grad_f[L].astype(np.float64).mean(0))
        rec = {"layer": int(L),
               "cos_u_wres_base": cosine(ub, wb), "cos_u_wres_forced": cosine(uf, wf),
               "cos_uf_wres_base": cosine(uf, wb), "cos_uf_ub": cosine(uf, ub),
               "cos_wf_wb": cosine(wf, wb),
               "auc_along_u_forced": auc_scalar(Hf[~trm] @ uf, y[~trm]),
               "auc_wres_forced": auc_scalar(Hf[~trm] @ wf, y[~trm])}
        geo.append(rec)
        if (L % 4 == 0) or (L == layers[-1]):
            logger.info("  L%d: cos(u,w_res) base=%+.3f -> FORCED=%+.3f | cos(u_f,u_b)=%.3f | AUC along u_f=%.3f",
                        L, rec["cos_u_wres_base"], rec["cos_u_wres_forced"], rec["cos_uf_ub"], rec["auc_along_u_forced"])

    # ---------- (B) intact steering in the forced regime ----------
    held = [i for i in range(nP) if not trm[i]]
    ta = [i for i in held if y[i] == 0][: args.max_targets]
    tb = [i for i in held if y[i] == 1][: args.max_targets]
    targets = [(i, "beta") for i in ta] + [(i, "alpha") for i in tb]

    def run_steer(i, L, delta):
        inp = tok([forced_text[i]], return_tensors="pt").to(args.device)
        dt = torch.tensor(delta, dtype=torch.float32, device=args.device)
        def pre(m, a):
            hs = a[0].clone(); hs[0, -1, :] = hs[0, -1, :] + dt; return (hs,)
        h = tap(L).register_forward_pre_hook(pre, with_kwargs=False)
        try:
            with torch.no_grad():
                row = model(**inp, use_cache=False).logits[0, -1, :].float()
            lp = torch.log_softmax(row, 0)
            return float(lp[beta_id] - lp[alpha_id]), int(row.argmax().item())
        finally:
            h.remove()

    steer_rows = []
    for L in [L for L in args.steer_layers if L in layers]:
        Hf = res_f[L].astype(np.float64)
        wf = fisher_axis(Hf[trm], y[trm], args.shrink)
        uf = unit_raw(grad_f[L].astype(np.float64).mean(0))
        sig = float(np.std(Hf[trm] @ wf))
        dirs = {"w_res_forced": wf, "usage_forced": uf}
        for r in range(args.n_random):
            dirs[f"random{r}"] = unit_raw(rng.standard_normal(d))
            dirs[f"shuffled{r}"] = fisher_axis(Hf[trm], rng.permutation(y[trm]), args.shrink)
        for c in args.c_grid:
            for name, v in dirs.items():
                fl_i, fl_m = [], []
                for i, toward in targets:
                    s = +1.0 if toward == "beta" else -1.0
                    m1, t1 = run_steer(i, L, (s * c * sig) * unit_raw(v))
                    target_id = beta_id if toward == "beta" else alpha_id
                    fl_i.append(int(t1 == target_id and int_f[i] == 1 and
                                    ((cm_f[i] < 0) if toward == "beta" else (cm_f[i] > 0))))
                    fl_m.append(int((cm_f[i] < 0 and m1 > 0) if toward == "beta" else (cm_f[i] > 0 and m1 < 0)))
                steer_rows.append({"layer": int(L), "c": float(c), "dir": name,
                                   "intact_flip": float(np.mean(fl_i)), "margin_flip": float(np.mean(fl_m))})
        for name in ["w_res_forced", "usage_forced"]:
            best = max([r for r in steer_rows if r["layer"] == L and r["dir"] == name], key=lambda r: r["intact_flip"])
            nulls = [r["intact_flip"] for r in steer_rows if r["layer"] == L and ("random" in r["dir"] or "shuffled" in r["dir"])]
            logger.info("  L%d steer %s: max intact-flip=%.3f (null max=%.3f) | margin-flip=%.3f",
                        L, name, best["intact_flip"], max(nulls) if nulls else float("nan"), best["margin_flip"])

    # ---------- (C) cue-subset alignment (base regime, offline) ----------
    cue_rows = []
    if any(f is not None for f in cue_fam):
        keep = np.array([f is not None and f not in set(args.exclude_families) for f in cue_fam])
        logger.info("(C) cue-subset: keeping %d/%d prompts (excluding %s)", int(keep.sum()), nP, args.exclude_families)
        for L in layers:
            Hb = res_b[L].astype(np.float64)
            wb = fisher_axis(Hb[trm], y[trm], args.shrink)
            u_sub = subset_mean_direction(grad_b[L], keep)
            u_all = unit_raw(grad_b[L].astype(np.float64).mean(0))
            cue_rows.append({"layer": int(L), "cos_usub_wres": cosine(u_sub, wb),
                             "cos_uall_wres": cosine(u_all, wb)})
    else:
        logger.warning("(C) no cue families available (no cue_type field and no --cue_csv) -> skipped")

    # ---------- save + verdict ----------
    def wcsv(name, rws):
        if not rws:
            return
        with open(out / name, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=list(rws[0].keys())); w.writeheader(); [w.writerow(r) for r in rws]
    wcsv("forced_geometry.csv", geo); wcsv("forced_steering.csv", steer_rows); wcsv("cue_subset_alignment.csv", cue_rows)
    json.dump({"clean_intact_base": float(int_b.mean()), "clean_intact_forced": float(int_f.mean()),
               "acc_base": acc_b, "acc_forced": acc_f}, open(out / "forced_summary.json", "w"), indent=2)

    mb = float(np.median([abs(r["cos_u_wres_base"]) for r in geo]))
    mf = float(np.median([abs(r["cos_u_wres_forced"]) for r in geo]))
    print("\n" + "=" * 96)
    print("FORCED-MODE ALIGNMENT -- does usage rotate toward the readable axis when the shortcut is closed?")
    print("=" * 96)
    print(f"clean intact-rate: base={int_b.mean():.3f} -> forced={int_f.mean():.3f} "
          f"({'ceiling removed' if int_f.mean() > 0.5 else 'ceiling NOT removed; margin metrics only'}) | acc base={acc_b:.3f} forced={acc_f:.3f}")
    print(f"USAGE ROTATION: median |cos(u, w_res)| base={mb:.3f} -> forced={mf:.3f} "
          f"({'ROTATES toward the axis -> causality is task-regime-conditional (H2 proven constructively)' if mf > mb + 0.15 else 'no rotation -> representation stays unused even under forcing'})")
    if cue_rows:
        ms = float(np.median([abs(r["cos_usub_wres"]) for r in cue_rows]))
        ma = float(np.median([abs(r["cos_uall_wres"]) for r in cue_rows]))
        print(f"CUE-SUBSET (base, shortcut-free prompts): median |cos(u_subset, w_res)| = {ms:.3f} (all prompts {ma:.3f})")
    print("=" * 96 + "\n")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self_test", action="store_true")
    p.add_argument("--prompts", default="physics_decay_type_probe_v2_train.jsonl")
    p.add_argument("--out_dir", default="data/analysis/runD_v2/forced_mode")
    p.add_argument("--model_name", default="Qwen/Qwen3-4B")
    p.add_argument("--device", default="cuda")
    p.add_argument("--alpha_answer", default=" alpha")
    p.add_argument("--beta_answer", default=" beta")
    p.add_argument("--suffix", default="\nAnswer (alpha or beta):")
    p.add_argument("--layers", type=int, nargs="*", default=None, help="default = ALL layers")
    p.add_argument("--steer_layers", type=int, nargs="*", default=[16, 21, 24, 35])
    p.add_argument("--c_grid", type=float, nargs="*", default=[1, 4, 16])
    p.add_argument("--n_random", type=int, default=3)
    p.add_argument("--max_targets", type=int, default=40)
    p.add_argument("--cue_csv", default="", help="csv with prompt_idx,cue_type to join cue families")
    p.add_argument("--exclude_families", nargs="*", default=["mass_A", "emission"])
    p.add_argument("--train_frac", type=float, default=0.6)
    p.add_argument("--shrink", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    return p


def main():
    args = build_parser().parse_args()
    if args.self_test:
        self_test(); return
    run_real(args)


if __name__ == "__main__":
    main()
