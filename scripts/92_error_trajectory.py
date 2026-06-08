"""
92_error_trajectory.py    [WHERE does the wrong alpha/beta answer form, across layers?]
=========================================================================================
Motivation. We want to suppress the surface 'crutch' (nucleon->alpha habit) at the layer where it
decides the answer -- but we have never measured WHERE the wrong answer forms on the failed-beta
prompts. "Concept is written ~L21" (exp 80) is NOT the same as "the error commits at L21". This
script measures the commitment layer directly, and is a prerequisite/diagnostic for any healing:

  - if the failed/succeeded trajectories diverge EARLY/MID, there are downstream layers in which a
    re-awakened concept could surface -> suppression has room to work.
  - if divergence is a LATE readout phenomenon, mid-layer suppression is pointless and the result is
    already predicted to be "concept not recoverable".

Method (logit-lens trajectory + concept check), all at the answer position:
  For every layer L (hidden_states[L], i.e. residual after L blocks):
    * lens margin  m_lens[L] = logit_alpha - logit_beta  via  lm_head(final_norm(h_L))   (>0 => alpha)
    * concept proj p_wres[L] = <h_L, w_res[L]>   where w_res[L] = Fisher(train, TRUE labels), signed +beta
  Groups (by the model's OWN final prediction, recomputed here):
    * failed_beta    : true beta, model says alpha   (the errors; mostly nucleon-framed)
    * succeeded_beta : true beta, model says beta     (mostly lepton-framed)
    * correct_alpha  : true alpha, model says alpha   (reference)
  Reported per layer:
    * mean lens margin of each group  -> the trajectories (where red=failed and green=succ split)
    * AUC_lens(failed vs succ)        -> divergence of the ANSWER (departs 0.5 at the commitment layer)
    * AUC_wres(failed vs succ)        -> does the CONCEPT distinguish them? (should stay ~0.5: both beta)
    * held-out AUC_wres(true alpha vs beta) -> concept decodes correctly at every layer (~0.99)
  Verdict locates the commitment layer and states whether the concept stays correct throughout
  (i.e. the error is a readout-path phenomenon, not a representational one).

Honest caveat: logit-lens is unreliable in absolute terms at early layers on a base model. We only
use it for a BETWEEN-GROUP contrast through the SAME lens, so the lens bias cancels in the difference
and the divergence layer is robust even if early absolute numbers are off.

SELF-TEST (no torch / no repo):  python 92_error_trajectory.py --self_test
"""

from __future__ import annotations
import argparse, csv as _csv, json, logging, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("err_traj")


# ---------------- shared numpy core (exercised by --self_test) ----------------
def unit_raw(v):
    n = np.linalg.norm(v); return v / n if n > 1e-30 else v


def fisher_axis(H, y, shrink=0.1):
    """Fisher/LDA axis, signed so that higher projection => class 1 (beta)."""
    mu0, mu1 = H[y == 0].mean(0), H[y == 1].mean(0)
    X0, X1 = H[y == 0] - mu0, H[y == 1] - mu1
    Sw = (X0.T @ X0 + X1.T @ X1) / max(H.shape[0] - 2, 1)
    Sw = 0.5 * (Sw + Sw.T); Sw = (1 - shrink) * Sw + shrink * np.diag(np.diag(Sw))
    Sw += 1e-6 * float(np.mean(np.diag(Sw)) + 1e-12) * np.eye(Sw.shape[0])
    w = np.linalg.solve(Sw, mu1 - mu0)
    return unit_raw(w)


def auc_scalar(s, y):
    s = np.asarray(s, float); y = np.asarray(y)
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    o = np.argsort(s); r = np.empty(len(s), float); r[o] = np.arange(1, len(s) + 1)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def divergence_onset(auc_curve, thr=0.70):
    """First layer where the answer-separation |AUC-0.5| crosses (thr-0.5) and stays. Sign-robust:
    AUC may move away from 0.5 in either direction depending on the arbitrary label assignment."""
    sep = np.abs(np.asarray(auc_curve, float) - 0.5)
    need = thr - 0.5
    for L in range(len(sep)):
        if sep[L] >= need and np.all(sep[L:] >= need - 0.05):
            return L
    return int(np.argmax(sep))  # fallback: peak separation


# ---------------- self-test ----------------
def self_test():
    rng = np.random.default_rng(0)
    nlayer, d, n = 12, 8, 60
    # unembedding: alpha row, beta row; lens margin = h.(U_a - U_b)
    Ua, Ub = unit_raw(rng.standard_normal(d)), None
    Ub = unit_raw(rng.standard_normal(d)); axis = unit_raw(Ua - Ub)
    # three groups; true label: beta=1 for the two beta groups, 0 for alpha group
    y_true = np.array([1] * 20 + [1] * 20 + [0] * 20)            # failedβ, succβ, correctα
    grp = np.array([0] * 20 + [1] * 20 + [2] * 20)
    H = np.zeros((nlayer, n, d))
    diverge_at = 6
    for L in range(nlayer):
        base = 0.3 * rng.standard_normal((n, d))
        # concept signal present for BOTH beta groups at all layers (beta => -axis, alpha => +axis)
        concept = np.where((grp == 0) | (grp == 1), -1.0, 1.0)[:, None] * axis[None, :] * 0.8
        # readout/answer signal after diverge_at: failedβ drifts to alpha(+axis), succβ to beta(-axis)
        ans = np.zeros((n, d))
        if L >= diverge_at:
            frac = (L - diverge_at + 1) / (nlayer - diverge_at)
            ans += (grp == 0)[:, None] * (axis)[None, :] * 1.5 * frac    # failedβ -> alpha side
            ans += (grp == 1)[:, None] * (-axis)[None, :] * 1.5 * frac   # succβ   -> beta side
            ans += (grp == 2)[:, None] * (axis)[None, :] * 1.5 * frac    # correctα-> alpha side
        H[L] = base + concept + ans
    # lens margin (alpha - beta) = h.(Ua-Ub) = h.axis ; alpha when negative-of-beta => sign:
    m_lens = np.einsum("lnd,d->ln", H, axis)   # >0 means toward (Ua-Ub) i.e. alpha-ish
    failed = grp == 0; succ = grp == 1
    auc_lens = [auc_scalar(m_lens[L], (failed.astype(int))[ (failed|succ) ] if False else None) for L in range(nlayer)]
    # proper: AUC separating failed(1) vs succ(0) using lens margin, on failed|succ subset
    idx = np.where(failed | succ)[0]; lab = failed[idx].astype(int)
    auc_lens = [auc_scalar(m_lens[L][idx], lab) for L in range(nlayer)]
    onset = divergence_onset(auc_lens)
    assert abs(onset - diverge_at) <= 1, f"divergence onset {onset} != planted {diverge_at}"
    # concept axis: Fisher on TRUE labels should decode beta vs alpha well at every layer
    w = fisher_axis(H[nlayer - 1], y_true)
    proj = H[nlayer - 1] @ w
    assert auc_scalar(proj, y_true) > 0.9, "concept axis must decode true label"
    # concept should NOT separate failedβ from succβ (both beta): AUC ~ 0.5
    wmid = fisher_axis(H[4], y_true); pj = H[4][idx] @ wmid
    assert auc_scalar(pj, lab) < 0.8, "concept should barely separate failed vs succ beta"
    print(f"[self_test] OK — divergence onset detected at L{onset} (planted {diverge_at}); "
          f"concept decodes true label; concept does not split failed/succ. ")


# ---------------- real run ----------------
def run_real(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.float32, low_cpu_mem_usage=True, trust_remote_code=True,
    ).to(args.device).eval()
    base = model.model
    norm, head = base.norm, model.lm_head
    n_layers = len(base.layers); d = model.config.hidden_size
    a_id = tok.encode(args.alpha_answer, add_special_tokens=False)[0]
    b_id = tok.encode(args.beta_answer, add_special_tokens=False)[0]
    logger.info("model %d layers, d=%d; alpha_id=%d beta_id=%d", n_layers, d, a_id, b_id)

    prompts = [json.loads(l) for l in open(args.prompts)]
    nP = len(prompts)
    y_true = np.array([1 if p["correct_answer"].strip() == "beta" else 0 for p in prompts])  # beta=1
    fams = sorted({p.get("surface_family", str(i)) for i, p in enumerate(prompts)})
    rng.shuffle(fams); train_fams = set(fams[: int(round(len(fams) * args.train_frac))])
    is_train = np.array([p.get("surface_family", "") in train_fams for p in prompts])

    def S(x):
        return str(x) if x is not None else "NA"

    def framing(p):
        rt = S(p.get("relation_type")).lower(); cr = S(p.get("concept_route")).lower()
        if any(k in rt for k in ["neutron_to_proton", "n_to_p", "quark", "antineutrino", "full_beta"]):
            return "nucleon"
        if any(k in cr for k in ["lepton", "electron", "muon"]) or "weak_force" in rt:
            return "lepton"
        return "other"

    # capture hidden states at the answer position for all prompts
    nLp1 = n_layers + 1
    H = np.zeros((nLp1, nP, d), np.float32)        # H[L][i] = residual after L blocks (L=0 => embeddings)
    m_lens = np.zeros((nLp1, nP), np.float32)      # logit-lens margin alpha - beta
    final_pred_alpha = np.zeros(nP, bool)
    bs = args.batch_size
    logger.info("Capturing hidden-state trajectories over %d prompts (bs=%d)...", nP, bs)
    with torch.no_grad():
        for s in range(0, nP, bs):
            chunk = [p["prompt"] for p in prompts[s: s + bs]]
            enc = tok(chunk, return_tensors="pt", padding=True).to(args.device)
            o = model(**enc, output_hidden_states=True, use_cache=False)
            hs = o.hidden_states                      # tuple len nLayers+1, each (b, seq, d)
            real_logits = o.logits[:, -1, :]          # (b, vocab)
            for L in range(nLp1):
                h_last = hs[L][:, -1, :]              # (b, d) — left padded so -1 is the answer pos
                H[L, s: s + len(chunk)] = h_last.float().cpu().numpy()
                ll = head(norm(h_last))               # logit-lens at layer L
                m_lens[L, s: s + len(chunk)] = (ll[:, a_id] - ll[:, b_id]).float().cpu().numpy()
            fp = (real_logits[:, a_id] > real_logits[:, b_id]).cpu().numpy()
            final_pred_alpha[s: s + len(chunk)] = fp
            if (s // bs) % 5 == 0:
                logger.info("  %d/%d", s + len(chunk), nP)

    # groups
    failed_beta = (y_true == 1) & (final_pred_alpha)        # true beta, predicted alpha (errors)
    succ_beta = (y_true == 1) & (~final_pred_alpha)         # true beta, predicted beta
    correct_alpha = (y_true == 0) & (final_pred_alpha)      # reference
    acc_beta = succ_beta.sum() / max((y_true == 1).sum(), 1)
    logger.info("groups: failed_beta=%d  succ_beta=%d (beta-recall %.3f)  correct_alpha=%d",
                failed_beta.sum(), succ_beta.sum(), acc_beta, correct_alpha.sum())
    fr = np.array([framing(p) for p in prompts])
    logger.info("framing among failed_beta: nucleon=%d lepton=%d other=%d | among succ_beta: nucleon=%d lepton=%d other=%d",
                ((fr == "nucleon") & failed_beta).sum(), ((fr == "lepton") & failed_beta).sum(), ((fr == "other") & failed_beta).sum(),
                ((fr == "nucleon") & succ_beta).sum(), ((fr == "lepton") & succ_beta).sum(), ((fr == "other") & succ_beta).sum())

    idx_fs = np.where(failed_beta | succ_beta)[0]
    lab_fs = failed_beta[idx_fs].astype(int)                # 1 = failed (alpha), 0 = succ (beta)

    # per-layer table
    rows = []
    auc_lens_curve = []
    for L in range(nLp1):
        w_res = fisher_axis(H[L][is_train].astype(np.float64), y_true[is_train], args.shrink)
        proj = H[L] @ w_res                                  # higher => beta
        auc_lens = auc_scalar(m_lens[L][idx_fs], lab_fs)     # answer divergence (1=failed/alpha)
        auc_wres_fs = auc_scalar(proj[idx_fs], lab_fs)       # does concept split failed vs succ?
        auc_wres_true = auc_scalar(proj[~is_train], y_true[~is_train])  # held-out concept decodability
        auc_lens_curve.append(auc_lens)
        rows.append(dict(
            layer=L,
            lens_failedbeta=float(m_lens[L][failed_beta].mean()),
            lens_succbeta=float(m_lens[L][succ_beta].mean()),
            lens_correctalpha=float(m_lens[L][correct_alpha].mean()) if correct_alpha.any() else float("nan"),
            lens_gap_failed_minus_succ=float(m_lens[L][failed_beta].mean() - m_lens[L][succ_beta].mean()),
            wres_failedbeta=float(proj[failed_beta].mean()),
            wres_succbeta=float(proj[succ_beta].mean()),
            auc_lens_failed_vs_succ=auc_lens,
            auc_wres_failed_vs_succ=auc_wres_fs,
            auc_wres_truelabel_heldout=auc_wres_true,
        ))

    onset = divergence_onset(auc_lens_curve, args.divergence_thr)
    # where does failed_beta lens margin cross 0 (commit to alpha)?
    lf = np.array([r["lens_failedbeta"] for r in rows])
    cross = next((L for L in range(1, nLp1) if lf[L - 1] <= 0 < lf[L]), None)

    with open(out / "error_trajectory.csv", "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); [w.writerow(r) for r in rows]

    print("\n" + "=" * 100)
    print("ERROR-FORMATION TRAJECTORY — where does the wrong alpha/beta answer commit?")
    print("=" * 100)
    print(f"{'L':>3} | {'lens failedβ':>12} {'lens succβ':>11} {'lens α(ref)':>11} | "
          f"{'gap f-s':>8} | {'AUC_lens f|s':>12} | {'AUC_wres f|s':>12} | {'AUC_wres true':>13}")
    for r in rows:
        mark = "  <-- diverge" if r["layer"] == onset else ("  (α-cross)" if r["layer"] == cross else "")
        print(f"{r['layer']:>3} | {r['lens_failedbeta']:>12.3f} {r['lens_succbeta']:>11.3f} {r['lens_correctalpha']:>11.3f} | "
              f"{r['lens_gap_failed_minus_succ']:>8.3f} | {r['auc_lens_failed_vs_succ']:>12.3f} | "
              f"{r['auc_wres_failed_vs_succ']:>12.3f} | {r['auc_wres_truelabel_heldout']:>13.3f}{mark}")
    print("-" * 100)
    print(f"Divergence onset (answer commits) at layer L{onset}  (AUC_lens(failed vs succ) crosses {args.divergence_thr}).")
    if cross is not None:
        print(f"failed-beta running lens-margin crosses 0 into alpha at layer L{cross}.")
    mid = onset <= 0.7 * n_layers
    conc_ok = np.mean([r["auc_wres_truelabel_heldout"] for r in rows[max(1, onset):]]) > 0.85
    conc_blind_to_fs = np.mean([r["auc_wres_failed_vs_succ"] for r in rows]) < 0.65
    print(f"Concept (w_res) decodes the TRUE label across layers: {'YES (~'+format(np.mean([r['auc_wres_truelabel_heldout'] for r in rows[1:]]),'.2f')+')' if conc_ok else 'patchy'}.")
    print(f"Concept distinguishes failed-β from succ-β: {'NO — both read as beta (AUC~%.2f) => error is NOT in the concept' % np.mean([r['auc_wres_failed_vs_succ'] for r in rows]) if conc_blind_to_fs else 'partially'}.")
    if mid:
        print(f"=> Commitment is EARLY/MID (L{onset} <= {int(0.7*n_layers)}): there ARE downstream layers for a re-awakened "
              f"concept to surface -> mid-layer suppression has room to work. Suppress around L{max(1,onset-2)}-L{onset}.")
    else:
        print(f"=> Commitment is LATE (L{onset} > {int(0.7*n_layers)}): little downstream room; mid-layer suppression is "
              f"unlikely to help and the healing result is largely pre-determined as 'not recoverable'.")
    print("Caveat: logit-lens absolute values are unreliable early on a base model; we use only the BETWEEN-GROUP "
          "contrast through the same lens, so lens bias cancels and the divergence layer is robust.")
    print("=" * 100 + "\n")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self_test", action="store_true")
    p.add_argument("--prompts", default="physics_decay_type_probe_v2_train.jsonl")
    p.add_argument("--out_dir", default="data/analysis/runD_v2/error_trajectory")
    p.add_argument("--model_name", default="Qwen/Qwen3-4B")
    p.add_argument("--device", default="cuda")
    p.add_argument("--alpha_answer", default=" alpha")
    p.add_argument("--beta_answer", default=" beta")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--train_frac", type=float, default=0.6)
    p.add_argument("--shrink", type=float, default=0.1)
    p.add_argument("--divergence_thr", type=float, default=0.70)
    p.add_argument("--seed", type=int, default=0)
    return p


def main():
    args = build_parser().parse_args()
    if args.self_test:
        self_test(); return
    run_real(args)


if __name__ == "__main__":
    main()
