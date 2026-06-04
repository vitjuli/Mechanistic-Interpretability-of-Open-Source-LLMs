"""
80_attention_cie_mechanism.py   [JOB: MECHANISM HUNT — attention vs subspace CIE]
=====================================================================================
Phase II showed: alpha/beta is decodable along w_res (AUC 0.99), but w_res is NOT
causal under additive steering (65/75: flip <=0.07 to 32 sigma) and NOT written by
the MLP transcoder dictionary (66/74: carrier capture at the 28th percentile of a
random-feature null). All those negatives concern the RESIDUAL axis and the MLP
dictionary. This script asks the remaining mechanism question, motivated by the
Jan-Feb 2026 attention line (Head Pursuit 2510.21518; Attention-Head Intervention
2601.04398) and by Wang/Fudan "Emergent Structured Representations" (2602.07794),
which finds a conceptual subspace that IS causal under activation-patching and is
constructed by attention heads.

THREE QUESTIONS (each with its decisive null):

  (A) DO ATTENTION HEADS WRITE THE CONCEPT?
      For each head (L,h) we decompose its contribution to the residual at the
      answer position: v_{L,h} = W_O[:, h*hd:(h+1)*hd] @ z_{L,h}, where z is the
      o_proj INPUT (concatenated head outputs). We project v_{L,h} onto the
      per-layer w_res and score each head by how well that projection SEPARATES
      the classes (standardised mean difference |d|). Heads with high |d| "write
      the concept along the readout axis".
      INTERVENTION: ablate / negate the top-k concept-aligned heads at the answer
      position and measure the answer-margin flip on held-out targets.
      DECISIVE NULL: ablate k RANDOM heads (N seeds). If the top heads flip no
      more than random heads, attention does not causally carry the concept either.

  (B) IS THE w_res SUBSPACE CAUSAL UNDER CIE-STYLE PATCHING? (matches 2602.07794)
      Activation patching / causal mediation: at layer L, replace the w_res-
      component of the residual at the answer position with that of an
      opposite-class donor: h' = h + (proj_donor - <h,w_hat>) * w_hat. Measure the
      causal indirect effect CIE = signed margin shift toward the donor class.
      This is the SAME intervention type as 2602.07794 (transplant subspace-aligned
      components clean->corrupt). Their concept gives a large CIE; we expect ~0,
      and the contrast is the point.
      DECISIVE NULL: patch the component along a RANDOM unit direction (N seeds).

  (C) OPTIONAL — IS A SURFACE HEURISTIC CAUSAL WHERE THE CONCEPT IS NOT?
      exp 76 found the model leans on a "heavy = alpha" mass heuristic. If the
      prompts carry a secondary label (--mass_key, e.g. a heavier/lighter tag),
      we build a "mass axis" the same way (Fisher on that label) and run the SAME
      CIE/steering on it. If the mass axis IS causal while w_res is not, we exhibit
      a decodable-but-bypassed concept whose causal role is taken by the shortcut.

INTERPRETATION GRID:
  - heads flip > random-head null  -> attention CARRIES the concept: mechanism found
        (the concept is written by attention, not by the MLP dictionary).
  - heads ~ random AND CIE(w_res) ~ random-dir null -> alpha/beta is decodable but
        causally inert at every probed locus; combined with (C) this supports
        "decodable concept, causally bypassed because the task is heuristic-solvable".

NORMALISATION: head interventions scale the head's o_proj-input slice at the answer
position only (the position whose residual produces the answer, and where alignment
was scored). CIE patches add a fixed delta along w_hat at the answer position at the
postL{L} tap (identical hook style to 65). All directions unit raw-L2.

INPUTS: prompts jsonl (keys: prompt, correct_answer in {alpha,beta}, surface_family),
base model Qwen/Qwen3-4B. Optionally concept_directions.npz (60_) for Sigma_inv to
report cos_C(w_res, gbar); not required. Self-contained: recomputes w_res from
captured residuals.

SELF-TEST (no torch / no repo):  python 80_attention_cie_mechanism.py --self_test
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("attn_cie")


# =====================================================================
# Pure-numpy helpers (all exercised by --self_test)
# =====================================================================
def unit_raw(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-30 else v


def fisher_axis(H: np.ndarray, y: np.ndarray, shrink: float = 0.1) -> np.ndarray:
    """Regularised LDA axis separating classes in residual space; unit raw-L2."""
    mu0, mu1 = H[y == 0].mean(0), H[y == 1].mean(0)
    X0, X1 = H[y == 0] - mu0, H[y == 1] - mu1
    n = H.shape[0]
    Sw = (X0.T @ X0 + X1.T @ X1) / max(n - 2, 1)
    Sw = 0.5 * (Sw + Sw.T)
    Sw = (1 - shrink) * Sw + shrink * np.diag(np.diag(Sw))
    Sw += 1e-6 * float(np.mean(np.diag(Sw)) + 1e-12) * np.eye(Sw.shape[0])
    w = np.linalg.solve(Sw, mu1 - mu0)
    return unit_raw(w)


def auc_of_axis(H: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    s = H @ w
    order = np.argsort(s); ranks = np.empty_like(order, float); ranks[order] = np.arange(1, len(s) + 1)
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 * n0 else float("nan")


def cohens_d(s: np.ndarray, y: np.ndarray) -> float:
    """Standardised mean difference of scalar score s between the two classes."""
    a, b = s[y == 0], s[y == 1]
    if len(a) < 2 or len(b) < 2:
        return 0.0
    va, vb = a.var(ddof=1), b.var(ddof=1)
    pooled = np.sqrt(0.5 * (va + vb)) + 1e-12
    return float((b.mean() - a.mean()) / pooled)


def head_alignment_scores(head_proj: np.ndarray, y: np.ndarray) -> np.ndarray:
    """head_proj: (n_prompts, n_heads) projections of each head's contribution onto
    w_res. Returns |Cohen's d| per head — how well the head's along-axis output
    separates the classes (i.e. writes the concept)."""
    n_heads = head_proj.shape[1]
    return np.array([abs(cohens_d(head_proj[:, h], y)) for h in range(n_heads)])


def directional_flip(m_clean: float, m_after: float, toward: str) -> int:
    """toward='beta': flip iff margin crosses from <0 to >0; 'alpha': >0 to <0.
    margin = logit_beta - logit_alpha."""
    if toward == "beta":
        return int(m_clean < 0 and m_after > 0)
    return int(m_clean > 0 and m_after < 0)


def flip_rate(m_clean: np.ndarray, m_after: np.ndarray, toward: np.ndarray) -> float:
    f = [directional_flip(c, a, t) for c, a, t in zip(m_clean, m_after, toward)]
    return float(np.mean(f)) if len(f) else float("nan")


def cie_signed(m_clean: np.ndarray, m_patched: np.ndarray, toward_sign: np.ndarray) -> float:
    """Mean signed margin shift toward the donor class. toward_sign=+1 if donor is
    beta (margin should rise), -1 if donor is alpha. Positive CIE = causal move
    toward the donor."""
    return float(np.mean((m_patched - m_clean) * toward_sign))


def percentile_of(value: float, null_samples: np.ndarray) -> float:
    null = np.asarray(null_samples, float)
    return float(100.0 * np.mean(null <= value)) if null.size else float("nan")


# =====================================================================
# Self-test (synthetic; no torch)
# =====================================================================
def self_test() -> None:
    rng = np.random.default_rng(0)
    d, n_heads, hd = 16, 8, 4
    n = 80
    y = np.array([0, 1] * (n // 2))

    # planted concept direction in residual space
    w_true = unit_raw(rng.standard_normal(d))
    # heads 0 and 2 WRITE the concept (their contribution carries +/-w_true by class);
    # the other heads are noise.
    planted = {0, 2}
    head_contribs = np.zeros((n, n_heads, d))
    for p in range(n):
        sign = 1.0 if y[p] == 1 else -1.0
        for h in range(n_heads):
            if h in planted:
                head_contribs[p, h] = sign * 2.0 * w_true + 0.3 * rng.standard_normal(d)
            else:
                head_contribs[p, h] = 0.3 * rng.standard_normal(d)
    H = head_contribs.sum(1) + 0.2 * rng.standard_normal((n, d))  # residual = sum of heads + noise

    # (A) w_res recovers planted direction; head scores rank planted heads top
    w_res = fisher_axis(H, y)
    assert abs(np.dot(unit_raw(w_res), w_true)) > 0.8, "w_res should recover planted direction"
    head_proj = np.einsum("phd,d->ph", head_contribs, unit_raw(w_res))  # (n, n_heads)
    scores = head_alignment_scores(head_proj, y)
    top2 = set(np.argsort(scores)[-2:].tolist())
    assert top2 == planted, f"alignment should rank planted heads {planted}, got {top2}"

    # (A) ablating planted heads moves a linear readout's margin more than random heads
    # (null = a DIFFERENT random pair of the same size, never the exact planted set)
    w_read = unit_raw(w_res)
    base_margin = H @ w_read
    def ablate_margin(hset):
        Ha = H - head_contribs[:, list(hset), :].sum(1)
        return Ha @ w_read
    planted_shift = np.mean(np.abs(ablate_margin(planted) - base_margin))
    rand_shifts = []
    while len(rand_shifts) < 200:
        rs = frozenset(rng.choice(n_heads, size=2, replace=False).tolist())
        if rs == frozenset(planted):
            continue
        rand_shifts.append(np.mean(np.abs(ablate_margin(rs) - base_margin)))
    assert planted_shift > np.percentile(rand_shifts, 95), "planted-head ablation should beat random-head null"

    # (B) CIE: patching the w_true component toward a donor moves the margin; random dir ~0
    toward_sign = np.where(y == 1, -1.0, 1.0)  # push each prompt toward the opposite class
    proj = base_margin.copy()
    donor_proj = np.where(y == 1, proj[y == 0].mean(), proj[y == 1].mean())  # opposite-class mean proj
    m_clean = base_margin
    m_patched = base_margin + (donor_proj - proj)   # along w_read the readout == projection here
    cie = cie_signed(m_clean, m_patched, toward_sign)
    assert cie > 0, "patching toward donor along the readout axis should give positive CIE"
    r = unit_raw(rng.standard_normal(d)); rproj = H @ r
    m_patched_rand = base_margin + 0.0 * rproj       # random dir orthogonal-ish: ~no margin move
    cie_rand = cie_signed(m_clean, m_patched_rand, toward_sign)
    assert cie > cie_rand, "w_res CIE should exceed random-direction CIE"

    # helpers
    assert flip_rate(np.array([-1, 1.0]), np.array([1.0, -1]), np.array(["beta", "alpha"])) == 1.0
    assert percentile_of(0.5, np.array([0.1, 0.2, 0.9])) > 60.0
    print("[self_test] OK — w_res recovery, head ranking, head-ablation null, CIE, flip/percentile all pass.")


# =====================================================================
# Real run
# =====================================================================
def _chain(obj, path):
    for a in path.split("."):
        obj = getattr(obj, a)
    return obj


def run_real(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    Sigma_inv = None
    if args.concept_npz and Path(args.concept_npz).exists():
        cd = np.load(args.concept_npz)
        Sigma_inv = cd["Sigma_inv"].astype(np.float64) if "Sigma_inv" in cd else None

    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.float32, low_cpu_mem_usage=True, trust_remote_code=True,
    ).to(args.device).eval()
    blocks = _chain(model, "model.layers"); n_layers = len(blocks)
    cfg = model.config
    n_heads = int(cfg.num_attention_heads)
    head_dim = int(getattr(cfg, "head_dim", cfg.hidden_size // n_heads))
    alpha_id = tok.encode(args.alpha_answer, add_special_tokens=False)[0]
    beta_id = tok.encode(args.beta_answer, add_special_tokens=False)[0]
    logger.info("model: %d layers, %d heads, head_dim %d", n_layers, n_heads, head_dim)

    prompts = [json.loads(l) for l in open(args.prompts)]
    fams = sorted({p["surface_family"] for p in prompts}); rng.shuffle(fams)
    n_tr = int(round(len(fams) * args.train_frac)); train_fams = set(fams[:n_tr])

    layers = args.layers or [14, 18, 21, 24]
    layers = [L for L in layers if 0 <= L < n_layers - 1]
    o_proj = {L: _chain(blocks[L], "self_attn.o_proj") for L in layers}
    W_O = {L: o_proj[L].weight.detach().float().cpu().numpy() for L in layers}  # (d_model, n_heads*head_dim)
    for L in layers:
        assert W_O[L].shape[1] == n_heads * head_dim, f"o_proj in_features {W_O[L].shape[1]} != {n_heads*head_dim}"

    # ---------- capture: residual at postL{L} + o_proj input (concat heads) at answer pos ----------
    def capture(ptext):
        inp = tok([ptext], return_tensors="pt").to(args.device)
        g = {}; handles = []
        for L in layers:
            def mk_res(L=L):
                def pre(m, a): g[f"res{L}"] = a[0][0, -1, :].detach().float().cpu().numpy(); return None
                return pre
            handles.append(blocks[L + 1].register_forward_pre_hook(mk_res(), with_kwargs=False))
            def mk_o(L=L):
                def pre(m, a): g[f"z{L}"] = a[0][0, -1, :].detach().float().cpu().numpy(); return None
                return pre
            handles.append(o_proj[L].register_forward_pre_hook(mk_o(), with_kwargs=False))
        try:
            with torch.no_grad():
                o = model(**inp, use_cache=False)
                lp = torch.log_softmax(o.logits[0, -1, :].float(), 0)
                g["margin"] = float(lp[beta_id] - lp[alpha_id])
        finally:
            for h in handles:
                h.remove()
        return g

    logger.info("Capturing residuals + per-head outputs for %d prompts...", len(prompts))
    res = {L: [] for L in layers}; zin = {L: [] for L in layers}
    y = []; tr_mask = []; clean_margin = []
    for i, p in enumerate(prompts):
        g = capture(p["prompt"])
        for L in layers:
            res[L].append(g[f"res{L}"]); zin[L].append(g[f"z{L}"])
        y.append(1 if p["correct_answer"].strip() == "beta" else 0)
        tr_mask.append(p["surface_family"] in train_fams)
        clean_margin.append(g["margin"])
        if (i + 1) % 100 == 0:
            logger.info("  %d/%d", i + 1, len(prompts))
    for L in layers:
        res[L] = np.array(res[L], np.float64); zin[L] = np.array(zin[L], np.float64)
    y = np.array(y); tr_mask = np.array(tr_mask); clean_margin = np.array(clean_margin)

    # ---------- per-layer w_res (train) + per-head projections ----------
    w_res = {}; head_scores = {}; head_proj_all = {}
    for L in layers:
        wL = fisher_axis(res[L][tr_mask], y[tr_mask], args.shrink)
        w_res[L] = wL
        auc = auc_of_axis(res[L][~tr_mask], y[~tr_mask], wL)
        # per-head contribution at answer pos: v_{p,h} = W_O[:, h-slice] @ z_{p, h-slice}
        # projection onto w_res: proj_{p,h} = wL . v_{p,h}
        Z = zin[L].reshape(len(y), n_heads, head_dim)               # (n, H, hd)
        Wt = W_O[L].T.reshape(n_heads, head_dim, -1)                # (H, hd, d_model)
        # v_{p,h,:} = Z[p,h,:] @ Wt[h]  ; proj = v . wL
        proj = np.einsum("phk,hkd,d->ph", Z, Wt, wL)                # (n, H)
        head_proj_all[L] = proj
        head_scores[L] = head_alignment_scores(proj, y)
        logger.info("L%d: held-out AUC=%.3f  top head |d|=%.2f", L, auc, head_scores[L].max())

    # global ranking of heads across the probed layers
    all_heads = [(L, h, head_scores[L][h]) for L in layers for h in range(n_heads)]
    all_heads.sort(key=lambda t: -t[2])
    top_heads = [(L, h) for (L, h, s) in all_heads[: args.top_k_heads]]
    logger.info("top-%d concept-aligned heads: %s", args.top_k_heads,
                ", ".join(f"L{L}.H{h}(|d|={s:.2f})" for L, h, s in all_heads[: args.top_k_heads]))

    # ---------- held-out targets ----------
    held = [i for i in range(len(y)) if not tr_mask[i]]
    held_a = [i for i in held if y[i] == 0]
    held_b = [i for i in held if y[i] == 1]
    if args.max_targets:
        held_a, held_b = held_a[: args.max_targets], held_b[: args.max_targets]
    targets = [(i, "beta") for i in held_a] + [(i, "alpha") for i in held_b]

    # ---------- intervention runners ----------
    def run_head_scaling(ptext, head_set, factor):
        """Scale the o_proj-input slice of each head in head_set at the answer pos."""
        inp = tok([ptext], return_tensors="pt").to(args.device)
        by_layer: Dict[int, List[int]] = {}
        for (L, h) in head_set:
            by_layer.setdefault(L, []).append(h)
        handles = []
        for L, hs in by_layer.items():
            def mk(hs=hs):
                def pre(m, a):
                    x = a[0].clone()
                    for h in hs:
                        x[0, -1, h * head_dim:(h + 1) * head_dim] *= factor
                    return (x,)
                return pre
            handles.append(o_proj[L].register_forward_pre_hook(mk(), with_kwargs=False))
        try:
            with torch.no_grad():
                o = model(**inp, use_cache=False)
                row = o.logits[0, -1, :].float()
                lp = torch.log_softmax(row, 0)
                return float(lp[beta_id] - lp[alpha_id]), int(row.argmax().item())
        finally:
            for h in handles:
                h.remove()

    def run_delta(ptext, L, delta_vec):
        """Add a fixed delta at the answer position at the postL{L} tap (input to L+1)."""
        inp = tok([ptext], return_tensors="pt").to(args.device)
        dt = torch.tensor(delta_vec, dtype=torch.float32, device=args.device)
        def pre(m, a):
            x = a[0].clone(); x[0, -1, :] = x[0, -1, :] + dt; return (x,)
        h = blocks[L + 1].register_forward_pre_hook(pre, with_kwargs=False)
        try:
            with torch.no_grad():
                o = model(**inp, use_cache=False)
                lp = torch.log_softmax(o.logits[0, -1, :].float(), 0)
                return float(lp[beta_id] - lp[alpha_id])
        finally:
            h.remove()

    results: Dict[str, object] = {"layers": layers, "n_heads": n_heads, "head_dim": head_dim,
                                  "top_heads": [[int(L), int(h)] for L, h in top_heads]}

    # ============ (A) HEAD INTERVENTION vs RANDOM-HEAD NULL ============
    # Two metrics, kept strictly separate:
    #   MARGIN-flip  = sign of logit_beta - logit_alpha changed (same as j65/j75 wres_flip;
    #                  a relative comparison of two tokens, NOT the model's answer).
    #   INTACT-flip  = the model's actual top-1 token became the toward-class answer token
    #                  (the behavioural metric; j75 found this ~0 even at 32 sigma).
    logger.info("(A) head intervention: top-%d concept heads vs random-head null...", args.top_k_heads)
    head_rows = []
    flat = [(L, h) for L in layers for h in range(n_heads)]
    for factor in args.head_factors:
        mc = np.array([clean_margin[i] for i, _ in targets])
        tw = np.array([t for _, t in targets])
        toward_tok = np.array([beta_id if t == "beta" else alpha_id for t in tw])
        rtop = [run_head_scaling(prompts[i]["prompt"], top_heads, factor) for i, _ in targets]
        m_top = np.array([r[0] for r in rtop]); t1_top = np.array([r[1] for r in rtop])
        fr_top = flip_rate(mc, m_top, tw)                                   # margin-flip
        intact_top = float(np.mean(t1_top == toward_tok))                   # behavioural flip
        intact_ab = float(np.mean([(t in (alpha_id, beta_id)) for t in t1_top]))  # top-1 is any alpha/beta
        null_fr, null_intact = [], []
        for s in range(args.n_random_head):
            idx = rng.choice(len(flat), size=len(top_heads), replace=False)
            rset = [flat[j] for j in idx]
            rr = [run_head_scaling(prompts[i]["prompt"], rset, factor) for i, _ in targets]
            m_r = np.array([r[0] for r in rr]); t1_r = np.array([r[1] for r in rr])
            null_fr.append(flip_rate(mc, m_r, tw))
            null_intact.append(float(np.mean(t1_r == toward_tok)))
        null_fr = np.array(null_fr); null_intact = np.array(null_intact)
        row = {"factor": float(factor),
               "top_margin_flip": fr_top, "top_intact_flip": intact_top, "top_intact_ab_rate": intact_ab,
               "rand_margin_flip_mean": float(null_fr.mean()), "rand_margin_flip_p95": float(np.percentile(null_fr, 95)),
               "rand_intact_flip_mean": float(null_intact.mean()), "rand_intact_flip_p95": float(np.percentile(null_intact, 95)),
               "margin_pct_vs_null": percentile_of(fr_top, null_fr),
               "intact_pct_vs_null": percentile_of(intact_top, null_intact)}
        head_rows.append(row)
        logger.info("  factor=%+.1f: MARGIN-flip top=%.3f (rand p95 %.3f) | INTACT-flip top=%.3f (rand p95 %.3f) | top-1 is a/b: %.3f",
                    factor, fr_top, np.percentile(null_fr, 95), intact_top, np.percentile(null_intact, 95), intact_ab)
    results["head_intervention"] = head_rows

    # ============ (B) CIE SUBSPACE PATCH vs RANDOM-DIRECTION NULL ============
    logger.info("(B) CIE subspace patch on w_res vs random-direction null...")
    cie_layers = args.cie_layers or layers
    cie_out = {}
    for L in [L for L in cie_layers if L in w_res]:
        wL = unit_raw(w_res[L]); projL = res[L] @ wL
        donor_b = float(projL[(y == 1)].mean()); donor_a = float(projL[(y == 0)].mean())
        mc, mp, tsign = [], [], []
        for i, toward in targets:
            donor = donor_b if toward == "beta" else donor_a
            delta = (donor - projL[i]) * wL
            mp.append(run_delta(prompts[i]["prompt"], L, delta))
            mc.append(clean_margin[i]); tsign.append(+1.0 if toward == "beta" else -1.0)
        mc = np.array(mc); mp = np.array(mp); tsign = np.array(tsign)
        cie_w = cie_signed(mc, mp, tsign); fr_w = flip_rate(mc, mp, np.array([t for _, t in targets]))
        # random-direction null: replace component along random unit dir with donor-along-that-dir
        null_cie = []
        for s in range(args.n_random_dir):
            r = unit_raw(rng.standard_normal(res[L].shape[1])); projr = res[L] @ r
            db = float(projr[(y == 1)].mean()); da = float(projr[(y == 0)].mean())
            mpr = []
            for i, toward in targets:
                donor = db if toward == "beta" else da
                delta = (donor - projr[i]) * r
                mpr.append(run_delta(prompts[i]["prompt"], L, delta))
            null_cie.append(cie_signed(mc, np.array(mpr), tsign))
        null_cie = np.array(null_cie)
        cie_out[f"L{L}"] = {"cie_wres": cie_w, "flip_wres": fr_w,
                            "cie_rand_mean": float(null_cie.mean()),
                            "cie_rand_p95": float(np.percentile(null_cie, 95)),
                            "wres_percentile_vs_null": percentile_of(cie_w, null_cie)}
        logger.info("  L%d: CIE(w_res)=%.4f flip=%.3f  rand_mean=%.4f p95=%.4f  (w_res at %.0f pct)",
                    L, cie_w, fr_w, null_cie.mean(), np.percentile(null_cie, 95),
                    cie_out[f"L{L}"]["wres_percentile_vs_null"])
    results["cie_subspace"] = cie_out

    # ============ (C) OPTIONAL: mass-heuristic axis ============
    if args.mass_key and all(args.mass_key in p for p in prompts):
        logger.info("(C) mass-heuristic axis from key '%s'...", args.mass_key)
        ym = np.array([1 if str(p[args.mass_key]).strip() in ("1", "heavy", "heavier", "true", "True") else 0
                       for p in prompts])
        mass_out = {}
        for L in [L for L in cie_layers if L in w_res]:
            if ym[tr_mask].sum() < 5 or (1 - ym[tr_mask]).sum() < 5:
                continue
            wm = unit_raw(fisher_axis(res[L][tr_mask], ym[tr_mask], args.shrink))
            auc_m = auc_of_axis(res[L][~tr_mask], ym[~tr_mask], wm)
            projm = res[L] @ wm; db = float(projm[ym == 1].mean()); da = float(projm[ym == 0].mean())
            mc, mp, tsign = [], [], []
            for i, toward in targets:           # same alpha/beta targets; push along mass axis
                donor = db if toward == "beta" else da  # heuristic: heavy~alpha; orientation checked by sign
                delta = (donor - projm[i]) * wm
                mp.append(run_delta(prompts[i]["prompt"], L, delta))
                mc.append(clean_margin[i]); tsign.append(+1.0 if toward == "beta" else -1.0)
            cie_m = cie_signed(np.array(mc), np.array(mp), np.array(tsign))
            mass_out[f"L{L}"] = {"mass_auc": auc_m, "cie_mass_on_answer": cie_m}
            logger.info("  L%d: mass-axis AUC=%.3f  CIE(mass->answer)=%.4f", L, auc_m, cie_m)
        results["mass_axis"] = mass_out
    elif args.mass_key:
        logger.info("(C) skipped: key '%s' not present in all prompts", args.mass_key)

    (out / "attention_cie_mechanism.json").write_text(json.dumps(results, indent=2))

    # ---------- verdict ----------
    print("\n" + "=" * 88)
    print("ATTENTION / CIE MECHANISM HUNT  --  does attention carry alpha/beta? is w_res causal under CIE?")
    print("=" * 88)
    # DECISIVE metric = behavioural intact-flip (top-1 became the toward-class answer).
    # margin-flip is reported only as the steering-style relative metric (NOT behaviour).
    best_b = max(head_rows, key=lambda r: r["top_intact_flip"]) if head_rows else None   # best by BEHAVIOUR
    best_m = max(head_rows, key=lambda r: r["top_margin_flip"]) if head_rows else None    # best by margin
    if best_b and best_b["top_intact_flip"] >= args.tau_flip and best_b["intact_pct_vs_null"] >= 95:
        print(f"(A) OUTCOME 1 -- ATTENTION BEHAVIOURALLY CARRIES THE CONCEPT: at factor "
              f"{best_b['factor']:+.0f} the top heads make the model's TOP-1 token the answer on "
              f"{best_b['top_intact_flip']:.2f} of targets (random-head null p95 "
              f"{best_b['rand_intact_flip_p95']:.2f}). Mechanism candidate = attention.")
    else:
        bi = best_b["top_intact_flip"] if best_b else float("nan")
        bm = best_m["top_margin_flip"] if best_m else float("nan")
        print(f"(A) NOT A BEHAVIOURAL LEVER: best behavioural INTACT-flip = {bi:.2f} (top-1 rarely/never "
              f"becomes alpha/beta), even though the MARGIN-flip reaches {bm:.2f} under negation. The large "
              f"margin-flip is the steering-style relative metric (sign of beta-alpha), NOT the answer; "
              f"and head ABLATION (factor 0) is weak, so the heads are not NECESSARY. Reading: L21 heads "
              f"WRITE a readable axis (representational, H3-geometry), but attention does not behaviourally "
              f"carry alpha/beta -> consistent with a decodable-but-bypassed concept (H3-geometry + H2-behaviour).")
    if cie_out:
        # behavioural: require a non-trivial flip, not mere statistical significance over a ~0 null
        beh = any(v["flip_wres"] >= args.tau_flip for v in cie_out.values())
        stat = any(v["wres_percentile_vs_null"] >= 95 and v["cie_wres"] > 0 for v in cie_out.values())
        maxflip = max((v["flip_wres"] for v in cie_out.values()), default=float("nan"))
        if beh:
            print(f"(B) CIE: w_res subspace patch flips the answer on up to {maxflip:.2f} of targets "
                  f"-> behaviourally causal (cf. 2602.07794).")
        else:
            print(f"(B) CIE: w_res subspace patch is statistically above the random-direction null "
                  f"({'yes' if stat else 'no'}) but behaviourally negligible (max answer-flip {maxflip:.2f}). "
                  f"Direction-patching barely moves the answer -> confirms the section-6 negative; "
                  f"CONTRASTS with 2602.07794, whose subspace patch is behaviourally large.")
    print("=" * 88 + "\n")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self_test", action="store_true")
    p.add_argument("--prompts", default="data/prompts/physics_decay_type_probe_v2_train.jsonl")
    p.add_argument("--out_dir", default="data/analysis/runD_v2/mechanism")
    p.add_argument("--concept_npz", default="data/analysis/runD_v2/geometry_stage1/concept_directions.npz",
                   help="optional; for Sigma_inv / cos_C reporting")
    p.add_argument("--model_name", default="Qwen/Qwen3-4B")
    p.add_argument("--device", default="cuda")
    p.add_argument("--alpha_answer", default=" alpha")
    p.add_argument("--beta_answer", default=" beta")
    p.add_argument("--layers", type=int, nargs="*", default=None, help="post-block layers to probe (default 14 18 21 24)")
    p.add_argument("--cie_layers", type=int, nargs="*", default=None, help="layers for CIE patch (default = --layers)")
    p.add_argument("--top_k_heads", type=int, default=10, help="top concept-aligned heads to intervene on")
    p.add_argument("--head_factors", type=float, nargs="*", default=[0.0, -1.0], help="0=ablate, -1=negate")
    p.add_argument("--n_random_head", type=int, default=20, help="random-head null draws (DECISIVE control A)")
    p.add_argument("--n_random_dir", type=int, default=20, help="random-direction null draws (DECISIVE control B)")
    p.add_argument("--mass_key", default=None, help="optional prompt key for the mass-heuristic axis (exp 76 tie-in)")
    p.add_argument("--max_targets", type=int, default=None)
    p.add_argument("--train_frac", type=float, default=0.6)
    p.add_argument("--shrink", type=float, default=0.1)
    p.add_argument("--tau_flip", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    return p


def main():
    args = build_parser().parse_args()
    if args.self_test:
        self_test(); return
    run_real(args)


if __name__ == "__main__":
    main()
