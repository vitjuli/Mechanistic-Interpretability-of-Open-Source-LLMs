"""
61_subspace_interventions_pilot.py
===================================================================
PILOT for Stage-2 interventional sufficiency. Decides, on ~30 contrastive
pairs, whether the alpha/beta concept is interventionally sufficient in a
LOW-DIMENSIONAL subspace, and if so under which subspace DEFINITION:

    Method C  : span{lbar}, lbar = Sigma^{-1} gbar      (1-D, Park-Veitch steering)
    Method B  : top-r directions of a 148-cue-group Fisher (Sigma_within) whitening
                (r in {1,3,8,11}; B(r=1) is a deliberately weak lower bound)
    Method A  : span{d_f : f in cluster} for the L24 carrier (A1=C16 2 feats,
                A3=C13 ~20 feats) -- the decoder-defined baseline anchor

The pilot is built around the THREE failure modes that make naive subspace
patching meaningless; each is defended IN CODE, not "checked later":

  (1) Subspace illusion (Makelov, Lange, Nanda 2024): a norm-matched S^perp
      control patch. If S^perp flips as often as S, the result is void.
  (2) Label leakage (Method B uses Sigma_within fit on V_h labels): S is fit on
      a TRAIN half (by surface_family); flips are measured on a HELD-OUT half.
  (3) Format transfer: contrastive pairs are drawn WITHIN one surface_family
      (minimal pairs), and projections use the WHITENED metric that suppresses
      the format-dominant direction.

THREE CORRECTNESS FIXES baked in (these change whether numbers are valid):
  * M-orthogonal projection. The projector and the S^perp complement are built
    in the SAME metric M that defines S (M = Sigma_within^{-1} for B, Sigma^{-1}
    for C). We work in whitened coords h~ = A h (A = M^{1/2}); ordinary
    Euclidean projection there == M-orthogonal projection in raw coords, and
    S^perp is the Euclidean orthogonal complement THERE (the only complement
    that is truly M-orthogonal to S).
  * Norm-matched control. The S^perp patch is rescaled so its residual-stream
    L2 norm ||Delta h|| equals the S patch's, PER PAIR (asserted), so "control
    is weaker" can't be a norm artefact.
  * Sign-relative-to-clean flip. A flip counts only if the patch MOVES the sign
    of Delta logit = logit(beta) - logit(alpha) from its clean value to the
    donor's value (Geiger interchange criterion), robust to near-zero logits.

PATCH POINT: residual stream INPUT to block ell (register_forward_pre_hook on
model.layers[ell]). This differs from scripts 51/52/54 (which patch the MLP
input at post_attention_layernorm); residual-stream patching is the point
consistent with the theory (gbar/lbar live in the residual basis the
unembedding reads). Toggle with --patch_point if your context needs the old one.

OUTPUTS (data/analysis/iia_failure_diagnosis/):
  subspace_pilot_results.csv   one row per (pair, method, layer_config, patch/control)
  subspace_pilot_summary.json  per-(method,layer_config) flip-rate, S^perp-rate,
                               gap, bootstrap CIs; flip-rate-as-function-of-r;
                               steering sweep for C; the verdict-table
  subspace_pilot_steering.csv  steering sweep (Method C, no donor)

SELF-TEST (no torch / no repo): python 61_subspace_interventions_pilot.py --self_test
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Add project root to sys.path so `from src.transcoder import ...` works
# (matches the pattern used by scripts 52/53/54)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("subspace_pilot")

FEATURE_ID_RE = re.compile(r"^[Ll](\d+)[_:\-][Ff]?(\d+)$")

# Module path to the transformer blocks. Scripts 51 use model.model.model.layers
# (ModelWrapper), 52 uses hf_model.model.layers (raw HF). This script builds raw HF.
HF_LAYERS_PATH = "model.layers"   # accessed as getattr-chain on the loaded model


# =====================================================================
# Geometry core (pure numpy; unit-tested by --self_test)
# =====================================================================

def whitener_from_cov(Sigma: np.ndarray, ridge: float = 1e-3) -> Tuple[np.ndarray, np.ndarray, float]:
    """Return (A = Sigma^{-1/2}, Sigma_inv, cond) for an SPD covariance, ridge-regularised."""
    d = Sigma.shape[0]
    Sigma = 0.5 * (Sigma + Sigma.T)
    ridge_abs = ridge * float(np.mean(np.diag(Sigma)))
    evals, evecs = np.linalg.eigh(Sigma + ridge_abs * np.eye(d))
    evals = np.clip(evals, 1e-30, None)
    cond = float(evals.max() / evals.min())
    A = (evecs * (1.0 / np.sqrt(evals))) @ evecs.T
    Sigma_inv = (evecs * (1.0 / evals)) @ evecs.T
    return A, Sigma_inv, cond


def whitened_basis(S_raw: np.ndarray, A: np.ndarray, rank_tol: float = 1e-8) -> np.ndarray:
    """
    Given subspace directions S_raw (k, d) in raw coords and whitener A (d, d),
    return an orthonormal basis Q (d, r) of A·S in WHITENED coords (r = numeric rank).
    Projection in whitened coords onto Q is M-orthogonal projection in raw coords.
    """
    if S_raw.ndim == 1:
        S_raw = S_raw[None, :]
    Sw = (A @ S_raw.T)                       # (d, k) whitened columns
    U, s, _ = np.linalg.svd(Sw, full_matrices=False)
    r = int(np.sum(s > rank_tol * s.max())) if s.size else 0
    return U[:, :r]                          # (d, r), orthonormal in whitened coords


def project_whitened(h_raw: np.ndarray, Q: np.ndarray, A: np.ndarray, Ainv: np.ndarray) -> np.ndarray:
    """M-orthogonal projection of h onto S, returned in RAW coords. Q is whitened basis."""
    hw = A @ h_raw
    pw = Q @ (Q.T @ hw)
    return Ainv @ pw


def interchange_delta(h_target_raw, h_donor_raw, Q, A, Ainv):
    """
    Subspace interchange in raw coords: replace S-projection of target with donor's.
        h' = h_target - P_S h_target + P_S h_donor
    Returns (h_patched_raw, delta_vec_raw) where delta = h' - h_target.
    """
    p_t = project_whitened(h_target_raw, Q, A, Ainv)
    p_d = project_whitened(h_donor_raw, Q, A, Ainv)
    delta = p_d - p_t
    return h_target_raw + delta, delta


def perp_control_delta(delta_S_raw, Q, A, Ainv, rng, mode="random"):
    """
    Norm-matched control patch living in S^perp (M-orthogonal complement).
    Built in whitened coords so it is truly M-orthogonal to S, then rescaled so
    its RAW residual-stream L2 norm equals ||delta_S_raw||.
      mode='random': random direction in whitened S^perp.
    Returns delta_perp_raw with ||delta_perp_raw|| == ||delta_S_raw|| (asserted by caller).
    """
    d = A.shape[0]
    # random vector, remove its whitened-S component -> lands in whitened S^perp
    v = rng.standard_normal(d)
    vw = v - Q @ (Q.T @ v)
    nv = np.linalg.norm(vw)
    if nv < 1e-12:
        vw = rng.standard_normal(d); vw = vw - Q @ (Q.T @ vw); nv = np.linalg.norm(vw)
    vw = vw / nv
    delta_perp_raw = np.linalg.inv(A) @ vw if Ainv is None else Ainv @ vw
    # rescale to match raw L2 norm of the S patch
    target = float(np.linalg.norm(delta_S_raw))
    cur = float(np.linalg.norm(delta_perp_raw))
    if cur > 0:
        delta_perp_raw = delta_perp_raw * (target / cur)
    return delta_perp_raw


def flip_indicator(delta_logit_clean: float, delta_logit_patched: float, delta_logit_donor: float) -> int:
    """
    Geiger interchange flip: patch moved the SIGN of (logit_beta - logit_alpha)
    from its clean value to the donor's value.
    """
    s_clean = np.sign(delta_logit_clean)
    s_patch = np.sign(delta_logit_patched)
    s_donor = np.sign(delta_logit_donor)
    return int((s_patch == s_donor) and (s_patch != s_clean))


def bootstrap_ci(x: np.ndarray, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> Tuple[float, float, float]:
    """Mean and percentile bootstrap CI for a 0/1 (or real) array."""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    boots = np.array([rng.choice(x, size=x.size, replace=True).mean() for _ in range(n_boot)])
    return float(x.mean()), float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def fisher_within_cov(H: np.ndarray, group_ids: np.ndarray) -> np.ndarray:
    """
    Pooled within-class covariance Sigma_within over groups (148 cue-groups).
    H: (n, d) activations; group_ids: (n,) integer labels. Returns (d, d).
    """
    d = H.shape[1]
    Sw = np.zeros((d, d), dtype=np.float64)
    n_used = 0
    for g in np.unique(group_ids):
        Hg = H[group_ids == g]
        if Hg.shape[0] < 2:
            continue
        Xc = Hg - Hg.mean(axis=0)
        Sw += Xc.T @ Xc
        n_used += Hg.shape[0]
    if n_used == 0:
        raise ValueError("no group has >=2 members; cannot form Sigma_within")
    return Sw / n_used


def fisher_topr_directions(H: np.ndarray, group_ids: np.ndarray, gbar: np.ndarray,
                           A_within: np.ndarray, r: int) -> np.ndarray:
    """
    Method B subspace: top-r whitened-PCA directions of the between-group structure,
    i.e. PCA of Sigma_within-whitened class MEANS, ranked by |corr with gbar| so the
    most V_h-aligned directions come first. Returns r raw-space directions (r, d).
    """
    # whiten, compute group means, PCA on the mean cloud
    Hw = H @ A_within.T
    gids = np.unique(group_ids)
    means = np.array([Hw[group_ids == g].mean(axis=0) for g in gids])  # (G, d) whitened
    means_c = means - means.mean(axis=0)
    U, s, Vt = np.linalg.svd(means_c, full_matrices=False)
    comps_w = Vt                                       # (k, d) whitened PCA dirs
    gbar_w = A_within @ gbar
    gbar_w = gbar_w / (np.linalg.norm(gbar_w) + 1e-12)
    aligns = np.abs(comps_w @ gbar_w)
    order = np.argsort(-aligns)
    top_w = comps_w[order[:r]]                         # (r, d) whitened
    # back to raw coords (directions transform by A^{-1}); caller re-whitens anyway
    Ainv_within = np.linalg.inv(A_within)
    return (Ainv_within @ top_w.T).T                   # (r, d) raw


def parse_feature_id(fid: str) -> Tuple[int, int]:
    m = FEATURE_ID_RE.match(str(fid).strip())
    if not m:
        raise ValueError(f"cannot parse feature_id {fid!r}")
    return int(m.group(1)), int(m.group(2))


# =====================================================================
# Self-test: synthetic model, validates the three fixes behave correctly
# =====================================================================

def self_test() -> None:
    rng = np.random.default_rng(11)
    d = 48
    # anisotropic activation space: a big FORMAT axis (e_f) ~orthogonal to concept,
    # a smaller CONCEPT axis (e_c) that actually separates classes.
    e_f = np.zeros(d); e_f[0] = 1.0
    e_c = np.zeros(d); e_c[1] = 1.0
    n = 600
    labels = rng.integers(0, 2, size=n)              # 0=alpha, 1=beta
    groups = rng.integers(0, 50, size=n)             # 50 cue-groups
    H = (rng.standard_normal((n, d)) * 0.3)
    H[:, 0] += rng.standard_normal(n) * 3.0          # format: huge variance, label-free
    H[:, 1] += (labels * 2.0 - 1.0) * 1.2            # concept: sign tracks label
    # concept covariance for C: make e_c moderate-variance
    Sigma = np.cov(H, rowvar=False)
    A, Sigma_inv, cond = whitener_from_cov(Sigma)
    gbar = 2.0 * e_c                                  # concept contrast ~ along e_c
    lbar = Sigma_inv @ gbar

    # Method C subspace (1-D lbar), whitened basis
    Q_C = whitened_basis(lbar, A)
    Ainv = np.linalg.inv(A)

    # a "donor" with beta-ward concept value and a "target" with alpha-ward
    h_t = -1.2 * e_c + 4.0 * e_f + 0.2 * rng.standard_normal(d)   # alpha-ish, some format
    h_d = +1.2 * e_c - 1.0 * e_f + 0.2 * rng.standard_normal(d)   # beta-ish

    h_patched, delta_S = interchange_delta(h_t, h_d, Q_C, A, Ainv)
    # Toy logit read the way the REAL model reads it: along the concept contrast
    # gbar in the dual (logits = unembedding . residual). Method C's S = span{lbar}.
    ghat = gbar / (np.linalg.norm(gbar) + 1e-12)
    def toy_logit(h): return float(h @ ghat)          # logit(beta)-logit(alpha) proxy
    f_C = flip_indicator(toy_logit(h_t), toy_logit(h_patched), toy_logit(h_d))

    # A 2-D subspace that actually spans the concept axis (e_c) plus one nuisance:
    # this is the analogue of Method B with enough rank to capture the concept.
    S2 = np.vstack([e_c, e_f * 0.0 + np.eye(d)[2]])   # {e_c, e_2}
    Q2 = whitened_basis(S2, A)
    h_patched2, delta_S2 = interchange_delta(h_t, h_d, Q2, A, Ainv)
    f_B = flip_indicator(toy_logit(h_t), toy_logit(h_patched2), toy_logit(h_d))

    # control in S^perp, norm-matched (built against the 2-D concept subspace)
    delta_perp = perp_control_delta(delta_S2, Q2, A, Ainv, rng)
    norm_match = abs(np.linalg.norm(delta_S2) - np.linalg.norm(delta_perp))
    h_perp = h_t + delta_perp
    f_perp = flip_indicator(toy_logit(h_t), toy_logit(h_perp), toy_logit(h_d))

    # M-orthogonality of S^perp to S (in whitened coords): Q^T (A delta_perp) ~ 0
    ortho_resid = float(np.linalg.norm(Q2.T @ (A @ delta_perp)))

    print("\n--- SELF TEST -------------------------------------------------")
    print(f"d={d}  cond(Sigma)={cond:.2e}")
    print(f"C (1-D lbar):   concept {toy_logit(h_t):+.3f} -> {toy_logit(h_patched):+.3f} "
          f"(donor {toy_logit(h_d):+.3f})  flip_C={f_C}  [1-D may undershoot -- informative]")
    print(f"B (2-D w/ e_c): concept {toy_logit(h_t):+.3f} -> {toy_logit(h_patched2):+.3f}  flip_B={f_B}")
    print(f"S^perp ctrl:    ||delta_perp||={np.linalg.norm(delta_perp):.4f}  "
          f"norm_match_err={norm_match:.2e}  flip_perp={f_perp}")
    print(f"M-orthogonality of S^perp to S (whitened residual): {ortho_resid:.2e}")

    # Fisher Method B sanity: top-1 should partly align with gbar but < 1
    A_w = whitener_from_cov(fisher_within_cov(H, groups))[0]
    dirs_b = fisher_topr_directions(H, groups, gbar, A_w, r=1)
    cos_b = abs(float((A @ dirs_b[0]) @ (A @ gbar) /
                      (np.linalg.norm(A @ dirs_b[0]) * np.linalg.norm(A @ gbar) + 1e-12)))
    print(f"Method B(r=1) top dir cos_C with gbar: {cos_b:.3f}  (expected: partial, <1)")

    assert f_B == 1, "subspace interchange spanning the concept axis must flip the sign"
    assert f_perp == 0, "norm-matched S^perp control must NOT flip (else illusion)"
    assert norm_match < 1e-9, "control must be exactly norm-matched"
    assert ortho_resid < 1e-9, "S^perp must be M-orthogonal to S"
    print("\nALL SELF-TEST ASSERTIONS PASSED ")
    print("(note: flip_C is reported, not asserted -- whether 1-D suffices is the empirical question)")
    print("---------------------------------------------------------------\n")


# =====================================================================
# Real run (CSD3) -- torch / transformers / repo
# =====================================================================

def _getattr_chain(obj, path):
    for a in path.split("."):
        obj = getattr(obj, a)
    return obj


def run_real(args: argparse.Namespace) -> None:
    import pandas as pd
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    device = args.device
    rng = np.random.default_rng(args.seed)

    # ---- concept directions from 60_ (gbar, lbar, whitener) ----
    cd = np.load(args.concept_npz)
    gbar = cd["gbar"].astype(np.float64)
    lbar = cd["lbar"].astype(np.float64) if "lbar" in cd else None
    # full-Sigma whitener for Method C; recompute Sigma_inv if needed
    if "Sigma" in cd:
        A_full, Sigma_inv_full, _ = whitener_from_cov(cd["Sigma"].astype(np.float64), ridge=args.ridge)
    else:
        raise SystemExit("concept_npz must contain 'Sigma' (re-run 60_ to dump it)")
    if lbar is None:
        lbar = Sigma_inv_full @ gbar
    Ainv_full = np.linalg.inv(A_full)
    d = gbar.shape[0]

    # ---- model + tokenizer (raw HF, base model!) ----
    logger.info("Loading %s (base)", args.model_name)
    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.float32, low_cpu_mem_usage=True, trust_remote_code=True,
    ).to(device).eval()
    blocks = _getattr_chain(model, HF_LAYERS_PATH)

    a_id = tok.encode(args.alpha_answer, add_special_tokens=False)
    b_id = tok.encode(args.beta_answer, add_special_tokens=False)
    if len(a_id) != 1 or len(b_id) != 1:
        logger.warning("answers not single-token (alpha=%s beta=%s); using first sub-token", a_id, b_id)
    alpha_id, beta_id = a_id[0], b_id[0]

    # ---- prompts + split by surface_family ----
    prompts = [json.loads(l) for l in open(args.prompts)]
    fams = sorted({p["surface_family"] for p in prompts})
    rng.shuffle(fams)
    n_train = int(round(len(fams) * args.train_frac))
    train_fams = set(fams[:n_train]); held_fams = set(fams[n_train:])
    train_prompts = [p for p in prompts if p["surface_family"] in train_fams]
    logger.info("families: %d train / %d held-out; %d train prompts",
                len(train_fams), len(held_fams), len(train_prompts))

    # ---- capture residual-stream input to each candidate layer ----
    layer_set = sorted(set(sum([cfg for cfg in args.layer_configs], [])))

    def capture_resid(prompt_text: str, layers: List[int]) -> Dict[int, np.ndarray]:
        inputs = tok([prompt_text], return_tensors="pt").to(device)
        grabbed: Dict[int, np.ndarray] = {}
        handles = []
        for L in layers:
            def _make(L=L):
                def _pre(module, args_in):
                    hs = args_in[0]
                    grabbed[L] = hs[0, -1, :].detach().float().cpu().numpy()
                    return None
                return _pre
            handles.append(blocks[L].register_forward_pre_hook(_make(), with_kwargs=False))
        try:
            with torch.no_grad():
                model(**inputs, use_cache=False)
        finally:
            for h in handles:
                h.remove()
        return grabbed

    def clean_delta_logit(prompt_text: str) -> float:
        inputs = tok([prompt_text], return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs, use_cache=False)
        lp = torch.log_softmax(out.logits[0, -1, :].float(), dim=0)
        return float(lp[beta_id] - lp[alpha_id])      # canonical: beta - alpha

    def patched_delta_logit(prompt_text: str, patches: Dict[int, np.ndarray]) -> float:
        inputs = tok([prompt_text], return_tensors="pt").to(device)
        handles = []
        for L, delta in patches.items():
            dt = torch.tensor(delta, dtype=torch.float32, device=device)
            def _make(dt=dt):
                def _pre(module, args_in):
                    hs = args_in[0].clone()
                    hs[0, -1, :] = hs[0, -1, :] + dt   # add delta to residual input
                    return (hs,)
                return _pre
            handles.append(blocks[L].register_forward_pre_hook(_make(), with_kwargs=False))
        try:
            with torch.no_grad():
                out = model(**inputs, use_cache=False)
        finally:
            for h in handles:
                h.remove()
        lp = torch.log_softmax(out.logits[0, -1, :].float(), dim=0)
        return float(lp[beta_id] - lp[alpha_id])

    # ---- build Sigma_within (148 cue-groups) on TRAIN activations, per layer ----
    # group key: cue_type (falls back to group_id). One Sigma_within per layer.
    def group_key(p):  # 148 cue-groups
        return p.get("cue_type") or p.get("group_id") or p["surface_family"]

    logger.info("Capturing TRAIN residuals for Sigma_within (layers %s)...", layer_set)
    train_acts = {L: [] for L in layer_set}
    train_groups = []
    for p in train_prompts:
        g = capture_resid(p["prompt"], layer_set)
        for L in layer_set:
            train_acts[L].append(g[L])
        train_groups.append(group_key(p))
    train_groups = np.array(train_groups)
    gid_map = {g: i for i, g in enumerate(sorted(set(train_groups)))}
    train_gids = np.array([gid_map[g] for g in train_groups])
    A_within: Dict[int, np.ndarray] = {}
    for L in layer_set:
        H = np.array(train_acts[L])
        A_within[L] = whitener_from_cov(fisher_within_cov(H, train_gids), ridge=args.ridge)[0]
        logger.info("  layer %d: Sigma_within over %d groups, n=%d", L, len(gid_map), H.shape[0])

    # ---- assemble subspace direction sets (raw coords) per layer ----
    # Method A: decoder rows for the named clusters
    cl = pd.read_csv(args.cluster_labels)
    if "feature_id" not in cl.columns:
        cl = cl.rename(columns={cl.columns[0]: "feature_id"})
    cl["feature_id"] = cl["feature_id"].astype(str)

    def _norm_id(cid: str) -> str:
        """Strip optional leading 'C' from cluster IDs ('C16' -> '16'),
        and trailing '.0' from float-coerced ints ('16.0' -> '16')."""
        s = str(cid).strip().lstrip("C").lstrip("c")
        if s.endswith(".0"):
            s = s[:-2]
        return s

    A_feats = {}
    # A1 uses --cluster_col (default agglo_coimp_subgroup_k30)
    if args.cluster_col in cl.columns:
        cid = _norm_id(args.A1_cluster)
        cl_normed = cl[args.cluster_col].apply(_norm_id)
        sub = cl[cl_normed == cid]
        A_feats[args.A1_cluster] = [parse_feature_id(f) for f in sub["feature_id"]]
        logger.info("A1 cluster '%s' (col=%s): %d features",
                    args.A1_cluster, args.cluster_col, len(sub))
    else:
        logger.warning("--cluster_col %s not in cluster_labels", args.cluster_col)

    # A3 uses --A3_cluster_col (default = same as --cluster_col, but can override)
    a3_col = args.A3_cluster_col or args.cluster_col
    if a3_col in cl.columns:
        cid = _norm_id(args.A3_cluster)
        cl_normed_a3 = cl[a3_col].apply(_norm_id)
        sub3 = cl[cl_normed_a3 == cid]
        A_feats[args.A3_cluster] = [parse_feature_id(f) for f in sub3["feature_id"]]
        logger.info("A3 cluster '%s' (col=%s): %d features",
                    args.A3_cluster, a3_col, len(sub3))
    else:
        logger.warning("--A3_cluster_col %s not in cluster_labels", a3_col)
    from src.transcoder import load_transcoder_set
    tset = load_transcoder_set(model_size=args.model_size, device=device, lazy_load=True,
                               layers=layer_set)

    def decoder_dirs(feat_list):
        import torch as _t
        rows = []
        for (L, fi) in feat_list:
            tc = tset[L]
            rows.append(tc._get_decoder_vectors(_t.tensor([fi])).detach().float().cpu().numpy()[0])
        return np.array(rows, dtype=np.float64) if rows else np.zeros((0, d))

    # Method B directions per layer & r ; Method C is layer-independent (lbar)
    r_list = args.r_values
    H_train_byL = {L: np.array(train_acts[L]) for L in layer_set}

    def make_subspaces_for_layer(L):
        S = {}
        # C: 1-D lbar (full-Sigma metric); basis whitened by A_full
        S["C(1D)"] = ("full", whitened_basis(lbar, A_full))
        # B(r): top-r Fisher dirs (within-metric); basis whitened by A_within[L]
        for r in r_list:
            dirs = fisher_topr_directions(H_train_byL[L], train_gids, gbar, A_within[L], r=r)
            S[f"B(r={r})"] = ("within", whitened_basis(dirs, A_within[L]), L)
        # A1/A3 decoder spans (full-Sigma metric for projection comparability)
        if A_feats:
            S["A1"] = ("full", whitened_basis(decoder_dirs(A_feats[args.A1_cluster]), A_full))
            S["A3"] = ("full", whitened_basis(decoder_dirs(A_feats[args.A3_cluster]), A_full))
        return S

    # ---- contrastive pairs from HELD-OUT, within surface_family, balanced ----
    held = [p for p in prompts if p["surface_family"] in held_fams]
    by_fam: Dict[str, Dict[str, list]] = {}
    for p in held:
        lab = "beta" if p["correct_answer"].strip() == "beta" else "alpha"
        by_fam.setdefault(p["surface_family"], {"alpha": [], "beta": []})[lab].append(p)
    pairs = []
    for fam, d2 in by_fam.items():
        for pa in d2["alpha"]:
            for pb in d2["beta"]:
                pairs.append((pa, pb)); break
        if len(pairs) >= args.n_pairs:
            break
    rng.shuffle(pairs)
    pairs = pairs[:args.n_pairs]
    logger.info("Using %d held-out within-family contrastive pairs", len(pairs))

    # ---- run interchange + control across methods/layer-configs ----
    def metric_pack(metric_name, L):
        if metric_name == "full":
            return A_full, Ainv_full
        return A_within[L], np.linalg.inv(A_within[L])

    rows = []
    # precompute donor & target residuals for all needed layers (Option 2)
    cap_cache: Dict[str, Dict[int, np.ndarray]] = {}
    def cap(ptext):
        if ptext not in cap_cache:
            cap_cache[ptext] = capture_resid(ptext, layer_set)
        return cap_cache[ptext]

    for (pa, pb) in pairs:
        # both interchange directions: alpha<-beta (target=pa) and beta<-alpha (target=pb)
        for (tgt, dnr, direction) in [(pa, pb, "a<-b"), (pb, pa, "b<-a")]:
            dlc = clean_delta_logit(tgt["prompt"])
            dld = clean_delta_logit(dnr["prompt"])
            ha = cap(tgt["prompt"]); hd = cap(dnr["prompt"])
            for cfg in args.layer_configs:
                cfg_name = "+".join(f"L{L}" for L in cfg)
                subs = {L: make_subspaces_for_layer(L) for L in cfg}
                method_names = list(subs[cfg[0]].keys())
                for m in method_names:
                    # build per-layer S patch and a norm-matched S^perp patch
                    patchS, patchP = {}, {}
                    tot_norm = 0.0
                    for L in cfg:
                        entry = subs[L][m]
                        metric_name, Q = entry[0], entry[1]
                        A_m, Ainv_m = metric_pack(metric_name, L)
                        _, dS = interchange_delta(ha[L], hd[L], Q, A_m, Ainv_m)
                        dP = perp_control_delta(dS, Q, A_m, Ainv_m, rng)
                        # enforce exact norm match
                        assert abs(np.linalg.norm(dS) - np.linalg.norm(dP)) < 1e-6, "norm match failed"
                        patchS[L] = dS; patchP[L] = dP
                        tot_norm += float(np.linalg.norm(dS))
                    dlp_S = patched_delta_logit(tgt["prompt"], patchS)
                    dlp_P = patched_delta_logit(tgt["prompt"], patchP)
                    rows.append({
                        "fam": tgt["surface_family"], "direction": direction,
                        "tgt_id": tgt.get("prompt_id"), "dnr_id": dnr.get("prompt_id"),
                        "method": m, "layer_config": cfg_name,
                        "dl_clean": dlc, "dl_donor": dld,
                        "dl_patch_S": dlp_S, "dl_patch_perp": dlp_P,
                        "flip_S": flip_indicator(dlc, dlp_S, dld),
                        "flip_perp": flip_indicator(dlc, dlp_P, dld),
                        "patch_norm": tot_norm,
                    })

    df = pd.DataFrame(rows)
    df.to_csv(out / "subspace_pilot_results.csv", index=False)

    # ---- steering sweep for Method C (no donor): add c * sigma_lambda * lhat ----
    # sigma along lbar on TRAIN (whitened-metric units)
    lhat = lbar / np.sqrt(float(lbar @ Sigma_inv_full @ lbar) + 1e-12)   # unit in C-metric
    # raw steering vector direction (unit raw L2) for hooking
    steer_raw = lhat.copy()
    proj_train = []
    for L in args.layer_configs[0]:
        H = H_train_byL[L]
        proj_train.append((H @ Sigma_inv_full @ lbar))
    sigma_lam = float(np.std(np.concatenate(proj_train))) if proj_train else 1.0
    steer_rows = []
    steer_layer_cfg = args.layer_configs[0]
    for pa, pb in pairs[: min(len(pairs), args.n_pairs)]:
        for tgt in (pa,):  # steer alpha-target toward beta
            dlc = clean_delta_logit(tgt["prompt"])
            for c in args.steer_sweep:
                delta = (c * sigma_lam) * steer_raw
                patches = {L: delta for L in steer_layer_cfg}
                dlp = patched_delta_logit(tgt["prompt"], patches)
                steer_rows.append({
                    "tgt_id": tgt.get("prompt_id"), "c_sigma": c,
                    "dl_clean": dlc, "dl_steered": dlp,
                    "flipped": int(np.sign(dlp) != np.sign(dlc) and dlp > 0),
                    "patch_norm": float(np.linalg.norm(delta)) * len(steer_layer_cfg),
                })
    steer_df = pd.DataFrame(steer_rows)
    steer_df.to_csv(out / "subspace_pilot_steering.csv", index=False)

    # ---- summarise: flip-rate, S^perp-rate, gap, bootstrap CIs ----
    summary = {"n_pairs": len(pairs), "directions": ["a<-b", "b<-a"],
               "tau_flip": args.tau_flip, "tau_ctrl": args.tau_ctrl,
               "layer_configs": ["+".join(f"L{L}" for L in c) for c in args.layer_configs],
               "by_method_layer": [], "flip_rate_vs_r": {}, "verdict": None}
    best = None
    for (m, cfg), g in df.groupby(["method", "layer_config"]):
        fS, loS, hiS = bootstrap_ci(g["flip_S"].values, seed=args.seed)
        fP, loP, hiP = bootstrap_ci(g["flip_perp"].values, seed=args.seed)
        gap, glo, ghi = bootstrap_ci(g["flip_S"].values - g["flip_perp"].values, seed=args.seed)
        rec = {"method": m, "layer_config": cfg, "n": int(len(g)),
               "flip_S": round(fS, 3), "flip_S_CI": [round(loS, 3), round(hiS, 3)],
               "flip_perp": round(fP, 3), "flip_perp_CI": [round(loP, 3), round(hiP, 3)],
               "gap_S_minus_perp": round(gap, 3), "gap_CI": [round(glo, 3), round(ghi, 3)],
               "passes": bool(fS >= args.tau_flip and fP <= args.tau_ctrl)}
        summary["by_method_layer"].append(rec)
        if best is None or fS > best["flip_S"]:
            best = rec
    # flip-rate vs r (best layer config per r): the minimal-sufficient-dimension curve
    for cfg in summary["layer_configs"]:
        sub = {rec["method"]: rec["flip_S"] for rec in summary["by_method_layer"]
               if rec["layer_config"] == cfg and rec["method"].startswith("B(r=")}
        if sub:
            summary["flip_rate_vs_r"][cfg] = sub
    # steering summary
    if len(steer_df):
        srate = steer_df.groupby("c_sigma")["flipped"].mean().to_dict()
        summary["steering_flip_by_c_sigma"] = {str(k): round(float(v), 3) for k, v in srate.items()}

    # verdict
    def rate(method_prefix):
        vals = [r["flip_S"] for r in summary["by_method_layer"] if r["method"].startswith(method_prefix)]
        return max(vals) if vals else 0.0
    cR, bR = rate("C("), rate("B(")
    perp_ok = all(r["flip_perp"] <= args.tau_ctrl for r in summary["by_method_layer"] if r["passes"]) \
        if any(r["passes"] for r in summary["by_method_layer"]) else None
    if best and best["flip_perp"] > args.tau_ctrl and best["flip_S"] >= args.tau_flip:
        verdict = "OUTCOME 3 (subspace illusion): S flips but norm-matched S^perp also flips -> result void; reduce norm / re-examine S."
    elif cR >= args.tau_flip and cR >= bR:
        verdict = "OUTCOME 1 (narrow LRH): 1-D lbar is interventionally sufficient. Scale C only."
    elif bR >= args.tau_flip and bR > cR:
        verdict = "OUTCOME 2 (distributed linear): top-r subspace sufficient, 1-D not. Scale B; find minimal r from flip_rate_vs_r."
    else:
        verdict = "OUTCOME 4 (no subspace sufficiency): nothing reaches tau_flip -> weaken theorem to representational (Cond I + necessity); drop/qualify Thm 1(2) & Corollary 1."
    summary["verdict"] = verdict

    with open(out / "subspace_pilot_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=float)

    # ---- console verdict table ----
    print("\n" + "=" * 78)
    print("SUBSPACE INTERVENTION PILOT  --  verdict table")
    print("=" * 78)
    print(f"{'method':10} {'layers':10} {'flip_S':>16} {'flip_perp':>16} {'gap':>14} pass")
    for rec in sorted(summary["by_method_layer"], key=lambda r: (-r["flip_S"], r["method"])):
        ciS = f"[{rec['flip_S_CI'][0]:.2f},{rec['flip_S_CI'][1]:.2f}]"
        ciP = f"[{rec['flip_perp_CI'][0]:.2f},{rec['flip_perp_CI'][1]:.2f}]"
        print(f"{rec['method']:10} {rec['layer_config']:10} "
              f"{rec['flip_S']:.2f} {ciS:>11} {rec['flip_perp']:.2f} {ciP:>11} "
              f"{rec['gap_S_minus_perp']:+.2f} {('YES' if rec['passes'] else '·'):>5}")
    if summary["flip_rate_vs_r"]:
        print("\nflip-rate vs r (minimal sufficient dimension):")
        for cfg, sub in summary["flip_rate_vs_r"].items():
            order = sorted(sub.items(), key=lambda kv: int(kv[0].split("=")[1].rstrip(")")))
            print(f"  {cfg}: " + "  ".join(f"{k}={v:.2f}" for k, v in order))
    if "steering_flip_by_c_sigma" in summary:
        print("\nMethod C steering sweep (flip rate by c·sigma):")
        print("  " + "  ".join(f"{k}={v}" for k, v in summary["steering_flip_by_c_sigma"].items()))
    print("\nVERDICT: " + verdict)
    print(f"\nwrote: {out}/subspace_pilot_results.csv, subspace_pilot_summary.json, subspace_pilot_steering.csv")
    print("=" * 78)


# =====================================================================
# CLI
# =====================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self_test", action="store_true")

    p.add_argument("--concept_npz", type=str, help="concept_directions.npz from 60_ (needs gbar, Sigma)")
    p.add_argument("--prompts", type=str, default="physics_decay_type_probe_v2_train.jsonl")
    p.add_argument("--cluster_labels", type=str, help="cluster_labels.csv (for Method A)")
    p.add_argument("--cluster_col", type=str, default="agglo_coimp_subgroup_k30",
                   help="cluster column for A1 (default k=30)")
    p.add_argument("--A3_cluster_col", type=str, default=None,
                   help="optional separate column for A3 (e.g. agglo_coimp_k14 for whole L24)")
    p.add_argument("--A1_cluster", type=str, default="16",
                   help="A1 cluster ID in --cluster_col (numeric label, 'C' prefix optional)")
    p.add_argument("--A3_cluster", type=str, default="13",
                   help="A3 cluster ID in --A3_cluster_col (e.g. 13 in k=14 = whole L24)")
    p.add_argument("--out_dir", type=str, default="data/analysis/iia_failure_diagnosis")

    p.add_argument("--model_size", type=str, default="4b")
    p.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--alpha_answer", type=str, default=" alpha")
    p.add_argument("--beta_answer", type=str, default=" beta")

    p.add_argument("--layer_configs", type=lambda s: [[int(x) for x in grp.split(",")] for grp in s.split(";")],
                   default=[[18], [22], [24], [18, 24]],
                   help="semicolon-separated, comma within a config: '18;22;24;18,24'")
    p.add_argument("--r_values", type=int, nargs="*", default=[1, 3, 8, 11])
    p.add_argument("--n_pairs", type=int, default=30)
    p.add_argument("--train_frac", type=float, default=0.6)
    p.add_argument("--steer_sweep", type=float, nargs="*", default=[-2, -1, -0.5, 0.5, 1, 2, 4])
    p.add_argument("--ridge", type=float, default=1e-3)
    p.add_argument("--tau_flip", type=float, default=0.7)
    p.add_argument("--tau_ctrl", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=0)
    return p


def main():
    args = build_parser().parse_args()
    if args.self_test:
        self_test(); return
    if not args.concept_npz:
        raise SystemExit("provide --concept_npz (from 60_) or use --self_test. See --help.")
    run_real(args)


if __name__ == "__main__":
    main()
