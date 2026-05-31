"""
60_causal_inner_product_geometry.py
====================================
Compute Park & Veitch (2024) causal inner product geometry for the α/β concept:

    γ̄  =  γ(β-token) − γ(α-token)        — concept direction (residual basis)
    Σ  =  Cov(γ)                          — covariance over unembedding distribution
    λ̄  =  Σ⁻¹ γ̄                          — dual steering direction

Saves these to a .npz that script 61_subspace_interventions_pilot.py consumes.

The covariance Σ is computed over the FULL unembedding matrix W_U (vocab rows
in residual basis). This is the natural Park & Veitch construction: causal
inner product ⟨a, b⟩_C = a^T Σ⁻¹ b makes causally-independent concept
directions approximately orthogonal.

OUTPUT
------
data/analysis/runD_v2/geometry_stage1/concept_directions.npz
  gbar       (d,)        γ̄ = γ(beta_id) − γ(alpha_id)
  Sigma      (d, d)      Cov(γ) over vocab
  Sigma_inv  (d, d)      Σ⁻¹ (ridge-regularised)
  lbar       (d,)        λ̄ = Σ⁻¹ γ̄
  alpha_id   int         single-token ID for ' alpha' (or first sub-token)
  beta_id    int         single-token ID for ' beta'  (or first sub-token)
  d_model    int
  vocab_size int

Usage (local, ~30 sec, no GPU strictly required):
  python scripts/60_causal_inner_product_geometry.py

On CSD3 (faster, with GPU):
  python -u scripts/60_causal_inner_product_geometry.py --device cuda
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_name",   default="Qwen/Qwen3-4B")
    ap.add_argument("--alpha_answer", default=" alpha",
                    help="answer string for V_h=α (default ' alpha' with leading space)")
    ap.add_argument("--beta_answer",  default=" beta")
    ap.add_argument("--device",       default="cpu",
                    help="cpu or cuda; cpu suffices since this is one matmul + eigendecomp")
    ap.add_argument("--ridge",        type=float, default=1e-6,
                    help="ridge regulariser for Σ⁻¹ as fraction of trace(Σ)/d")
    ap.add_argument("--out_dir",
                    default="data/analysis/runD_v2/geometry_stage1")
    args = ap.parse_args()

    root = Path(__file__).parent.parent
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model + tokenizer ──────────────────────────────────────────────
    logger.info(f"Loading {args.model_name} on {args.device}…")
    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.float32,
        low_cpu_mem_usage=True, trust_remote_code=True,
    ).to(args.device).eval()

    # ── Token IDs ───────────────────────────────────────────────────────────
    def first_token(s: str) -> int:
        ids = tok.encode(s, add_special_tokens=False)
        if len(ids) != 1:
            logger.warning(f"'{s}' tokenises to {len(ids)} sub-tokens: {ids}. Using first.")
        return ids[0]
    alpha_id = first_token(args.alpha_answer)
    beta_id  = first_token(args.beta_answer)
    logger.info(f"alpha_id = {alpha_id} ('{tok.decode([alpha_id])}'), "
                f"beta_id = {beta_id} ('{tok.decode([beta_id])}')")

    # ── Unembedding matrix W_U: (vocab, d_model) ────────────────────────────
    if hasattr(model, "lm_head"):
        W_U = model.lm_head.weight.detach().float().cpu().numpy()
    else:
        # tied embeddings fallback
        W_U = model.get_output_embeddings().weight.detach().float().cpu().numpy()
    V, d = W_U.shape
    logger.info(f"W_U shape: {W_U.shape} (vocab × d_model)")

    # ── γ̄ = γ(β) − γ(α) ────────────────────────────────────────────────────
    gamma_alpha = W_U[alpha_id]   # (d,)
    gamma_beta  = W_U[beta_id]
    gbar = gamma_beta - gamma_alpha
    logger.info(f"||γ̄|| = {np.linalg.norm(gbar):.4f}")

    # ── Σ = Cov(γ) over full vocab ──────────────────────────────────────────
    W_centered = W_U - W_U.mean(axis=0, keepdims=True)
    Sigma = (W_centered.T @ W_centered) / (V - 1)   # (d, d)
    Sigma = 0.5 * (Sigma + Sigma.T)                 # numerical symmetry
    logger.info(f"Σ shape: {Sigma.shape}, "
                f"trace/d = {np.trace(Sigma)/d:.4f}, "
                f"||Σ||_F = {np.linalg.norm(Sigma):.2f}")

    # ── Σ⁻¹ via eigendecomposition with ridge ───────────────────────────────
    eigvals, eigvecs = np.linalg.eigh(Sigma)
    ridge_abs = args.ridge * float(np.mean(np.diag(Sigma)))
    eigvals_r = np.maximum(eigvals, ridge_abs)
    cond = float(eigvals_r.max() / eigvals_r.min())
    Sigma_inv = (eigvecs * (1.0 / eigvals_r)) @ eigvecs.T
    logger.info(f"Σ rank: {np.linalg.matrix_rank(Sigma)}, "
                f"min eigval: {eigvals.min():.2e}, "
                f"ridge floor: {ridge_abs:.2e}, cond(Σ_inv) ≈ {cond:.2e}")

    # ── λ̄ = Σ⁻¹ γ̄ ─────────────────────────────────────────────────────────
    lbar = Sigma_inv @ gbar
    logger.info(f"||λ̄|| = {np.linalg.norm(lbar):.4f}")

    # Sanity: cos(γ̄, λ̄) in Euclidean — they're NOT identical (unless Σ ≈ I)
    cos_gl = float(gbar @ lbar / (np.linalg.norm(gbar) * np.linalg.norm(lbar) + 1e-12))
    logger.info(f"cos(γ̄, λ̄)_Euclidean = {cos_gl:.4f}  "
                f"(close to 1 if Σ near-isotropic, < 1 if anisotropic)")

    # Park & Veitch causal IP: ⟨γ̄, λ̄⟩_C = γ̄^T Σ⁻¹ λ̄ = γ̄^T Σ⁻¹ Σ⁻¹ γ̄
    # Simpler sanity: ⟨λ̄, γ̄⟩_Euclidean = γ̄^T Σ⁻¹ γ̄ — should be positive
    quad = float(gbar @ Sigma_inv @ gbar)
    logger.info(f"γ̄ᵀ Σ⁻¹ γ̄ = {quad:.4f}  (Mahalanobis squared concept norm)")

    # ── Save ─────────────────────────────────────────────────────────────────
    out_path = out_dir / "concept_directions.npz"
    np.savez(
        out_path,
        gbar=gbar.astype(np.float64),
        Sigma=Sigma.astype(np.float64),
        Sigma_inv=Sigma_inv.astype(np.float64),
        lbar=lbar.astype(np.float64),
        alpha_id=np.int64(alpha_id),
        beta_id=np.int64(beta_id),
        d_model=np.int64(d),
        vocab_size=np.int64(V),
        ridge_abs=np.float64(ridge_abs),
    )
    logger.info(f"Saved → {out_path}")
    print(f"\nConcept geometry computed:")
    print(f"  ||γ̄|| = {np.linalg.norm(gbar):.4f}")
    print(f"  ||λ̄|| = {np.linalg.norm(lbar):.4f}")
    print(f"  γ̄ᵀΣ⁻¹γ̄ = {quad:.4f}")
    print(f"  cos(γ̄, λ̄) = {cos_gl:.4f}")
    print(f"  saved to: {out_path}")


if __name__ == "__main__":
    main()
