"""
Test the rotation hypothesis: does w_res^(ℓ+1) ≈ R · w_res^(ℓ) with R approximately
a rotation restricted to a low-dimensional plane?

We have 13 w_res vectors (postL14..postL25 + final) in d=2560.

Approach:
  (1) Consecutive cosines — how stable is the axis layer-to-layer?
  (2) SVD of the W matrix — effective dimensionality of the family
  (3) If low-rank: project onto dominant plane, check angles
  (4) Procrustes-style test: how much of W^(ℓ+1) is explained by rotating W^(ℓ) in
      the plane spanned by them?
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/julia/Desktop/courses/thesis/project")
GEOM = ROOT / "data/analysis/runD_v2/geometry_stage1"

TAPS = ["postL14", "postL15", "postL16", "postL17", "postL18", "postL19", "postL20",
        "postL21", "postL22", "postL23", "postL24", "postL25", "final"]
SHORT = ["L14", "L15", "L16", "L17", "L18", "L19", "L20",
         "L21", "L22", "L23", "L24", "L25", "fin"]

W = np.stack([np.load(GEOM / f"w_res_{t}.npy").astype(np.float64) for t in TAPS])
# Normalize (they should already be unit, but just in case)
W = W / np.linalg.norm(W, axis=1, keepdims=True)
n, d = W.shape
print(f"W shape: {W.shape}  (n={n} layers, d={d})")
print("=" * 78)

# ── (1) Consecutive cosines ──────────────────────────────────────────────────
print("\n(1) CONSECUTIVE COSINES — how stable is w_res across adjacent layers?\n")
print(f"{'pair':<14} {'cos':>8}  {'angle deg':>10}")
print("-" * 36)
cons_cos = []
for i in range(n - 1):
    c = float(W[i] @ W[i + 1])
    angle_deg = np.degrees(np.arccos(np.clip(abs(c), 0, 1)))
    cons_cos.append(c)
    print(f"{SHORT[i]:>5}→{SHORT[i+1]:<5}  {c:+.4f}  {angle_deg:>8.2f}°")
print(f"\nmedian = {np.median(cons_cos):+.4f}   mean = {np.mean(cons_cos):+.4f}")

# Random-baseline reference: |cos| ≈ √(2/(πd)) ≈ 0.016 for d=2560
rand_exp = np.sqrt(2 / (np.pi * d))
rand_p95 = 1.96 / np.sqrt(d)
print(f"random baseline in d={d}: E|cos|≈{rand_exp:.4f}, p95≈{rand_p95:.4f}")

# ── (2) SVD of W matrix — effective dimensionality ───────────────────────────
print("\n" + "=" * 78)
print("\n(2) SVD OF W MATRIX — does the family live in a low-rank subspace?\n")
U, S, Vt = np.linalg.svd(W, full_matrices=False)
print("Singular values:")
for i, s in enumerate(S):
    print(f"  σ_{i+1:>2} = {s:.4f}")

cumvar = np.cumsum(S ** 2) / np.sum(S ** 2)
print("\nCumulative variance explained:")
for i, c in enumerate(cumvar):
    print(f"  top-{i+1:>2}: {c:.4f}  ({c*100:.1f}%)")

pr_full = (S.sum() ** 2) / (S ** 2).sum()
pr_eig = ((S ** 2).sum() ** 2) / ((S ** 2) ** 2).sum()
print(f"\nParticipation ratio (σ): {pr_full:.2f}  (1=rank-1, n=isotropic)")
print(f"Participation ratio (σ²): {pr_eig:.2f}  (effective rank by mass)")

# ── (3) Project onto dominant 2D plane — is the trajectory a 2D curve? ───────
print("\n" + "=" * 78)
print("\n(3) PROJECTION ONTO DOMINANT 2D PLANE — is the trajectory a curve?\n")
plane = Vt[:2].T  # (d, 2) — first two principal directions of W
W_proj_2d = W @ plane  # (n, 2)
print(f"variance of W captured by top-2 plane: {cumvar[1]*100:.1f}%")

# Norm of each w_res INSIDE the plane (squared cosine with the plane)
plane_norms_sq = (W_proj_2d ** 2).sum(axis=1)
print("\nfraction of each w_res lying in the dominant 2D plane:")
print(f"{'layer':<10} {'frac (cos²)':>12}  {'angle in plane':>16}")
print("-" * 42)
angles_in_plane = []
for i, t in enumerate(SHORT):
    ang = np.degrees(np.arctan2(W_proj_2d[i, 1], W_proj_2d[i, 0]))
    angles_in_plane.append(ang)
    print(f"{t:<10} {plane_norms_sq[i]:>12.4f}  {ang:>+14.2f}°")

# Are the angles monotone? (rotation signature)
da = np.diff(angles_in_plane)
# Unwrap angles to detect monotone progression
da_unwrapped = np.array([d if d > -180 else d + 360 for d in da])
da_unwrapped = np.array([d if d < 180 else d - 360 for d in da_unwrapped])
print(f"\nangle differences (consecutive layer):")
for i, d_ in enumerate(da_unwrapped):
    print(f"  {SHORT[i]}→{SHORT[i+1]}: Δθ = {d_:+.2f}°")
print(f"\nmedian Δθ = {np.median(da_unwrapped):+.2f}°   mean = {np.mean(da_unwrapped):+.2f}°")
n_pos = (da_unwrapped > 0).sum()
n_neg = (da_unwrapped < 0).sum()
print(f"monotone signature: {n_pos} positive, {n_neg} negative steps")

# ── (4) Procrustes — does a single 2D rotation fit each consecutive pair? ────
print("\n" + "=" * 78)
print("\n(4) PROCRUSTES — does the rotation hypothesis fit the full trajectory?\n")

# For each consecutive pair, compute the 2D rotation in span(w_i, w_{i+1})
# that minimizes ||w_{i+1} - R w_i||. This is trivial: the rotation by their angle.
# More interesting: project all w_res onto a fixed plane, check if they trace a circle.

# Build orthonormal basis for the dominant plane
plane_basis = plane / np.linalg.norm(plane, axis=0)  # (d, 2)
proj_norm = W_proj_2d / np.linalg.norm(W_proj_2d, axis=1, keepdims=True)
# Each proj_norm is a unit 2-vector
radii = np.linalg.norm(W_proj_2d, axis=1)
print(f"Radii in plane (should be ≈ constant if pure rotation): "
      f"mean={radii.mean():.3f} std={radii.std():.3f} (std/mean={radii.std()/radii.mean():.3f})")

# Reconstruct each w_res from the plane-only 2D embedding (using sqrt(plane_norms_sq) on the plane)
reconstr = np.zeros_like(W)
for i in range(n):
    reconstr[i] = W_proj_2d[i, 0] * plane_basis[:, 0] + W_proj_2d[i, 1] * plane_basis[:, 1]
recon_err = np.linalg.norm(W - reconstr, axis=1)
print(f"\nReconstruction error from 2D plane projection per layer (1 = total loss):")
for i, t in enumerate(SHORT):
    print(f"  {t}: ||w - w_proj|| = {recon_err[i]:.3f}  (off-plane mass = {1 - plane_norms_sq[i]:.3f})")

# ── (5) Average pairwise cosine before/after best rotation ───────────────────
print("\n" + "=" * 78)
print("\n(5) HOW WELL DOES THE 2D PLANE EXPLAIN W?\n")

# Test: how well does 2D-projection preserve cosines?
print(f"{'layer pair':<14} {'cos full d':>10}  {'cos in plane':>12}")
print("-" * 40)
for i in range(n - 1):
    full = float(W[i] @ W[i + 1])
    plane_i = W_proj_2d[i] / np.linalg.norm(W_proj_2d[i])
    plane_j = W_proj_2d[i + 1] / np.linalg.norm(W_proj_2d[i + 1])
    in_plane = float(plane_i @ plane_j)
    print(f"{SHORT[i]:>5}→{SHORT[i+1]:<5}  {full:+.4f}  {in_plane:+.4f}")

print("\n" + "=" * 78)
print("\nVERDICT — interpretation guide")
print("=" * 78)
print(f"""
• cumulative variance top-2 = {cumvar[1]*100:.1f}%
• PR(σ) = {pr_full:.2f}  (n=13 layers; 1=rank-1, 13=isotropic)
• consecutive cos median = {np.median(cons_cos):+.4f}  (random ≈ 0)
• in-plane angle progression: {np.median(da_unwrapped):+.1f}° median step

If cumvar top-2 > 95% AND consecutive cos high (0.7+) AND angles monotone → STRONG
support for rotation hypothesis: w_res traces a curve in a 2D plane across depth.

If cumvar top-2 ~50-80% → trajectory is mostly-but-not-cleanly in a low-d subspace;
rotation hypothesis partially holds with off-plane noise.

If cumvar top-2 < 50% OR PR(σ) > 6 → trajectory NOT in a low-dim plane; rotation
hypothesis rejected, w_res evolves more chaotically.

If consecutive cos < random baseline ({rand_exp:.3f}) → layers are independent draws.
""")
