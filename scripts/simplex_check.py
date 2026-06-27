"""
Simplex / triangle test of the usage axis u for K=4 particles (pure geometry, no forward pass).
u(pair) = unit(mean gradient of logit(class1)-logit(class0)) at layer L, from field_dump_119/grad_L.npy.
Pairs (make_particle_pairs order, class0_vs_class1):
  e/n=gamma_n-gamma_e, e/g=gamma_g-gamma_e, e/p=gamma_p-gamma_e,
  n/g=gamma_g-gamma_n, n/p=gamma_p-gamma_n, g/p=gamma_p-gamma_g
Prediction (if u is a linear class-contrast): u(e,g)-u(e,n) ≈ u(n,g), etc. -> cosine ~1.
Runs on CSD3 login node (numpy only). NOTE: u≈gamma_bar (cos~0.99), so a high cosine is largely
EXPECTED (the unembedding contrast is linear); at intermediate layers a shared Jacobian makes it
non-trivial. Report honestly as a consistency check, not a surprise.
"""
import numpy as np
SD="data/analysis/runD_v2/particle_pairs"
PAIRS={"e_n":"electron_vs_neutron","e_g":"electron_vs_photon","e_p":"electron_vs_proton",
       "n_g":"neutron_vs_photon","n_p":"neutron_vs_proton","g_p":"photon_vs_proton"}
def unit(v): return v/(np.linalg.norm(v)+1e-12)
def cos(a,b): return float(unit(a)@unit(b))

# triangles: (lhs1 - lhs2) ?= rhs  (all = gamma differences)
TRI=[("e_g","e_n","n_g"),("e_p","e_n","n_p"),("e_p","e_g","g_p"),("n_p","n_g","g_p")]

print(f"{'layer':>5} | " + " | ".join(f"{a}-{b}~{c}" for a,b,c in TRI))
for L in [18,20,22,24,26,28,30,32]:
    u={}
    ok=True
    for k,d in PAIRS.items():
        try: u[k]=unit(np.load(f"{SD}/{d}/field_dump_119/grad_L{L:02d}.npy").astype(np.float64).mean(0))
        except Exception: ok=False; break
    if not ok: continue
    cells=[]
    for a,b,c in TRI:
        cells.append(f"{cos(u[a]-u[b], u[c]):>+.3f}".center(len(f'{a}-{b}~{c}')))
    print(f"{L:>5} | " + " | ".join(cells))
print("\ncos ~ +1 => triangle holds (linear simplex structure in u). Honest: largely expected since u≈gamma.")
