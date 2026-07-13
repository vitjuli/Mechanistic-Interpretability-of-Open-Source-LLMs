"""MLP crossover: MLP share of length falls and of rotation rises with depth (Δμ, decay)."""
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import scienceplots  # noqa: F401
plt.style.use(["science", "ieee", "grid", "no-latex"])
plt.rcParams.update({
    "font.size": 22, "axes.labelsize": 22, "axes.titlesize": 22,
    "xtick.labelsize": 15, "ytick.labelsize": 20, "legend.fontsize": 16,
    "lines.linewidth": 2.6, "lines.markersize": 6, "axes.linewidth": 1.1,
    "legend.title_fontsize": 14,
})
d=pd.read_csv("realized_decay.csv"); de=d[d.axis=="delta"].sort_values("layer").copy()
de=de[(de.layer>=13)&(de.layer<=33)]
de["mlp_rot"]=100*np.abs(de.rot_mlp)/(np.abs(de.rot_mlp)+np.abs(de.rot_attn)+1e-9)
de["mlp_len"]=100*np.abs(de.len_mlp)/(np.abs(de.len_mlp)+np.abs(de.len_attn)+1e-9)
L=de.layer.values
rot=pd.Series(de.mlp_rot.values).rolling(3,center=True,min_periods=1).mean()
ln =pd.Series(de.mlp_len.values).rolling(3,center=True,min_periods=1).mean()
fig,ax=plt.subplots(figsize=(14,4.2))
ax.plot(L,rot,"-o",ms=3,color="#1f77b4",label="MLP share of rotation")
ax.plot(L,ln,"-o",ms=3,color="#d62728",label="MLP share of length")
ax.axhline(50,ls=":",color="grey",lw=1)
ax.set_xlabel("layer $\\ell$"); ax.set_ylabel("MLP share (%)")
ax.set_ylim(20,100); ax.grid(True,alpha=0.3); ax.legend(fontsize=15)
fig.tight_layout(); fig.savefig("docs/figures/mlp_crossover.png",dpi=200, bbox_inches="tight")
print("wrote docs/figures/mlp_crossover.png")
