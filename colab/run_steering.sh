#!/usr/bin/env bash
# Task 4: w-steering generations (coherent flip text). One short call from Colab:
#   !bash colab/run_steering.sh
# Builds a minimal steering_csv (layer,sigma) from the field_dump, then decodes
# generations under baseline / delta / usage / w_res push, to show whether steering
# produces coherent flipped answers (not garbage). Qualitative illustration.
set -e
cd /content/project 2>/dev/null || cd "$(dirname "$0")/.."
CPT=B1_alpha_beta
DUMP=data/analysis/runD_v2/$CPT/field_dump

python - <<'PY'
import numpy as np, pandas as pd
dump="data/analysis/runD_v2/B1_alpha_beta/field_dump"
y=np.load(f"{dump}/meta.npz")["y"].astype(int); rows=[]
for L in range(36):
    H=np.load(f"{dump}/res_L{L:02d}.npy").astype(np.float64)
    mu0,mu1=H[y==0].mean(0),H[y==1].mean(0); d=mu1-mu0
    X0,X1=H[y==0]-mu0,H[y==1]-mu1
    Sw=(X0.T@X0+X1.T@X1)/(len(y)-2); Sw=0.5*(Sw+Sw.T); Sw=0.9*Sw+0.1*np.diag(np.diag(Sw))
    w=np.linalg.solve(Sw,d); w/=np.linalg.norm(w)
    rows.append(dict(layer=L, sigma=float(np.std(H@w))))
pd.DataFrame(rows).to_csv("steering_sigma_ab.csv",index=False)
print("wrote steering_sigma_ab.csv")
PY

python -u scripts/steering_decode_check.py \
    --dump "$DUMP" \
    --prompts data/prompts/$CPT.jsonl \
    --steering_csv steering_sigma_ab.csv \
    --layer 24 --c 16 --n 6 --gen 30 --device cuda \
    2>&1 | tee data/analysis/runD_v2/$CPT/steering_decode_L24.txt
