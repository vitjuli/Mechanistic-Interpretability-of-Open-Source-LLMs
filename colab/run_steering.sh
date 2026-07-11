#!/usr/bin/env bash
# Task 4: w-steering generations, BOTH directions (alpha->beta AND beta->alpha).
#   !bash colab/run_steering.sh
# Push is signed TOWARD the incorrect class per prompt (patched script), so beta
# prompts are steered toward alpha (the HARD direction, given the beta-suppressor
# circuit / beta-prompt asymmetry). Lets us check: is w_res inert BOTH ways, and
# is beta->alpha harder than alpha->beta (the documented asymmetry)?
set -e
cd /content/project 2>/dev/null || cd "$(dirname "$0")/.."
CPT=B1_alpha_beta
DUMP=data/analysis/runD_v2/$CPT/field_dump

python - <<'PY'
import numpy as np, pandas as pd, json
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
P=[json.loads(l) for l in open("data/prompts/B1_alpha_beta.jsonl")]
a=[p for p in P if p["correct_answer"]=="alpha"][:6]
b=[p for p in P if p["correct_answer"]=="beta"][:6]
open("prompts_alpha6.jsonl","w").write("\n".join(json.dumps(p) for p in a))
open("prompts_beta6.jsonl","w").write("\n".join(json.dumps(p) for p in b))
print("wrote sigma + alpha6 + beta6")
PY

echo "########## ALPHA prompts  ->  push toward BETA (easy direction) ##########"
python -u scripts/steering_decode_check.py --dump "$DUMP" --prompts prompts_alpha6.jsonl \
    --steering_csv steering_sigma_ab.csv --layer 24 --c 16 --n 6 --gen 30 --device cuda \
    2>&1 | tee data/analysis/runD_v2/$CPT/steering_decode_alpha.txt

echo "########## BETA prompts  ->  push toward ALPHA (hard direction) ##########"
python -u scripts/steering_decode_check.py --dump "$DUMP" --prompts prompts_beta6.jsonl \
    --steering_csv steering_sigma_ab.csv --layer 24 --c 16 --n 6 --gen 30 --device cuda \
    2>&1 | tee data/analysis/runD_v2/$CPT/steering_decode_beta.txt
