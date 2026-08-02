#!/usr/bin/env bash
# OVERNIGHT: semantic orientation of steering (thesis 3.3.1 / 3.3.2).
#   !bash colab/run_semantic_steering.sh
# Sweeps 6 particle pairs + alpha/beta + grammar, at layers 22/24/35, both push
# directions (toward the incorrect token), all axes (delta/usage/w_res). Captures
# predicted top-k tokens + full generations into ONE structured CSV for analysis:
#   -> data/analysis/runD_v2/semantic_steering/semantic_steering_all.csv
# Question: does writing/use steering shift the model to a semantically RELATED
# neighbour (e.g. electron->photon via the photoelectric effect) with coherent text,
# and is w_res inert everywhere?
set -e
cd /content/project 2>/dev/null || cd "$(dirname "$0")/.."
OUT=data/analysis/runD_v2/semantic_steering; mkdir -p "$OUT"
CSV="$OUT/semantic_steering_all.csv"; rm -f "$CSV"
# Must be the checkpoint the dumps were captured with (Qwen/Qwen3-4B). The 2026-07-11 run
# used the old script default (Qwen3-4B-Base) against Qwen3-4B dumps -- directions and sigma
# from one model, generations from another. steering_decode_check.py now asserts this when the
# dump records model_name (the particle dumps do not, hence the explicit flag here).
MODEL="Qwen/Qwen3-4B"
LAYERS="22 24 35"
C_GRID="1,2,4,8,16,32"   # push-strength sweep: graded margin response + where reading(w_res) orthogonality breaks
K=8             # prompts per class -> 16 per concept
GEN=30          # generation length

run_concept () {
  local tag="$1" dump="$2" prompts="$3"
  echo "################## $tag ##################"
  python - "$dump" <<'PY'
import numpy as np, pandas as pd, sys
dump=sys.argv[1]; y=np.load(f"{dump}/meta.npz")["y"].astype(int); rows=[]
for L in range(36):
    H=np.load(f"{dump}/res_L{L:02d}.npy").astype(np.float64)
    mu0,mu1=H[y==0].mean(0),H[y==1].mean(0); d=mu1-mu0
    X0,X1=H[y==0]-mu0,H[y==1]-mu1
    Sw=(X0.T@X0+X1.T@X1)/(len(y)-2); Sw=0.5*(Sw+Sw.T); Sw=0.9*Sw+0.1*np.diag(np.diag(Sw))
    w=np.linalg.solve(Sw,d); w/=np.linalg.norm(w)
    rows.append(dict(layer=L, sigma=float(np.std(H@w))))
pd.DataFrame(rows).to_csv("/tmp/sigma.csv",index=False)
PY
  python - "$prompts" "$K" <<'PY'
import json, sys
P=[json.loads(l) for l in open(sys.argv[1])]; K=int(sys.argv[2])
ans=sorted(set(p["correct_answer"] for p in P)); sub=[]
for a in ans[:2]:
    sub+=[p for p in P if p["correct_answer"]==a][:K]
open("/tmp/sub.jsonl","w").write("\n".join(json.dumps(p) for p in sub))
print(f"  balanced subset: {len(sub)} prompts ({ans[:2]})")
PY
  for L in $LAYERS; do
    python -u scripts/steering_decode_check.py --dump "$dump" --prompts /tmp/sub.jsonl \
        --steering_csv /tmp/sigma.csv --layer "$L" --c_grid "$C_GRID" --n 100 --gen "$GEN" --topk 15 \
        --device cuda --tag "${tag}" --out_csv "$CSV" --model "$MODEL" \
        2>&1 | tee -a "$OUT/${tag}_L${L}.txt"
  done
}

R=data/analysis/runD_v2
for pair in electron_vs_photon electron_vs_proton electron_vs_neutron neutron_vs_photon neutron_vs_proton photon_vs_proton; do
  run_concept "$pair" "$R/particles4_binary/$pair/field_dump" "data/prompts/particle_pairs/particles_$pair.jsonl"
done
run_concept "alpha_beta" "$R/B1_alpha_beta/field_dump" "data/prompts/B1_alpha_beta.jsonl"
run_concept "grammar"    "$R/B1_grammar_number/field_dump" "data/prompts/B1_grammar_number.jsonl"

echo "======== DONE ========"
echo "structured CSV -> $CSV"
python -c "import pandas as pd; d=pd.read_csv('$CSV'); s=d[d.push!='baseline'].copy(); s['adm']=s.dmargin.abs(); print('rows:',len(d),'concepts:',d.tag.nunique()); print('=== mean |dmargin| by push x c (graded margin response; watch w_res grow) ==='); print(s.pivot_table('adm','push','c','mean').round(2)); print('=== flip-rate by push x c ==='); print(s.pivot_table('flipped','push','c','mean').round(2))"

# persist to Google Drive so results survive an overnight disconnect
if [ -d /content/drive/MyDrive ]; then
  mkdir -p /content/drive/MyDrive/semantic_steering_out
  cp -r "$OUT"/* /content/drive/MyDrive/semantic_steering_out/ 2>/dev/null || true
  echo "✓ outputs copied to Drive/semantic_steering_out (safe to disconnect)"
fi
