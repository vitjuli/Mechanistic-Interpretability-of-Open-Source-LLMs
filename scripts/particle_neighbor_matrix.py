"""
Particle neighbour matrix: for prompts of each particle (baseline, NO steering), rank the 4 main
particle tokens at the answer position. Off-diagonal min rank = the particle's preferred 'switch-to'
neighbour. Tests whether each particle has a consistent partner (e.g. electron<->photon EM,
neutron<->proton nuclear). Forward-only.
"""
import json, argparse
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict

ap=argparse.ArgumentParser()
ap.add_argument("--prompts",default="data/prompts/physics_internal_candidate_selection_v2_train.jsonl")
ap.add_argument("--model",default="Qwen/Qwen3-4B-Base"); ap.add_argument("--device",default="cuda")
ap.add_argument("--n",type=int,default=447)
a=ap.parse_args()

tok=AutoTokenizer.from_pretrained(a.model)
model=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=torch.bfloat16).to(a.device).eval()
PARTS=["electron","neutron","photon","proton"]
pid={p:tok.encode(" "+p,add_special_tokens=False)[0] for p in PARTS}
print("token ids:",pid)

prompts=[json.loads(l) for l in open(a.prompts)][:a.n]
ranks=defaultdict(lambda:defaultdict(list))   # correct -> particle -> [ranks]
for p in prompts:
    c=str(p.get("correct_answer","")).strip()
    if c not in PARTS: continue
    inp=tok([p["prompt"]],return_tensors="pt").to(a.device)
    with torch.no_grad(): logits=model(**inp,use_cache=False).logits[0,-1,:].float()
    order=logits.argsort(descending=True).tolist()
    for q in PARTS:
        ranks[c][q].append(order.index(pid[q])+1)

hdr="correct \\ rankof"
print("\n"+f"{hdr:>16} | "+" ".join(f"{q:>9}" for q in PARTS)+" | preferred")
for c in PARTS:
    if not ranks[c]: continue
    med={q:int(np.median(ranks[c][q])) for q in PARTS}
    pref=min((q for q in PARTS if q!=c), key=lambda q:med[q])
    row=" ".join(f"{med[q]:>9}" for q in PARTS)
    print(f"{c:>14} | {row} | -> {pref}  (n={len(ranks[c]['electron'])})")
print("\nLower rank = more likely. Off-diagonal min = preferred switch-to neighbour.")
