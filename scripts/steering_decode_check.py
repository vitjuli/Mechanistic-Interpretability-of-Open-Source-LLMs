"""
Decode check: is delta-steering GARBAGE or a VALID-but-verbose answer?
intact only checks top-1. This script shows the FULL picture under baseline / delta-push / usage-push:
  (1) a ~20-token greedy GENERATION (the actual sentence the steered model produces),
  (2) the top-15 tokens at the answer position,
  (3) the RANK of the two class tokens (correct/incorrect particle) in the full vocab ordering
      -- so we see if the particle is rank 1, rank 15, rank 50, or buried at rank 8000.

push = c * sigma * unit(direction) added at the last position via a pre-hook on block[L+1].
delta = mean_res(class1) - mean_res(class0);  u = mean gradient.
"""
import argparse, json
import numpy as np, torch, pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

def unit(v):
    v=np.asarray(v,np.float64); return v/(np.linalg.norm(v)+1e-12)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dump",required=True); ap.add_argument("--prompts",required=True)
    ap.add_argument("--steering_csv",required=True)
    ap.add_argument("--layer",type=int,default=24); ap.add_argument("--c",type=float,default=16)
    ap.add_argument("--n",type=int,default=6); ap.add_argument("--gen",type=int,default=20)
    ap.add_argument("--topk",type=int,default=15)
    ap.add_argument("--model",default="Qwen/Qwen3-4B-Base"); ap.add_argument("--device",default="cuda")
    a=ap.parse_args()

    tok=AutoTokenizer.from_pretrained(a.model)
    model=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=torch.bfloat16).to(a.device).eval()
    blocks=model.model.layers; L=a.layer
    prompts=[json.loads(l) for l in open(a.prompts)]
    y=np.load(f"{a.dump}/meta.npz")["y"].astype(int)
    res=np.load(f"{a.dump}/res_L{L:02d}.npy").astype(np.float64)
    grad=np.load(f"{a.dump}/grad_L{L:02d}.npy").astype(np.float64)
    d_delta=unit(res[y==1].mean(0)-res[y==0].mean(0)); d_usage=unit(grad.mean(0))
    csv=pd.read_csv(a.steering_csv); sigma=float(csv[csv.layer==L]["sigma"].iloc[0])
    print(f"layer={L} c={a.c} sigma={sigma:.3f} | push norm={a.c*sigma:.2f} | gen={a.gen} tokens\n")

    def hook(push):
        dt=torch.tensor(push,dtype=model.dtype,device=a.device)
        def pre(m_,args_): hs=args_[0].clone(); hs[0,-1,:]=hs[0,-1,:]+dt; return (hs,)
        return blocks[L+1].register_forward_pre_hook(pre)

    def analyse(prompt, push, ids):
        inp=tok([prompt],return_tensors="pt").to(a.device)
        h=hook(push) if push is not None else None
        try:
            with torch.no_grad():
                logits=model(**inp,use_cache=False).logits[0,-1,:].float()
                gen=model.generate(**inp,max_new_tokens=a.gen,do_sample=False,
                                   pad_token_id=tok.eos_token_id)
        finally:
            if h: h.remove()
        top=[tok.decode([i]).strip() for i in logits.topk(a.topk).indices.tolist()]
        order=logits.argsort(descending=True).tolist()
        rank={tok.decode([t]).strip(): order.index(t)+1 for t in ids}
        cont=tok.decode(gen[0][inp.input_ids.shape[1]:],skip_special_tokens=True).replace("\n"," ")
        return top, rank, cont

    for i in range(min(a.n,len(prompts))):
        p=prompts[i]; ca=p.get("correct_answer"); ia=p.get("incorrect_answer")
        ids=[tok.encode(ca,add_special_tokens=False)[0], tok.encode(ia,add_special_tokens=False)[0]]
        print(f"===== prompt {i} | correct={ca.strip()} incorrect={ia.strip()} =====")
        for name,push in [("baseline",None),("delta",(a.c*sigma)*d_delta),("usage",(a.c*sigma)*d_usage)]:
            top,rank,cont=analyse(p["prompt"],push,ids)
            print(f"  [{name}] class-ranks={rank}")
            print(f"     top{a.topk}: {top}")
            print(f"     gen: {cont!r}")
        print()

if __name__=="__main__": main()
