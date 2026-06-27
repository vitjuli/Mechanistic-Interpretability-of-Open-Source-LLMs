"""
Decode check: is delta-steering producing GARBAGE or a VALID-but-verbose answer?
The intact metric only checks the top-1 token. This script DECODES the top-8 tokens at the answer
position (and a short greedy continuation) under baseline / delta-push / usage-push, so we can SEE
whether delta gives junk (e.g. 'the', '##x') or a valid answer not-quite-first (e.g. 'this is neutron').

Reuses 122's mechanism: push = c * sigma * unit(direction) added at the last position via a
forward_pre_hook on block[L+1]. delta = mean_res(class1) - mean_res(class0); u = mean gradient.
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
    ap.add_argument("--n",type=int,default=8); ap.add_argument("--gen",type=int,default=6)
    ap.add_argument("--model",default="Qwen/Qwen3-4B-Base"); ap.add_argument("--device",default="cuda")
    a=ap.parse_args()

    tok=AutoTokenizer.from_pretrained(a.model)
    model=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=torch.bfloat16).to(a.device).eval()
    blocks=model.model.layers
    L=a.layer
    prompts=[json.loads(l) for l in open(a.prompts)]
    y=np.load(f"{a.dump}/meta.npz")["y"].astype(int)
    res=np.load(f"{a.dump}/res_L{L:02d}.npy").astype(np.float64)
    grad=np.load(f"{a.dump}/grad_L{L:02d}.npy").astype(np.float64)
    d_delta=unit(res[y==1].mean(0)-res[y==0].mean(0))
    d_usage=unit(grad.mean(0))
    csv=pd.read_csv(a.steering_csv); sigma=float(csv[csv.layer==L]["sigma"].iloc[0])
    print(f"layer={L} c={a.c} sigma={sigma:.3f} | push norm = {a.c*sigma:.2f}\n")

    def topk(prompt, push):
        inp=tok([prompt],return_tensors="pt").to(a.device)
        h=None
        if push is not None:
            dt=torch.tensor(push,dtype=model.dtype,device=a.device)
            def pre(m_,args_): hs=args_[0].clone(); hs[0,-1,:]=hs[0,-1,:]+dt; return (hs,)
            h=blocks[L+1].register_forward_pre_hook(pre)
        try:
            with torch.no_grad(): logits=model(**inp,use_cache=False).logits[0,-1,:].float()
        finally:
            if h: h.remove()
        ids=logits.topk(8).indices.tolist()
        return [tok.decode([i]).strip() for i in ids]

    for i in range(min(a.n,len(prompts))):
        p=prompts[i]; cls=p.get("correct_answer","?").strip()
        push_d=(a.c*sigma)*d_delta; push_u=(a.c*sigma)*d_usage
        print(f"--- prompt {i} (answer={cls}) ---")
        print(f"  baseline top8: {topk(p['prompt'],None)}")
        print(f"  delta-push   : {topk(p['prompt'],push_d)}")
        print(f"  usage-push   : {topk(p['prompt'],push_u)}")

if __name__=="__main__": main()
