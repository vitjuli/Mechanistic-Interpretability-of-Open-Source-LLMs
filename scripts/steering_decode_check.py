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

def fisher_axis(H, y, shrink=0.1):
    """w_res = Sigma_w^-1 delta (LDA reading axis), shrinkage-regularised (n<d)."""
    mu0=H[y==0].mean(0); mu1=H[y==1].mean(0); delta=mu1-mu0
    H0=H[y==0]-mu0; H1=H[y==1]-mu1
    Sw=(H0.T@H0 + H1.T@H1)/max(1,len(H)-2)
    Sw=(1-shrink)*Sw + shrink*(np.trace(Sw)/Sw.shape[0])*np.eye(Sw.shape[0])
    return unit(np.linalg.solve(Sw, delta))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dump",required=True); ap.add_argument("--prompts",required=True)
    ap.add_argument("--steering_csv",required=True)
    ap.add_argument("--layer",type=int,default=24); ap.add_argument("--c",type=float,default=16)
    ap.add_argument("--n",type=int,default=6); ap.add_argument("--gen",type=int,default=20)
    ap.add_argument("--c_grid", default=None, help="comma-sep c values to sweep (overrides --c); model loads once")
    ap.add_argument("--topk",type=int,default=15)
    ap.add_argument("--model",default="Qwen/Qwen3-4B",
                    help="must match the dump's meta.npz model_name (asserted when recorded)")
    ap.add_argument("--device",default="cuda")
    ap.add_argument("--out_csv", default=None, help="append structured rows here (for analysis)")
    ap.add_argument("--tag", default="", help="concept/pair label written into the CSV")
    a=ap.parse_args()

    prompts=[json.loads(l) for l in open(a.prompts)]
    _meta=np.load(f"{a.dump}/meta.npz"); y=_meta["y"].astype(int)
    class_a=str(_meta["class_a"]).strip() if "class_a" in _meta else None  # name of class0 (d_delta points class0->class1)
    if "model_name" in _meta.files:
        # The directions and sigma come from this dump; steering a different checkpoint makes
        # the generations incomparable with every other number in the paper.
        assert str(_meta["model_name"])==a.model, (
            f"model mismatch: dump captured with {str(_meta['model_name'])!r}, this run steers "
            f"{a.model!r} (pass --model {str(_meta['model_name'])})")

    tok=AutoTokenizer.from_pretrained(a.model)
    model=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=torch.bfloat16).to(a.device).eval()
    blocks=model.model.layers; L=a.layer
    res=np.load(f"{a.dump}/res_L{L:02d}.npy").astype(np.float64)
    grad=np.load(f"{a.dump}/grad_L{L:02d}.npy").astype(np.float64)
    d_delta=unit(res[y==1].mean(0)-res[y==0].mean(0)); d_usage=unit(grad.mean(0))
    d_wres=fisher_axis(res, y)
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
        lp=torch.log_softmax(logits,0)
        margin=float(lp[ids[1]]-lp[ids[0]])   # logprob(incorrect) - logprob(correct): <0 = correct preferred, >0 = FLIPPED
        return top, rank, cont, margin

    # The two class labels of THIS corpus, ordered (class0, class1) so that d_delta points
    # class0 -> class1. class_a from the dump names class0 when present.
    raw_classes = sorted({p["correct_answer"] for p in prompts if p.get("correct_answer")})
    pair_classes = ()
    if len(raw_classes) == 2:
        if class_a is not None and class_a in [str(c).strip() for c in raw_classes]:
            first = next(c for c in raw_classes if str(c).strip() == class_a)
            pair_classes = (first, next(c for c in raw_classes if c != first))
        else:
            pair_classes = (raw_classes[0], raw_classes[1])
        print(f"corpus contrast: class0={pair_classes[0]!r} class1={pair_classes[1]!r}")

    rows=[]
    c_values=[float(x) for x in a.c_grid.split(",")] if a.c_grid else [a.c]
    dirs={"delta":d_delta, "usage":d_usage, "w_res":d_wres}   # writing / use / reading
    for i in range(min(a.n,len(prompts))):
        p=prompts[i]; ca=p.get("correct_answer"); ia=p.get("incorrect_answer")
        if "tok_id_class0" in p and "y_canonical" in p:
            c0,c1=int(p["tok_id_class0"]),int(p["tok_id_class1"]); yy=int(p["y_canonical"])
            cid,iid=(c0,c1) if yy==0 else (c1,c0)
            ids=[cid,iid]; ca=ca or tok.decode([cid]); ia=tok.decode([iid])
        elif pair_classes and ca in pair_classes:
            # Contrast = the OTHER class of this corpus, i.e. the class the sweep actually
            # steers toward. NOT p["incorrect_answer"]: in the v2 particle corpora that field
            # carries an arbitrary distractor (proton for an electron/photon pair), so the
            # margin was being read against a token no direction was ever pointed at.
            ia = pair_classes[1] if ca == pair_classes[0] else pair_classes[0]
            ids=[tok.encode(ca,add_special_tokens=False)[0], tok.encode(ia,add_special_tokens=False)[0]]
        elif ia is not None:
            print(f"  [warn] prompt {i}: falling back to incorrect_answer={str(ia).strip()!r} "
                  f"-- corpus does not define a two-class contrast")
            ids=[tok.encode(ca,add_special_tokens=False)[0], tok.encode(ia,add_special_tokens=False)[0]]
        else:
            raise SystemExit(f"prompt {i}: no way to determine the contrast token")
        # push TOWARD the incorrect token. d_delta points class0->class1.
        if class_a is not None:
            s = 1.0 if str(ca).strip()==class_a else -1.0
        elif "y_canonical" in p:
            s = 1.0 if int(p["y_canonical"]) == 0 else -1.0
        else:
            s = 1.0
        # baseline once (c-independent) -> reference margin
        top,rank,cont,base_m=analyse(p["prompt"],None,ids)
        rows.append(dict(tag=a.tag, layer=L, c=0.0, prompt_idx=i,
                         correct=str(ca).strip(), incorrect=str(ia).strip(), push="baseline",
                         margin=round(base_m,3), base_margin=round(base_m,3), dmargin=0.0,
                         flipped=0, top1=top[0] if top else "", top5="|".join(top[:5]), gen=cont))
        print(f"===== prompt {i} | correct={str(ca).strip()} incorrect={str(ia).strip()} | base_margin={base_m:+.2f} =====")
        for cval in c_values:
            line=f"  c={cval:>5.1f}:"
            for name,d in dirs.items():
                top,rank,cont,margin=analyse(p["prompt"], s*(cval*sigma)*d, ids)
                fl = int(base_m<0 and margin>0)
                rows.append(dict(tag=a.tag, layer=L, c=cval, prompt_idx=i,
                                 correct=str(ca).strip(), incorrect=str(ia).strip(), push=name,
                                 margin=round(margin,3), base_margin=round(base_m,3),
                                 dmargin=round(margin-base_m,3), flipped=fl,
                                 top1=top[0] if top else "", top5="|".join(top[:5]), gen=cont))
                line+=f"  {name} dm={margin-base_m:+.2f}{'*' if fl else ''}"
            print(line)
        print()

    if a.out_csv:
        import os
        df=pd.DataFrame(rows)
        hdr=not os.path.exists(a.out_csv)
        df.to_csv(a.out_csv, mode="a", header=hdr, index=False)
        print(f"[out_csv] appended {len(df)} rows -> {a.out_csv}")

if __name__=="__main__": main()
