#!/usr/bin/env python3
import torch
from src.transcoder import load_transcoder_set

def main():
    print("Loading transcoder for layer 15...")
    transcoder_set = load_transcoder_set(
        model_size="4b",
        layers=[15],
        device=torch.device("cpu"),
        lazy_load=False,
    )
    transcoder = transcoder_set[15]

    print("\n" + "="*60)
    print("CHECK 1: activation_function class")
    print("="*60)
    af = getattr(transcoder, "activation_function", None)
    print("activation_function:", af)
    print("type:", type(af))
    has_thr = hasattr(af, "threshold")
    print("has .threshold:", has_thr)

    thr = None
    if has_thr:
        thr = af.threshold
        # thr might be python float, 0-d tensor, or vector
        if isinstance(thr, torch.Tensor):
            print("threshold tensor shape:", tuple(thr.shape))
            print("threshold dtype:", thr.dtype)
            if thr.numel() == 1:
                print("threshold value (scalar):", thr.item())
        else:
            print("threshold (non-tensor):", thr)

    print("\n" + "="*60)
    print("CHECK 2: state_dict keys")
    print("="*60)
    sd = transcoder.state_dict()
    thr_keys = [k for k in sd.keys() if ("thresh" in k.lower()) or ("theta" in k.lower())]
    print("keys containing 'thresh' or 'theta':", thr_keys if thr_keys else "(none)")
    print("\nAll state_dict keys:")
    for k in sorted(sd.keys()):
        v = sd[k]
        shp = tuple(v.shape) if hasattr(v, "shape") else "N/A"
        print(f"  {k:45s} : {shp}")

    print("\n" + "="*60)
    print("Threshold stats (only if per-feature vector)")
    print("="*60)
    if isinstance(thr, torch.Tensor) and thr.ndim >= 1 and thr.numel() >= 10:
        print("shape:", tuple(thr.shape))
        print("min/max/mean/median/std:",
              thr.min().item(),
              thr.max().item(),
              thr.mean().item(),
              thr.median().item(),
              thr.std().item())

        print("\nTop 10 LOWEST:")
        vals, idx = thr.topk(k=10, largest=False)
        for j, (v, i) in enumerate(zip(vals, idx), 1):
            print(f"  {j:2d}. feature {i.item():6d} : {v.item():.6f}")

        print("\nTop 10 HIGHEST:")
        vals, idx = thr.topk(k=10, largest=True)
        for j, (v, i) in enumerate(zip(vals, idx), 1):
            print(f"  {j:2d}. feature {i.item():6d} : {v.item():.6f}")
    else:
        # scalar or missing threshold -> effectively ReLU threshold=0 (or shared scalar)
        if thr is None:
            print("No threshold attribute found -> likely plain ReLU or equivalent.")
        elif isinstance(thr, torch.Tensor) and thr.numel() == 1:
            print(f"Scalar threshold = {thr.item()} -> JumpReLU degenerates to ReLU if 0.")
        else:
            print(f"Threshold exists but not a per-feature vector: {thr}")

if __name__ == "__main__":
    main()
