"""
Local capture: baseline logits + MLP-input activations for physics_decay_type_probe_v2.

Resumable: saves a JSONL checkpoint after each prompt. On restart, skips
already-completed prompts. Activations stored in a pre-allocated memmap.

Outputs (in data/results/local_capture/physics_decay_type_probe_v2/):
  checkpoint.jsonl          — one line per completed prompt (logits + meta)
  mlp_activations.npy       — memmap float16 (N_prompts, 36, 2560)
  mlp_activations_meta.json — layer indices, prompt order, hidden_size
  baseline_summary.json     — accuracy / mean delta at the end

Usage:
    python scripts/local_capture_v2.py
    python scripts/local_capture_v2.py --behaviour physics_decay_type_probe_v2 --split train
"""

import argparse
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.model_utils import ModelWrapper

# ── constants ────────────────────────────────────────────────────────────────
MODEL_NAME  = "Qwen/Qwen3-4B"
N_LAYERS    = 36
HIDDEN_SIZE = 2560
DTYPE       = "bfloat16"


def load_prompts(path: Path) -> list:
    prompts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))
    return prompts


def combined_forward(model: ModelWrapper, prompt: str, correct: str, incorrect: str):
    """
    Single forward pass: capture MLP-input activations at all layers
    and extract logits for correct / incorrect tokens.

    Returns:
        delta       — float, logit(correct) - logit(incorrect)
        acts        — np.ndarray float16, shape (N_LAYERS, HIDDEN_SIZE)
        l_correct   — float
        l_incorrect — float
    """
    tok = model.tokenizer
    inputs = tok(prompt, return_tensors="pt", add_special_tokens=True)
    inputs = model._move_inputs(inputs)

    blocks = model.model.model.layers
    captured = {}

    def make_hook(li):
        def hook(module, inp, out):
            x = out[0] if isinstance(out, (tuple, list)) else out
            captured[li] = x.detach()
        return hook

    handles = [
        blocks[li].post_attention_layernorm.register_forward_hook(make_hook(li))
        for li in range(N_LAYERS)
    ]
    try:
        with torch.no_grad():
            out = model.model(**inputs)
    finally:
        for h in handles:
            h.remove()

    # logits at last token position
    last_pos = inputs["attention_mask"].sum(dim=1) - 1
    logits_last = out.logits[0, last_pos[0], :]

    correct_id   = tok(correct,   add_special_tokens=False)["input_ids"][0]
    incorrect_id = tok(incorrect, add_special_tokens=False)["input_ids"][0]
    l_c = logits_last[correct_id].item()
    l_i = logits_last[incorrect_id].item()

    # activations at last (decision) token, all layers
    acts = np.zeros((N_LAYERS, HIDDEN_SIZE), dtype=np.float16)
    for li in range(N_LAYERS):
        if li in captured:
            a = captured[li][0, last_pos[0], :]   # (hidden,)
            if a.dtype == torch.bfloat16:
                a = a.float()
            acts[li] = a.cpu().numpy().astype(np.float16)

    return l_c - l_i, acts, l_c, l_i


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--behaviour", default="physics_decay_type_probe_v2")
    parser.add_argument("--split", default="train")
    args = parser.parse_args()

    behaviour, split = args.behaviour, args.split
    prompt_file = Path("data/prompts") / f"{behaviour}_{split}.jsonl"
    out_dir = Path("data/results/local_capture") / behaviour
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path  = out_dir / "checkpoint.jsonl"
    mmap_path  = out_dir / "mlp_activations.npy"
    meta_path  = out_dir / "mlp_activations_meta.json"
    summary_path = out_dir / "baseline_summary.json"

    prompts = load_prompts(prompt_file)
    N = len(prompts)
    print(f"Prompts: {N}  |  behaviour: {behaviour}  |  split: {split}")
    print(f"Output:  {out_dir}")

    # ── load checkpoint ───────────────────────────────────────────────────────
    done_ids = set()
    done_results = []
    if ckpt_path.exists():
        with open(ckpt_path) as f:
            for line in f:
                r = json.loads(line)
                done_ids.add(r["prompt_idx"])
                done_results.append(r)
        print(f"Resuming: {len(done_ids)}/{N} already done")

    # ── pre-allocate memmap ───────────────────────────────────────────────────
    mode = "r+" if mmap_path.exists() else "w+"
    mmap = np.memmap(mmap_path, dtype=np.float16, mode=mode, shape=(N, N_LAYERS, HIDDEN_SIZE))

    if not meta_path.exists():
        meta = {
            "behaviour": behaviour, "split": split,
            "n_prompts": N, "n_layers": N_LAYERS, "hidden_size": HIDDEN_SIZE,
            "shape": [N, N_LAYERS, HIDDEN_SIZE], "dtype": "float16",
            "model": MODEL_NAME,
            "hook_point": "post_attention_layernorm_output",
            "token_position": "decision (last non-padding)",
            "prompt_order": [p.get("prompt_id", i) for i, p in enumerate(prompts)],
            "created": datetime.now().isoformat(),
        }
        meta_path.write_text(json.dumps(meta, indent=2))

    todo = [i for i in range(N) if i not in done_ids]
    if not todo:
        print("All prompts already done!")
        return

    print(f"To process: {len(todo)} prompts")
    print("Loading model ...")
    model = ModelWrapper(model_name=MODEL_NAME, dtype=DTYPE, device="cpu")
    print()

    ckpt_file = open(ckpt_path, "a")
    t_start = time.time()
    times = []

    for step, idx in enumerate(todo):
        p = prompts[idx]
        t0 = time.time()

        delta, acts, l_c, l_i = combined_forward(
            model, p["prompt"], p["correct_answer"], p["incorrect_answer"]
        )

        elapsed = time.time() - t0
        times.append(elapsed)

        # save activations
        mmap[idx] = acts
        mmap.flush()

        # save checkpoint
        rec = {
            "prompt_idx":    idx,
            "prompt_id":     p.get("prompt_id", idx),
            "correct":       delta > 0,
            "delta":         round(delta, 4),
            "l_correct":     round(l_c, 4),
            "l_incorrect":   round(l_i, 4),
            "correct_answer":   p["correct_answer"],
            "incorrect_answer": p["incorrect_answer"],
            "elapsed_s":     round(elapsed, 1),
        }
        ckpt_file.write(json.dumps(rec) + "\n")
        ckpt_file.flush()
        done_results.append(rec)

        # progress
        n_done  = step + 1
        n_total = len(todo)
        avg_t   = sum(times) / len(times)
        eta_s   = avg_t * (n_total - n_done)
        eta_str = str(timedelta(seconds=int(eta_s)))
        acc_so_far = sum(r["correct"] for r in done_results) / len(done_results)
        status  = "✓" if delta > 0 else "✗"
        print(
            f"[{n_done:3d}/{n_total}] {status} idx={idx:3d}  "
            f"delta={delta:+.3f}  {elapsed:.0f}s  "
            f"acc={acc_so_far:.1%}  ETA {eta_str}"
        )

    ckpt_file.close()

    # ── summary ───────────────────────────────────────────────────────────────
    all_results = []
    with open(ckpt_path) as f:
        for line in f:
            all_results.append(json.loads(line))

    n_correct = sum(r["correct"] for r in all_results)
    deltas    = [r["delta"] for r in all_results]
    summary   = {
        "behaviour": behaviour, "split": split,
        "n_prompts": len(all_results),
        "accuracy":  round(n_correct / len(all_results), 4),
        "n_correct": n_correct,
        "mean_delta": round(float(np.mean(deltas)), 4),
        "std_delta":  round(float(np.std(deltas)), 4),
        "total_time_h": round((time.time() - t_start) / 3600, 2),
        "model": MODEL_NAME,
        "finished": datetime.now().isoformat(),
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n{'='*60}")
    print(f"DONE  accuracy={summary['accuracy']:.1%}  mean_delta={summary['mean_delta']:.3f}")
    print(f"Activations: {mmap_path}  ({N}×{N_LAYERS}×{HIDDEN_SIZE} float16)")
    print(f"Summary:     {summary_path}")


if __name__ == "__main__":
    main()
