#!/usr/bin/env python3
import sys
from pathlib import Path
import json
import torch

sys.path.insert(0, str(Path(__file__).parent))

from src.model_utils import ModelWrapper

print("="*70)
print("TOKEN POSITION TEST")
print("="*70)

prompts_file = Path("data/prompts/grammar_agreement_train.jsonl")
test_prompts = []
with open(prompts_file) as f:
    for i, line in enumerate(f):
        if i >= 3:
            break
        test_prompts.append(json.loads(line))

print("\nLoading model...")
model = ModelWrapper(
    model_name="Qwen/Qwen3-4B-Instruct-2507",
    dtype="bfloat16",
    device="cpu",
)

test_texts = [p["prompt"] for p in test_prompts]

print("\nCapturing MLP inputs with token_positions='last_5'...")
result = model.capture_mlp_inputs(
    prompts=test_texts,
    layer_range=(15, 16),
    token_positions="last_5",
)

pos_map = result["metadata"]["position_map"]

print("\nCaptured token positions:")
for i, entry in enumerate(pos_map[:15]):
    tok = model.tokenizer.decode([entry["token_id"]])
    print(
        f"[{i}] prompt={entry['prompt_idx']} "
        f"pos={entry['token_pos']} "
        f"token='{tok}'"
    )

print("\nDONE")
