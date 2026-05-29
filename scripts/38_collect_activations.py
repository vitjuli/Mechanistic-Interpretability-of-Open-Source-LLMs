"""
Script 38: Collect raw transcoder feature activations for all prompts.

Runs one forward pass per prompt (no intervention), records a_k^(ℓ)(p)
for every feature in the 227-feature RunD v2 set (layers L10–L25).

This is the prerequisite for computing ICC (Condition I.a, script 38b).

Outputs (in --out_dir):
  activation_matrix.npy     float32, shape (n_features, n_prompts)
  feature_ids.txt           one feature_id per line (row order)
  prompt_idxs.txt           one prompt_idx per line (column order)

Usage (CSD3, ~10-15 min on Ampere):
    python scripts/38_collect_activations.py \\
        --behaviour physics_decay_type_probe_v2 \\
        --split train \\
        --clustering_dir data/analysis/runD_v2/clustering_full \\
        --out_dir data/analysis/runD_v2/activations \\
        --device cuda
"""
import json, yaml, argparse, sys, logging, csv as csvlib
from pathlib import Path

import torch
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.model_utils import ModelWrapper
from src.transcoder import load_transcoder_set

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--behaviour",       default="physics_decay_type_probe_v2")
    parser.add_argument("--split",           default="train")
    parser.add_argument("--clustering_dir",  type=Path, required=True)
    parser.add_argument("--out_dir",         type=Path, required=True)
    parser.add_argument("--device",          default="cuda")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load feature list ─────────────────────────────────────────────────
    with open(args.clustering_dir / "cluster_labels.csv") as f:
        rows = list(csvlib.DictReader(f))
    feature_ids = [r["feature_id"] for r in rows]  # ordered list, 227 features

    def parse_feat(fid):
        layer = int(fid.split("_")[0][1:])
        feat  = int(fid.split("_F")[1])
        return layer, feat

    feat_layer_idx = [parse_feat(fid) for fid in feature_ids]
    all_layers = sorted(set(l for l, _ in feat_layer_idx))
    log.info(f"Features: {len(feature_ids)} | Layers: {all_layers}")

    # Map: layer → list of (feature_id_row_index, feat_idx_in_transcoder)
    from collections import defaultdict
    layer_to_feat_rows: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for row_i, (layer, feat) in enumerate(feat_layer_idx):
        layer_to_feat_rows[layer].append((row_i, feat))

    # ── Load prompts ──────────────────────────────────────────────────────
    prompts_path = ROOT / "data/prompts" / f"{args.behaviour}_{args.split}.jsonl"
    prompts = []
    with open(prompts_path) as f:
        for line in f:
            prompts.append(json.loads(line.strip()))
    n_prompts = len(prompts)
    log.info(f"Prompts: {n_prompts}")

    # ── Load model + transcoders ──────────────────────────────────────────
    tc_cfg = yaml.safe_load(open(ROOT / "configs/transcoder_config.yaml"))
    model_size = tc_cfg.get("model_size", "4b")
    model_name = tc_cfg["transcoders"][model_size]["model_name"]

    log.info(f"Loading model: {model_name}")
    model = ModelWrapper(model_name=model_name, dtype="bfloat16", device="auto",
                         trust_remote_code=True)
    model.model.eval()
    try:
        device = next(model.model.parameters()).device
    except StopIteration:
        device = torch.device(args.device)
    log.info(f"Model on: {device}")

    tc_set = load_transcoder_set(model_size=model_size, device=device,
                                  dtype=torch.bfloat16, lazy_load=True, layers=all_layers)
    log.info("Transcoders loaded.")

    # ── Activation matrix: (n_features, n_prompts) ───────────────────────
    act_matrix = np.zeros((len(feature_ids), n_prompts), dtype=np.float32)
    prompt_idxs = []

    for p_i, p in enumerate(prompts):
        prompt_idx = p.get("prompt_idx", p_i)
        prompt_idxs.append(int(prompt_idx))

        inputs = model.tokenize([p["prompt"]])
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Collect post_attention_layernorm outputs at all needed layers simultaneously
        mlp_inputs: dict[int, torch.Tensor] = {}

        def make_hook(layer_idx_):
            def hook(module, inp, out):
                t = out[0] if isinstance(out, tuple) else out
                mlp_inputs[layer_idx_] = t.detach()[:, -1, :]  # (1, H), last token
            return hook

        hooks = []
        for layer_idx in all_layers:
            block = model.model.model.layers[layer_idx]
            h = block.post_attention_layernorm.register_forward_hook(make_hook(layer_idx))
            hooks.append(h)

        try:
            with torch.no_grad():
                model.model(**inputs, use_cache=False)
        finally:
            for h in hooks:
                h.remove()

        # Encode through transcoders → get feature activations
        for layer_idx, mlp_in in mlp_inputs.items():
            tc = tc_set[layer_idx]
            with torch.no_grad():
                feats = tc.encode(mlp_in.to(tc.dtype)).float()  # (1, d_tc)
            feats_np = feats[0].cpu().numpy()  # (d_tc,)

            for row_i, feat_idx in layer_to_feat_rows[layer_idx]:
                act_matrix[row_i, p_i] = feats_np[feat_idx]

        if (p_i + 1) % 50 == 0:
            log.info(f"  {p_i + 1}/{n_prompts} prompts done")

    # ── Save ──────────────────────────────────────────────────────────────
    np.save(args.out_dir / "activation_matrix.npy", act_matrix)
    (args.out_dir / "feature_ids.txt").write_text("\n".join(feature_ids) + "\n")
    (args.out_dir / "prompt_idxs.txt").write_text(
        "\n".join(str(x) for x in prompt_idxs) + "\n"
    )
    log.info(f"Saved activation_matrix.npy shape={act_matrix.shape}")
    log.info(f"Saved feature_ids.txt ({len(feature_ids)} entries)")
    log.info(f"Saved prompt_idxs.txt ({len(prompt_idxs)} entries)")
    log.info(f"Non-zero entries: {(act_matrix != 0).sum()} / {act_matrix.size}")
    log.info(f"Mean activation: {act_matrix.mean():.4f}  std: {act_matrix.std():.4f}")


if __name__ == "__main__":
    main()
