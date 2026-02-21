"""
Run intervention experiments to validate circuit hypotheses using transcoders.

Implements three types of causal interventions:
1. Feature ablation: Zero/suppress specific transcoder features
2. Activation patching: Swap features between prompt pairs
3. Feature steering: Inject features to modify behaviour

Uses pre-trained transcoders from: https://github.com/safety-research/circuit-tracer

Usage:
    python scripts/07_run_interventions.py
    python scripts/07_run_interventions.py --experiment ablation
    python scripts/07_run_interventions.py --experiment patching --n_prompts 30
"""

import json
import yaml
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
import sys
from tqdm import tqdm
import dataclasses
from dataclasses import dataclass, asdict, field
from datetime import datetime
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_utils import ModelWrapper
from src.transcoder import load_transcoder_set, TranscoderSet

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_answer_tokens(prompt_data: Dict) -> Tuple[str, str]:
    """
    Get answer tokens from prompt data, supporting multiple schema formats.
    
    Supports:
    - correct_answer/incorrect_answer
    - answer_matching/answer_not_matching  
    - correct/incorrect
    
    Returns:
        (correct_token, incorrect_token)
    """
    if "correct_answer" in prompt_data and "incorrect_answer" in prompt_data:
        return prompt_data["correct_answer"], prompt_data["incorrect_answer"]
    if "answer_matching" in prompt_data and "answer_not_matching" in prompt_data:
        return prompt_data["answer_matching"], prompt_data["answer_not_matching"]
    if "correct" in prompt_data and "incorrect" in prompt_data:
        return prompt_data["correct"], prompt_data["incorrect"]
    raise KeyError(
        f"Can't find answer fields in prompt_data. "
        f"Available keys: {list(prompt_data.keys())}"
    )


def ensure_single_token(model: ModelWrapper, tok: str) -> int:
    """
    Ensure token is single-token and return its ID.
    
    Args:
        model: Model wrapper with tokenizer
        tok: Token string
    
    Returns:
        Token ID
    
    Raises:
        ValueError: If token is not single-token
    """
    ids = model.tokenizer.encode(tok, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(
            f"Answer must be single-token! Got: {tok!r} -> token_ids={ids}"
        )
    return ids[0]


def to_numpy(x: torch.Tensor) -> np.ndarray:
    """
    Safely convert torch tensor to numpy array.
    Detaches from autograd graph and handles bfloat16.
    
    Args:
        x: PyTorch tensor
        
    Returns:
        NumPy array
    """
    x = x.detach()  # CRITICAL: Detach from computation graph!
    if x.dtype == torch.bfloat16:
        x = x.float()  # Convert bfloat16 → float32
    return x.cpu().numpy()


@dataclass
class InterventionResult:
    """Store results from a single intervention experiment."""
    experiment_type: str  # "ablation", "patching", "steering"
    prompt_idx: int
    layer: int
    feature_indices: List[int]
    baseline_logit_diff: float  # Baseline margin (correct - incorrect)
    intervened_logit_diff: float  # Margin after intervention
    effect_size: float  # SIGNED: intervened - baseline (keeps direction!)
    abs_effect_size: float  # Absolute value for magnitude
    relative_effect: float  # abs_effect_size / abs(baseline)
    sign_flipped: bool  # Whether intervention flipped the model's prediction
    metadata: Dict = field(default_factory=dict)  # Extensible storage, avoids shared state


def load_config(config_path: str = "configs/experiment_config.yaml") -> Dict:
    """Load experiment configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_transcoder_config(config_path: str = "configs/transcoder_config.yaml") -> Dict:
    """Load transcoder configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_prompts(
    prompt_path: Path, 
    behaviour: str, 
    split: str = "train",
    prompts_file: Optional[str] = None
) -> List[Dict]:
    """
    Load prompts from JSONL file.
    
    Args:
        prompt_path: Base prompts directory
        behaviour: Behaviour name  
        split: Data split
        prompts_file: Optional custom JSONL path (overrides default)
    """
    # Use custom file if provided
    if prompts_file:
        file_path = Path(prompts_file)
        logger.info(f"Loading prompts from custom file: {file_path}")
    else:
        file_path = prompt_path / f"{behaviour}_{split}.jsonl"
    
    if not file_path.exists():
        raise FileNotFoundError(f"Prompts file not found: {file_path}")
    
    prompts = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))
    
    return prompts


def load_attribution_graph(
    results_path: Path,
    behaviour: str,
    split: str = "train",
    n_prompts: Optional[int] = None,
) -> Dict:
    """
    Load attribution graph to get top attributed features.
    
    Args:
        results_path: Base results directory
        behaviour: Behaviour name (e.g., "grammar_agreement")
        split: Data split ("train" or "test")
        n_prompts: Number of prompts (e.g., 20, 80). If None, tries to load without suffix first.
    
    Returns:
        Graph data dict or None if not found
    """
    base_path = results_path / "attribution_graphs" / behaviour

    # IMPORTANT: if n_prompts is provided, require the suffixed graph.
    # Silent fallback to old naming easily loads the WRONG graph type (or wrong N)
    # and makes sign-based ablation collapse to 0 -> fallback to features [0..k].
    if n_prompts is not None:
        graph_file = base_path / f"attribution_graph_{split}_n{n_prompts}.json"
        if not graph_file.exists():
            raise FileNotFoundError(
                f"Expected attribution graph not found: {graph_file}. "
                f"Run script 06 with matching --n_prompts (note: 06 may save n_prompts_used). "
                f"Tip: in 07 pass --graph_n_prompts equal to the n_used printed by 06."
            )
        logger.info(f"Loading attribution graph: {graph_file}")
        with open(graph_file, "r") as f:
            return json.load(f)

    # If n_prompts is None, allow legacy name as a fallback.
    graph_file = base_path / f"attribution_graph_{split}.json"
    if graph_file.exists():
        logger.info(f"Loading legacy attribution graph: {graph_file}")
        with open(graph_file, "r") as f:
            return json.load(f)

    logger.warning(f"Attribution graph not found: {graph_file}")
    return None


def get_top_attributed_features(
    graph_data: Dict,
    n_features: int = 10,
) -> List[Tuple[int, int, float, dict]]:
    """
    Extract top attributed features from graph.

    Detects graph type from graph_attrs.union_params or metadata.graph_type:
    - per_prompt_union: uses mean_abs_score_conditional (or specific_score)
    - correlation (old): uses abs_corr

    Returns:
        List of (layer, feature_idx, attribution_score, node_dict) tuples
    """
    if not graph_data or "nodes" not in graph_data:
        logger.warning("Graph data is None or missing 'nodes' key")
        return []

    # Detect graph type
    meta = graph_data.get("metadata", {}) or {}
    gattrs = graph_data.get("graph_attrs", {}) or {}
    graph_type = meta.get("graph_type", None)
    if graph_type is None and "union_params" in gattrs:
        graph_type = "per_prompt_union"

    features = []
    for node in graph_data["nodes"]:
        if node.get("type") != "feature":
            continue

        if graph_type == "per_prompt_union":
            # Fields written by 06_build_attribution_graph.py union method
            # CRITICAL: Prioritize specific_score (prompt-specific) over raw magnitude activation
            score = node.get("specific_score", None)
            if score is None:
                score = node.get("mean_abs_score_conditional", None)
            if score is None:
                score = abs(node.get("mean_score_conditional", 0.0))
        else:
            # Old correlation-style graph
            score = node.get("abs_corr", None)
            if score is None:
                score = abs(node.get("corr", 0.0))

        features.append((
            int(node["layer"]),
            int(node["feature_idx"]),
            float(score),
            node,  # full node dict for sign info
        ))

    features.sort(key=lambda x: x[2], reverse=True)
    return features[:n_features]




from contextlib import contextmanager

@contextmanager
def patch_mlp_input(
    model_hf,
    layer_idx: int,
    token_pos: int,
    new_mlp_input: torch.Tensor,
):
    """
    Transcoders are trained on MLP inputs (post_attention_layernorm output).
    This hook intercepts post_attention_layernorm and replaces its output.

    WARNING: torch.compile() may silently disable or reorder forward hooks.
    If the model was compiled with torch.compile(), hook_called will still be
    >0 (the hook fires) but the patched tensor may not propagate correctly
    through the compiled graph. Avoid torch.compile() when running interventions.
    
    Qwen3/Llama architecture:
        mlp_input = self.post_attention_layernorm(hidden_states)  ← WE HOOK HERE!
        mlp_output = self.mlp(mlp_input)  ← This receives our MODIFIED mlp_input
    
    Args:
        model_hf: HuggingFace model
        layer_idx: Layer index
        token_pos: Token position (-1 for last)
        new_mlp_input: Modified MLP input from transcoder.decode()
    """
    # Get transformer block
    try:
        block = model_hf.model.layers[layer_idx]
    except AttributeError:
        try:
            block = model_hf.transformer.h[layer_idx]
        except AttributeError:
            raise RuntimeError(f"Could not locate block {layer_idx}")
    
    # Get post_attention_layernorm (MLP input point)
    if hasattr(block, "post_attention_layernorm"):
        hook_module = block.post_attention_layernorm
    elif hasattr(block, "ln_2"):  # GPT-2 style
        hook_module = block.ln_2
    else:
        # Better diagnostics: show relevant modules
        available_modules = [name for name, _ in block.named_modules()]
        relevant = [m for m in available_modules if any(kw in m.lower() for kw in ['norm', 'mlp', 'ffn', 'feed'])]
        raise RuntimeError(
            f"Could not find post_attention_layernorm in layer {layer_idx}. "
            f"Relevant modules: {relevant[:10] if relevant else 'none found'}. "
            f"All modules: {available_modules[:10]}"
        )
    
    # Track hook execution for sanity check
    hook_called = {"count": 0}
    
    # capture the patched tensor for error diagnostic
    _patch_err_logged = {"done": False}

    def hook(module, inp, out):
        """Replace MLP input at token position."""
        hook_called["count"] += 1
        if isinstance(out, tuple):
            h = out[0].clone()
            new_tok = new_mlp_input.to(h.dtype).to(h.device)
            # Log BEFORE assignment: pre_patch_diff measures how different the
            # original activation is from the patch value.
            # >0 = patch is doing real work; ~0 = patch has no effect on this prompt.
            # Logged at INFO so it's visible at default log level.
            if not _patch_err_logged["done"]:
                pre_err = (h[:, token_pos, :] - new_tok).norm().item()
                logger.info(
                    f"[patch_mlp_input] layer={layer_idx} "
                    f"pre_patch_diff={pre_err:.4e} (>0 means patch changes something)"
                )
                _patch_err_logged["done"] = True
            h[:, token_pos, :] = new_tok
            return (h,) + out[1:]
        else:
            h = out.clone()
            new_tok = new_mlp_input.to(h.dtype).to(h.device)
            if not _patch_err_logged["done"]:
                pre_err = (h[:, token_pos, :] - new_tok).norm().item()
                logger.info(
                    f"[patch_mlp_input] layer={layer_idx} "
                    f"pre_patch_diff={pre_err:.4e} (>0 means patch changes something)"
                )
                _patch_err_logged["done"] = True
            h[:, token_pos, :] = new_tok
            return h

    handle = hook_module.register_forward_hook(hook)
    exc = None  # Track if exception occurred
    try:
        yield
    except Exception as e:
        exc = e
        raise  # Re-raise original exception
    finally:
        handle.remove()
        # CRITICAL sanity check: verify hook actually fired!
        # But ONLY if no exception occurred (avoid masking original error!)
        if exc is None:
            assert hook_called["count"] > 0, (
                f"MLP hook didn't fire for layer {layer_idx}! "
                f"Forward pass may have bypassed post_attention_layernorm. "
                f"Check model architecture or use_cache settings."
            )


def get_mlp_input_activation(
    model: ModelWrapper,
    inputs: Dict,
    layer_idx: int,
    token_pos: int = -1,
) -> torch.Tensor:
    """
    Extract MLP input activation (matches script 04 extraction point).
    
    Returns:
        MLP input tensor (1, hidden_dim) - same space as transcoder training!
    """
    # Get transformer blocks
    try:
        blocks = model.model.model.layers
    except AttributeError:
        try:
            blocks = model.model.transformer.h
        except AttributeError:
            raise RuntimeError("Could not locate transformer blocks")
    
    # Get post_attention_layernorm
    if hasattr(blocks[layer_idx], "post_attention_layernorm"):
        hook_module = blocks[layer_idx].post_attention_layernorm
    elif hasattr(blocks[layer_idx], "ln_2"):
        hook_module = blocks[layer_idx].ln_2
    else:
        raise RuntimeError(f"Could not find post_attention_layernorm in {layer_idx}")
    
    # Capture MLP input
    captured = {}
    
    def hook(module, inp, out):
        if isinstance(out, tuple):
            captured['mlp_input'] = out[0].detach()
        else:
            captured['mlp_input'] = out.detach()
    
    handle = hook_module.register_forward_hook(hook)
    try:
        with torch.no_grad():
            _ = model.model(**inputs, use_cache=False)
    finally:
        handle.remove()
    
    # CRITICAL: Verify hook fired!
    assert 'mlp_input' in captured, (
        f"Hook didn't fire for layer {layer_idx}! "
        f"post_attention_layernorm hook may not be executing. "
        f"Check model architecture."
    )
    
    mlp_input_full = captured['mlp_input']  # (batch, seq_len, hidden)
    
    # Log shape once for sanity (first call only)
    if not hasattr(get_mlp_input_activation, '_logged_shape'):
        logger.info(f"MLP input captured successfully! Shape: {mlp_input_full.shape}")
        get_mlp_input_activation._logged_shape = True
    
    return mlp_input_full[:, token_pos, :].clone()  # (batch, hidden)


class TranscoderInterventionExperiment:
    """
    Run intervention experiments using pre-trained transcoders.

    Supports:
    - Feature ablation (zero/inhibit)
    - Activation patching (swap features between prompts)
    - Feature steering (inject features)
    """

    def __init__(
        self,
        model: ModelWrapper,
        transcoder_set: TranscoderSet,
        device: torch.device,
        layers: List[int],
    ):
        """
        Initialize intervention experiment.

        Args:
            model: Language model wrapper
            transcoder_set: Pre-trained transcoders
            device: Computation device
            layers: Layers to run interventions on
        """
        self.model = model
        self.transcoder_set = transcoder_set
        self.device = device
        self.layers = layers

    def compute_logit_diff(
        self,
        prompt: str,
        correct_token: str,
        incorrect_token: str,
    ) -> float:
        """
        Compute log-probability difference (margin) between correct and incorrect tokens.

        Uses log_softmax for consistency with script 06 (which also uses log_probs).
        Returns log_prob[correct] - log_prob[incorrect].
        """
        inputs = self.model.tokenize([prompt])
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.model(**inputs, use_cache=False)
            logits = outputs.logits[0, -1, :]

        # Use log_softmax for consistency with script 06
        log_probs = torch.log_softmax(logits, dim=0)

        cid = ensure_single_token(self.model, correct_token)
        iid = ensure_single_token(self.model, incorrect_token)

        return (log_probs[cid] - log_probs[iid]).item()

    def run_ablation_experiment(
        self,
        prompt: str,
        prompt_idx: int,
        correct_token: str,
        incorrect_token: str,
        layer: int,
        feature_indices: List[int],
        mode: str = "zero",
        inhibition_factor: float = 1.0,
    ) -> InterventionResult:
        """
        Run feature ablation with REAL forward pass intervention using hooks.
        
        This is the CORRECT causal implementation:
        1. Get baseline margin
        2. Extract activations at decision token
        3. Ablate features in transcoder space
        4. Use hook to patch modified activation during forward pass
        5. Measure intervened margin
        
        Args:
            prompt: Input text
            prompt_idx: Index of prompt (for tracking)
            correct_token: Correct answer token
            incorrect_token: Incorrect alternative
            layer: Layer to intervene on
            feature_indices: Features to ablate
            mode: "zero" (set to 0) or "inhibit" (negate)
            inhibition_factor: Multiplier for inhibition mode

        Returns:
            InterventionResult with true causal effects
        """
        # Step 1: Baseline margin
        baseline_margin = self.compute_logit_diff(prompt, correct_token, incorrect_token)
        
        # Step 2: Get MLP INPUT activation (NOT residual stream!)
        # *** CRITICAL: Must match script 04 extraction point! ***
        inputs = self.model.tokenize([prompt])
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Extract MLP input (post_attention_layernorm output)
        mlp_input_act = get_mlp_input_activation(
            self.model,
            inputs,
            layer_idx=layer,
            token_pos=-1,  # Last token (decision token)
        )
            
        # Step 3: Ablate features in transcoder space
        transcoder = self.transcoder_set[layer]
        features = transcoder.encode(mlp_input_act.to(transcoder.dtype))
        
        # Store original for inhibit mode
        original = features[:, feature_indices].clone()
        
        # Apply ablation
        if mode == "zero":
            features[:, feature_indices] = 0.0
        elif mode == "inhibit":
            features[:, feature_indices] = -inhibition_factor * original
        elif mode == "mean":
            # Set to mean activation (if available)
            for feat_idx in feature_indices:
                mean_val = 0.0  # Could load from candidate_features if needed
                features[:, feat_idx] = mean_val
        else:
            raise ValueError(f"Unknown ablation mode : {mode}")
            
        # Decode to modified MLP input
        modified_mlp_input = transcoder.decode(features).to(mlp_input_act.dtype)
        
        # Sanity check: log norm change (should be non-zero!)
        delta_norm = (modified_mlp_input - mlp_input_act).norm().item()
        if prompt_idx % 10 == 0:  # Log occasionally
            logger.info(
                f"[ Ablation ] prompt={prompt_idx} layer={layer} "
                f"||Δmlp_input||={delta_norm:.4f} baseline_margin={baseline_margin:.4f}"
            )
        
        # Step 4: Intervened forward pass with hook on MLP INPUT
        with torch.no_grad():
            # *** FIX: Hook MLP input, not block output! ***
            with patch_mlp_input(
                self.model.model,
                layer_idx=layer,
                token_pos=-1,  # Last token (decision token)
                new_mlp_input=modified_mlp_input,
            ):
                # This forward pass now uses the MODIFIED MLP input at layer
                intervened_outputs = self.model.model(**inputs, use_cache=False)
                logits = intervened_outputs.logits[0, -1, :]
        
        # Get intervened margin (log_softmax — same scale as baseline)
        cid = ensure_single_token(self.model, correct_token)
        iid = ensure_single_token(self.model, incorrect_token)
        log_probs = torch.log_softmax(logits, dim=0)
        intervened_margin = (log_probs[cid] - log_probs[iid]).item()

        # Step 5: Compute causal effects
        margin_change = intervened_margin - baseline_margin  # SIGNED!
        eps = 1e-6
        baseline_sign = 1 if baseline_margin > eps else (-1 if baseline_margin < -eps else 0)
        intervened_sign = 1 if intervened_margin > eps else (-1 if intervened_margin < -eps else 0)
        sign_flipped = (baseline_sign != 0 and intervened_sign != 0 and baseline_sign != intervened_sign)
        
        return InterventionResult(
            experiment_type=f"ablation_{mode}",
            prompt_idx=prompt_idx,
            layer=layer,
            feature_indices=feature_indices,
            baseline_logit_diff=baseline_margin,
            intervened_logit_diff=intervened_margin,
            effect_size=margin_change,  # SIGNED: keeps direction!
            abs_effect_size=abs(margin_change),  # Magnitude
            relative_effect=abs(margin_change) / (abs(baseline_margin) + 1e-8),
            sign_flipped=sign_flipped,
            metadata={
                "prompt": prompt[:100],
                "correct_token": correct_token,
                "incorrect_token": incorrect_token,
                "mode": mode,
            },
        )

    def run_patching_experiment(
        self,
        source_prompt: str,
        target_prompt: str,
        prompt_idx: int,
        source_correct: str,
        target_correct: str,
        target_incorrect: str,
        layer: int,
        feature_indices: Optional[List[int]] = None,
        candidate_features: Optional[List[int]] = None,
        top_k: int = 10,
    ) -> InterventionResult:
        """
        Run activation patching with REAL forward pass intervention.

        Patches features from source prompt into target prompt using hooks.
        This measures the causal effect of transferring specific features.
        
        NOTE: This is "feature-diff patching" (causal helpfulness via transfer),
        not "high margin patching" strictly. We select features by |source - target| diff.

        Args:
            source_prompt: Prompt to get features from
            target_prompt: Prompt to patch features into
            prompt_idx: Index for tracking
            source_correct: Correct answer for source
            target_correct: Correct answer for target
            target_incorrect: Incorrect answer for target
            layer: Layer to patch
            feature_indices: Exact features to patch (overrides candidate_features selection)
            candidate_features: Pool of candidate features; top_k selected by |src-tgt| diff
            top_k: How many features to select from candidate_features

        Returns:
            InterventionResult with TRUE patching effects
        """
        # Step 1: Baseline margin for target
        baseline_margin = self.compute_logit_diff(target_prompt, target_correct, target_incorrect)

        # Step 2: Get MLP INPUT activations for both prompts
        # *** CRITICAL: Extract from same point as script 04! ***
        source_inputs = self.model.tokenize([source_prompt])
        target_inputs = self.model.tokenize([target_prompt])
        source_inputs = {k: v.to(self.device) for k, v in source_inputs.items()}
        target_inputs = {k: v.to(self.device) for k, v in target_inputs.items()}

        # Extract MLP inputs (post_attention_layernorm outputs)
        source_mlp_input = get_mlp_input_activation(
            self.model, source_inputs, layer_idx=layer, token_pos=-1
        )
        target_mlp_input = get_mlp_input_activation(
            self.model, target_inputs, layer_idx=layer, token_pos=-1
        )

        # Step 3: Patch features in transcoder space
        transcoder = self.transcoder_set[layer]

        source_features = transcoder.encode(source_mlp_input.to(transcoder.dtype))
        target_features = transcoder.encode(target_mlp_input.to(transcoder.dtype))

        # Resolve which features to patch:
        # - feature_indices (explicit) takes priority
        # - else: pick top_k from candidate_features by |source - target| diff
        # - else: raise (patching all is too broad)
        if feature_indices is not None:
            selected = feature_indices
            _selected_by_diff = False
        elif candidate_features is not None:
            d = transcoder.d_transcoder
            cands = [fi for fi in candidate_features if 0 <= fi < d]
            if not cands:
                raise ValueError(f"No valid candidate features for layer {layer}")
            cand_t = torch.tensor(cands, device=source_features.device, dtype=torch.long)
            diff = (source_features - target_features).abs()[0]  # (d_tc,)
            k = min(top_k, len(cands))
            topk_idx = torch.topk(diff[cand_t], k=k).indices
            selected = cand_t[topk_idx].tolist()
            _selected_by_diff = True
        else:
            raise ValueError(
                f"Refusing to patch ALL {transcoder.d_transcoder} features! "
                f"Pass feature_indices or candidate_features."
            )
            _selected_by_diff = False  # unreachable but keeps linter happy
        
        # Patch only specified features
        patched_features = target_features.clone()
        patched_features[:, selected] = source_features[:, selected]
        features_patched = selected

        # Decode to patched MLP input
        patched_mlp_input = transcoder.decode(patched_features).to(target_mlp_input.dtype)

        # Step 4: REAL intervened forward pass with hook on target MLP INPUT
        with torch.no_grad():
            with patch_mlp_input(
                self.model.model,
                layer_idx=layer,
                token_pos=-1,
                new_mlp_input=patched_mlp_input,
            ):
                # Target forward pass with patched MLP input
                intervened_outputs = self.model.model(**target_inputs, use_cache=False)
                logits = intervened_outputs.logits[0, -1, :]
        
        # Step 5: Get intervened margin and compute causal effects
        cid = ensure_single_token(self.model, target_correct)
        iid = ensure_single_token(self.model, target_incorrect)
        log_probs = torch.log_softmax(logits, dim=0)
        intervened_margin = (log_probs[cid] - log_probs[iid]).item()

        margin_change = intervened_margin - baseline_margin  # SIGNED!
        eps = 1e-6
        baseline_sign = 1 if baseline_margin > eps else (-1 if baseline_margin < -eps else 0)
        intervened_sign = 1 if intervened_margin > eps else (-1 if intervened_margin < -eps else 0)
        sign_flipped = (baseline_sign != 0 and intervened_sign != 0 and baseline_sign != intervened_sign)
        
        return InterventionResult(
            experiment_type="patching",
            prompt_idx=prompt_idx,
            layer=layer,
            feature_indices=features_patched,  # Always set (None raises above)
            baseline_logit_diff=baseline_margin,
            intervened_logit_diff=intervened_margin,
            effect_size=margin_change,  # SIGNED: keeps direction!
            abs_effect_size=abs(margin_change),  # Magnitude
            relative_effect=abs(margin_change) / (abs(baseline_margin) + 1e-8),
            sign_flipped=sign_flipped,
            metadata={
                "source_prompt": source_prompt[:100],
                "target_prompt": target_prompt[:100],
                "source_correct": source_correct,
                "target_correct": target_correct,
                "target_incorrect": target_incorrect,
                "selected_by_diff": _selected_by_diff,
                "candidate_pool_size": len(candidate_features) if candidate_features is not None else 0,
            },
        )
    def run_steering_experiment(
        self,
        prompts: List[Dict],
        layers: List[int],
        top_features_by_layer: Dict[int, List[int]],
        behaviour: str,
        coefficient: float = 10.0,
        top_k: int = 20,
        signed_features: Optional[Dict[int, Dict[int, float]]] = None,
    ) -> List[InterventionResult]:
        """
        Steering: for each prompt, add `coefficient` to selected feature(s) at the decision token
        in transcoder space, then patch the modified MLP input (post_attention_layernorm output)
        using `patch_mlp_input`.
        """
        logger.info(f"Running steering (coeff={coefficient}) on {len(prompts)} prompts...")
        results: List[InterventionResult] = []

        # Caller (main) passes the already-sliced sample_prompts
        sample_prompts = prompts

        for layer in layers:
            cand = top_features_by_layer.get(layer, [])[:top_k]
            if not cand:
                continue

            transcoder = self.transcoder_set[layer]
            d = transcoder.d_transcoder
            cand = [fi for fi in cand if 0 <= fi < d]
            if not cand:
                continue

            for i, prompt_data in enumerate(tqdm(sample_prompts, desc=f"Steering L{layer}")):
                prompt = prompt_data["prompt"]
                try:
                    correct, incorrect = get_answer_tokens(prompt_data)
                except KeyError as e:
                    logger.warning(f"[Steering] Skip prompt {i}: {e}")
                    continue

                # Baseline margin
                try:
                    baseline_margin = self.compute_logit_diff(prompt, correct, incorrect)
                except ValueError as e:
                    logger.warning(f"[Steering] Skip prompt {i}: {e}")
                    continue

                # Get MLP input activation at decision token
                inputs = self.model.tokenize([prompt])
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Get clean activation
                mlp_input_act = get_mlp_input_activation(self.model, inputs, layer_idx=layer, token_pos=-1)

                # Encode -> steer -> decode
                with torch.no_grad():
                    feats = transcoder.encode(mlp_input_act.to(transcoder.dtype))
                    feats_mod = feats.clone()
                    
                    # D2: Steer in direction of effect (sign-aware)
                    # Use signed_features info if available to determine direction
                    layer_signs = signed_features.get(layer, {}) if signed_features else {}
                    
                    for fi in cand:
                         # Default to +1 if no sign info (or if sign is 0)
                         s = np.sign(layer_signs.get(fi, 1.0))
                         if s == 0: s = 1.0
                         
                         # Add coefficient * sign
                         feats_mod[:, fi] += float(coefficient * s)
                    
                    steered_mlp_input = transcoder.decode(feats_mod).to(mlp_input_act.dtype)

                # Intervened margin via patching the MLP input
                # Note: patch_mlp_input is a global context manager in this file
                with torch.no_grad():
                     with patch_mlp_input(
                        self.model.model,
                        layer_idx=layer,
                        token_pos=-1,
                        new_mlp_input=steered_mlp_input,
                    ):
                        out = self.model.model(**inputs, use_cache=False)
                        logits = out.logits[0, -1, :]

                cid = ensure_single_token(self.model, correct)
                iid = ensure_single_token(self.model, incorrect)
                log_probs = torch.log_softmax(logits, dim=0)
                intervened_margin = (log_probs[cid] - log_probs[iid]).item()

                change = intervened_margin - baseline_margin
                eps = 1e-6
                b_sign = 1 if baseline_margin > eps else (-1 if baseline_margin < -eps else 0)
                i_sign = 1 if intervened_margin > eps else (-1 if intervened_margin < -eps else 0)
                sign_flipped = (b_sign != 0 and i_sign != 0 and b_sign != i_sign)

                results.append(
                    InterventionResult(
                        experiment_type="steering",
                        prompt_idx=i,
                        layer=layer,
                        feature_indices=cand,  # we steered the SET (top_k for that layer)
                        baseline_logit_diff=float(baseline_margin),
                        intervened_logit_diff=float(intervened_margin),
                        effect_size=float(change),
                        abs_effect_size=float(abs(change)),
                        relative_effect=float(abs(change) / (abs(baseline_margin) + 1e-8)),
                        sign_flipped=bool(sign_flipped),
                        metadata={
                            "behaviour": behaviour,
                            "coefficient": float(coefficient),
                            "n_features": len(cand),
                        },
                    )
                )

        return results

    def run_feature_importance_sweep(
        self,
        prompts: List[Dict],
        layer: int,
        n_prompts: int = 20,
        candidate_feature_indices: Optional[List[int]] = None,  # FIX: Accept candidates
        top_k_features: int = 50,  # Fallback if candidates not provided
    ) -> pd.DataFrame:
        """
        Sweep through top features and measure importance via ablation.

        For each feature, ablate it across multiple prompts and measure
        average effect on logit difference.

        Returns:
            DataFrame with feature importance scores
        """
        # If prompts are fewer than requested, use all (don't crash if subset < n_prompts)
        sample_prompts = prompts[:n_prompts] if len(prompts) > n_prompts else prompts
        transcoder = self.transcoder_set[layer]

        # Collect feature activations across prompts
        feature_activations = []
        logit_diffs = []

        logger.info(f"Collecting feature activations for {len(sample_prompts)} prompts...")

        for prompt_data in tqdm(sample_prompts, desc="Collecting features"):
            prompt = prompt_data["prompt"]
            
            # FIX: Use helper for answer tokens
            try:
                correct, incorrect = get_answer_tokens(prompt_data)
            except KeyError as e:
                logger.warning(f"Skipping prompt due to missing fields: {e}")
                continue

            # Get MLP input activation (NOT residual stream!)
            inputs = self.model.tokenize([prompt])
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            mlp_input = get_mlp_input_activation(
                self.model, inputs, layer_idx=layer, token_pos=-1
            )
            
            # Encode features (with no_grad for safety and speed)
            with torch.no_grad():
                features = transcoder.encode(mlp_input.to(transcoder.dtype))

            # Get logit diff FIRST (may raise ValueError for multi-token)
            try:
                logit_diff = self.compute_logit_diff(prompt, correct, incorrect)
            except ValueError as e:
                logger.warning(f"Skipping prompt due to logit diff error: {e}")
                continue
            
            # CRITICAL: Only append if logit_diff succeeded (keeps arrays synchronized!)
            feature_activations.append(to_numpy(features))
            logit_diffs.append(logit_diff)

        # SAFETY: Check if we have enough data
        if len(feature_activations) < 3:
            msg = f"Too few valid prompts for importance sweep (got {len(feature_activations)}, need >=3)."
            logger.error(msg)
            # Return empty DataFrame with correct columns to avoid downstream crashes
            return pd.DataFrame(columns=[
                "layer", "feature_idx", "mean_activation", 
                "std_activation", "activation_frequency", 
                "correlation_with_logit_diff", "abs_correlation"
            ])

        # Stack features
        feature_matrix = np.vstack(feature_activations)  # (n_prompts, d_transcoder)
        logit_diffs = np.array(logit_diffs)

        # Determine features to analyze
        if candidate_feature_indices is not None:
            features_to_analyze = candidate_feature_indices
        else:
            # Fallback: analyze top_k features (0..k)
            features_to_analyze = list(range(min(top_k_features, feature_matrix.shape[1])))

        # Compute correlation between each feature and logit diff
        results = []
        for feat_idx in features_to_analyze:
            # Bounds check
            if feat_idx >= feature_matrix.shape[1]:
                continue
                
            feat_acts = feature_matrix[:, feat_idx]

            # Only compute for features that have variance
            if np.std(feat_acts) > 1e-8:
                correlation = np.corrcoef(feat_acts, logit_diffs)[0, 1]
            else:
                correlation = 0.0

            results.append({
                "layer": layer,
                "feature_idx": feat_idx,
                "mean_activation": np.mean(feat_acts),
                "std_activation": np.std(feat_acts),
                # Fix D1: use small threshold to avoid counting near-zero activations as active
                "activation_frequency": np.mean(feat_acts > 1e-6),
                "correlation_with_logit_diff": correlation,
                "abs_correlation": abs(correlation) if not np.isnan(correlation) else 0,
            })

        df = pd.DataFrame(results)
        df = df.sort_values("abs_correlation", ascending=False)

        return df


def create_prompt_pairs(
    prompts: List[Dict],
    behaviour: str,
    source_prompts: Optional[List[Dict]] = None,
) -> List[Tuple[Dict, Dict]]:
    """
    Create (source, target) pairs for patching experiments.

    EVERY target prompt gets exactly one pair.  When class sizes are
    unbalanced, sources from the opposite class are cycled (mod-index)
    so no targets are silently dropped.

    If source_prompts is provided (Mode 1):
      Pair Target (from prompts) with Source (from source_prompts) of
      the opposite grammatical number.

    If only prompts is provided (Mode 2 / single-list):
      Self-pair: each singular target ← a plural source from the same
      list, and vice-versa, cycling when one class is shorter.

    Returns:
        List of (source_dict, target_dict) tuples — one per target.
    """
    pairs: List[Tuple[Dict, Dict]] = []

    # Helper to sort by margin (ascending: low to high)
    def sort_by_margin(p_list):
        if p_list and any('margin' in p for p in p_list):
            return sorted(p_list, key=lambda x: float(x.get('margin', 0.0)))
        return p_list

    def _pair_targets_with_sources(
        targets: List[Dict],
        sources: List[Dict],
        label: str,
    ) -> List[Tuple[Dict, Dict]]:
        """Pair every target with a source, cycling sources if fewer."""
        if not targets:
            return []
        if not sources:
            logger.warning(f"  {label}: 0 sources — skipping {len(targets)} targets")
            return []

        # Sort sources by margin descending (best first) so the first
        # targets get the strongest sources.
        sources = list(reversed(sort_by_margin(sources)))  # high → low

        out: List[Tuple[Dict, Dict]] = []
        for i, t in enumerate(targets):
            s = sources[i % len(sources)]
            out.append((s, t))
        if len(targets) > len(sources):
            logger.info(
                f"  {label}: {len(targets)} targets > {len(sources)} sources "
                f"— cycled sources (mod-index)"
            )
        return out

    if behaviour == "grammar_agreement":
        # Split targets by grammatical number
        t_sing = [p for p in prompts if p.get("number") == "singular"]
        t_plur = [p for p in prompts if p.get("number") == "plural"]

        logger.info(
            f"Pairing input: {len(prompts)} targets "
            f"(sing={len(t_sing)}, plur={len(t_plur)})"
        )

        # Mode 1: Separate Source/Target lists
        if source_prompts is not None:
            s_sing = [p for p in source_prompts if p.get("number") == "singular"]
            s_plur = [p for p in source_prompts if p.get("number") == "plural"]

            logger.info(
                f"  Sources: {len(source_prompts)} total "
                f"(sing={len(s_sing)}, plur={len(s_plur)})"
            )

            if not (t_sing or t_plur):
                raise ValueError(
                    f"No targets with 'number' field. Keys: "
                    f"{list(prompts[0].keys()) if prompts else '(empty)'}"
                )
            if not (s_sing or s_plur):
                raise ValueError(
                    f"No sources with 'number' field. Keys: "
                    f"{list(source_prompts[0].keys()) if source_prompts else '(empty)'}"
                )

            # Singular targets ← Plural sources
            t_sing = sort_by_margin(t_sing)
            pairs.extend(_pair_targets_with_sources(t_sing, s_plur, "SingTarget<-PlurSource"))

            # Plural targets ← Singular sources
            t_plur = sort_by_margin(t_plur)
            pairs.extend(_pair_targets_with_sources(t_plur, s_sing, "PlurTarget<-SingSource"))

            logger.info(f"Pairing result: {len(pairs)} pairs (Mode 1: separate source/target)")
            return pairs

        # Mode 2: Single List (self-pair across classes)
        if len(t_sing) == 0 or len(t_plur) == 0:
            logger.warning(
                f"Only one grammatical class present: "
                f"singular={len(t_sing)}, plural={len(t_plur)}. "
                f"Falling back to consecutive pairing."
            )
            for i in range(0, len(prompts) - 1, 2):
                pairs.append((prompts[i], prompts[i + 1]))
            return pairs

        t_sing = sort_by_margin(t_sing)
        t_plur = sort_by_margin(t_plur)

        # Each singular target ← a plural source (cycling if fewer plurals)
        pairs.extend(_pair_targets_with_sources(t_sing, t_plur, "SingTarget<-PlurSource"))

        # Each plural target ← a singular source (cycling if fewer singulars)
        pairs.extend(_pair_targets_with_sources(t_plur, t_sing, "PlurTarget<-SingSource"))

        logger.info(f"Pairing result: {len(pairs)} pairs (Mode 2: self-pair)")
        return pairs

    elif behaviour == "sentiment_continuation":
        positive = [p for p in prompts if p.get("sentiment") == "positive"]
        negative = [p for p in prompts if p.get("sentiment") == "negative"]
        # Every positive target ← negative source, and vice-versa
        pairs.extend(_pair_targets_with_sources(positive, negative, "Pos<-Neg"))
        pairs.extend(_pair_targets_with_sources(negative, positive, "Neg<-Pos"))

    else:
        # Generic pairing: consecutive prompts
        for i in range(0, len(prompts) - 1, 2):
            pairs.append((prompts[i], prompts[i + 1]))

    return pairs


def get_mlp_input_activations_full(
    model: ModelWrapper,
    inputs: Dict,
    layer_idx: int,
) -> torch.Tensor:
    """
    Capture full-sequence MLP inputs (post_attention_layernorm outputs) for a given layer.
    Returns: (batch, seq, hidden)
    """
    try:
        blocks = model.model.model.layers
    except AttributeError:
        blocks = model.model.transformer.h

    block = blocks[layer_idx]
    if hasattr(block, "post_attention_layernorm"):
        hook_module = block.post_attention_layernorm
    elif hasattr(block, "ln_2"):
        hook_module = block.ln_2
    else:
        raise RuntimeError(f"Could not find post_attention_layernorm in layer {layer_idx}")

    captured = {}

    def hook(module, inp, out):
        captured["mlp_input_full"] = out[0].detach() if isinstance(out, tuple) else out.detach()

    handle = hook_module.register_forward_hook(hook)
    try:
        with torch.no_grad():
            _ = model.model(**inputs, use_cache=False)
    finally:
        handle.remove()

    assert "mlp_input_full" in captured, f"Hook didn't fire for layer {layer_idx}"
    return captured["mlp_input_full"].clone()


def export_token_feature_examples(
    model: ModelWrapper,
    experiment: "TranscoderInterventionExperiment",
    prompts: List[Dict],
    behaviour: str,
    out_dir: Path,
    layers: List[int],
    top_features_by_layer: Dict[int, List[int]],
    n_prompts: int = 50,
    last_n_tokens: int = 12,
    top_k_per_layer: int = 5,
):
    """
    For each prompt, save last-N tokens and feature activations (transcoder space) for top features.
    Output: JSONL records.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"token_feature_examples_{behaviour}_n{n_prompts}_last{last_n_tokens}.jsonl"

    sample = prompts[:n_prompts] if len(prompts) > n_prompts else prompts

    with open(out_file, "w") as f:
        for i, prompt_data in enumerate(tqdm(sample, desc="Export token-feature examples")):
            prompt = prompt_data["prompt"]
            inputs = model.tokenize([prompt])
            inputs = {k: v.to(experiment.device) for k, v in inputs.items()}

            # tokens
            input_ids = inputs["input_ids"][0].detach().cpu().tolist()
            toks = model.tokenizer.convert_ids_to_tokens(input_ids)

            # slice last N
            N = min(last_n_tokens, len(toks))
            toks_last = toks[-N:]
            ids_last = input_ids[-N:]
            pos_last = list(range(len(toks) - N, len(toks)))  # absolute positions in seq

            rec = {
                "behaviour": behaviour,
                "prompt_idx": i,
                "prompt": prompt,
                "seq_len": len(toks),
                "last_n": N,
                "token_positions": pos_last,
                "tokens": toks_last,
                "token_ids": ids_last,
                "layers": {},
                "meta": {k: v for k, v in prompt_data.items() if k != "prompt"},
            }

            # For each layer: get full MLP inputs, take last N positions, encode, keep top features
            for layer in layers:
                cand = top_features_by_layer.get(layer, [])
                if not cand:
                    continue

                d = experiment.transcoder_set[layer].d_transcoder
                cand = [fi for fi in cand[:top_k_per_layer] if 0 <= fi < d]
                if not cand:
                    continue

                mlp_full = get_mlp_input_activations_full(model, inputs, layer_idx=layer)  # (1, seq, hidden)
                mlp_last = mlp_full[:, -N:, :]  # (1, N, hidden)

                transcoder = experiment.transcoder_set[layer]
                with torch.no_grad():
                    # Fix C: reshape to 2D before encode — many SAE/Transcoder impls
                    # expect (batch, hidden), not (batch, seq, hidden).
                    B, Nseq, H = mlp_last.shape
                    x2 = mlp_last.reshape(B * Nseq, H).to(transcoder.dtype)
                    z2 = transcoder.encode(x2)          # (B*N, d_transcoder)
                    feats = z2.reshape(B, Nseq, -1)     # (1, N, d_transcoder)

                feats_np = to_numpy(feats[0, :, cand])  # (N, K)

                rec["layers"][str(layer)] = {
                    "feature_indices": cand,
                    "feature_acts": feats_np.tolist(),  # shape (N, K)
                }

            f.write(json.dumps(rec) + "\n")

    logger.info(f"Saved token-feature examples to {out_file}")
    return out_file


def save_results(
    results: List[InterventionResult],
    output_path: Path,
    behaviour: str,
    experiment_type: str,
    metadata: Dict,
):
    """Save intervention results."""
    output_path.mkdir(parents=True, exist_ok=True)

    if len(results) == 0:
        logger.warning(f"No results to save for {experiment_type} / {behaviour}. Writing empty files.")
        # save empty CSV with correct columns
        empty_df = pd.DataFrame(columns=[f.name for f in dataclasses.fields(InterventionResult)])
        empty_df.to_csv(output_path / f"intervention_{experiment_type}_{behaviour}.csv", index=False)

        summary = {
            "behaviour": behaviour,
            "experiment_type": experiment_type,
            "n_experiments": 0,
            "timestamp": datetime.now().isoformat(),
            **metadata,
        }
        with open(output_path / f"intervention_{experiment_type}_{behaviour}_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        return

    # Convert to DataFrame
    df = pd.DataFrame([dataclasses.asdict(r) for r in results])
    df.to_csv(
        output_path / f"intervention_{experiment_type}_{behaviour}.csv",
        index=False,
    )

    # Save summary statistics with presentation metrics
    summary = {
        "behaviour": behaviour,
        "experiment_type": experiment_type,
        "n_experiments": len(results),
        
        # Effect size metrics (SIGNED - directional bias)
        "mean_effect_size": float(df["effect_size"].mean()),  # Can be ~0 if +/- cancel!
        "median_effect_size": float(df["effect_size"].median()),
        "std_effect_size": float(df["effect_size"].std()),
        
        # Absolute effect size (MAGNITUDE - doesn't cancel out)
        "mean_abs_effect_size": float(df["abs_effect_size"].mean()),
        "median_abs_effect_size": float(df["abs_effect_size"].median()),
        "std_abs_effect_size": float(df["abs_effect_size"].std()),
        
        # Relative effect (MAGNITUDE-ONLY: abs / baseline)
        "mean_relative_effect": float(df["relative_effect"].mean()),
        
        # Margin metrics
        "mean_baseline_logit_diff": float(df["baseline_logit_diff"].mean()),
        "median_baseline_logit_diff": float(df["baseline_logit_diff"].median()),
        "mean_intervened_logit_diff": float(df["intervened_logit_diff"].mean()),
        "median_intervened_logit_diff": float(df["intervened_logit_diff"].median()),
        
        # Key presentation metric: sign flip rate
        "sign_flip_rate": float(df["sign_flipped"].mean()) if "sign_flipped" in df.columns else 0.0,
        "n_sign_flips": int(df["sign_flipped"].sum()) if "sign_flipped" in df.columns else 0,
        
        "timestamp": datetime.now().isoformat(),
        **metadata,
    }

    with open(output_path / f"intervention_{experiment_type}_{behaviour}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Saved {len(results)} results to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run intervention experiments")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiment_config.yaml",
        help="Path to experiment config",
    )
    parser.add_argument(
        "--transcoder_config",
        type=str,
        default="configs/transcoder_config.yaml",
        help="Path to transcoder config",
    )
    parser.add_argument(
        "--behaviour",
        type=str,
        choices=["grammar_agreement"],
        default="grammar_agreement",
        help="Which behaviour to analyze (currently only grammar_agreement)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Which split to use",
    )
    parser.add_argument(
        "--experiment", 
        type=str, 
        default="all", 
        choices=["ablation", "patching", "feature_importance", "steering", "all"],
        help="Type of intervention to run"
    )
    parser.add_argument("--steering_coeff", type=float, default=10.0,
                        help="Coefficient for steering intervention (default: 10.0)")
                        
    # Token export args
    parser.add_argument("--export_token_examples", action="store_true",
                        help="Export token-feature examples JSONL for thesis visuals")
    parser.add_argument("--last_n_tokens", type=int, default=12,
                        help="How many last tokens to store per prompt")
    parser.add_argument("--token_examples_n_prompts", type=int, default=50,
                        help="How many prompts to export for token examples")
    parser.add_argument("--token_examples_topk", type=int, default=5,
                        help="Top-K features per layer to export")
    parser.add_argument(
        "--model_size",
        type=str,
        default=None,
        help="Model size (0.6b, 1.7b, 4b, 8b, 14b)",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="Specific layers to intervene on",
    )
    parser.add_argument(
        "--n_prompts",
        type=int,
        default=20,
        help="Number of prompts to use",
    )
    parser.add_argument(
        "--prompts_file",
        type=str,
        default=None,
        help="Path to specific JSONL file with prompts (Targets for patching)",
    )
    parser.add_argument(
        "--source_prompts_file",
        type=str,
        default=None,
        help="Path to specific JSONL file with Source prompts (for patching)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of top features to ablate (default: 5). Increase for ensemble ablation.",
    )
    parser.add_argument(
        "--ablation_sign",
        type=str,
        default="pos",
        choices=["pos", "neg", "all"],
        help="Which features to ablate by sign of mean_score_conditional: "
             "pos=positive drivers (expect margin↓), neg=negative drivers (expect margin↑), all=both",
    )
    parser.add_argument(
        "--graph_n_prompts",
        type=int,
        default=None,
        help="Explicit n_prompts suffix for loading attribution graph (e.g. 100 loads _n100.json). "
             "Defaults to --n_prompts if not set.",
    )
    parser.add_argument(
        "--allow_sharded",
        action="store_true",
        default=False,
        help="Allow running interventions on a model sharded across multiple devices (device_map). "
             "By default this raises an error because hooks may fire on the wrong device. "
             "Use only if you know what you are doing.",
    )
    args = parser.parse_args()

    # Load configs
    config = load_config(args.config)
    results_path = Path(config["paths"]["results"])
    output_path = results_path / "interventions"
    output_path.mkdir(parents=True, exist_ok=True)

    # Config
    tc_config = load_transcoder_config(args.transcoder_config)
    torch.manual_seed(config["seeds"]["intervention_sampling"])

    # Model size
    model_size = args.model_size or tc_config.get("model_size", "4b")

    # Layers
    if args.layers:
        layers = args.layers
    else:
        # Use middle layers for interventions
        layers = tc_config.get("analysis_layers", {}).get("middle", [15, 16, 17, 18, 19, 20])

    # Behaviours (single behaviour for pipeline testing)
    behaviours = [args.behaviour]

    print("=" * 70)
    print("INTERVENTION EXPERIMENTS (TRANSCODER-BASED)")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Model size: {model_size}")
    print(f"  Layers: {layers}")
    print(f"  Behaviours: {behaviours}")
    print(f"  N prompts: {args.n_prompts}")
    print(f"  Experiments: {args.experiment}")

    # ========== CRITICAL FIX: Load MODEL first, THEN transcoders! ==========
    # This ensures transcoders use the ACTUAL device from loaded model
    
    # Load language model FIRST
    print(f"\nLoading language model...")
    model_name = tc_config["transcoders"][model_size]["model_name"]
    print(f"Model: {model_name}")
    
    # CRITICAL: Use BASE model to match transcoders and features
    model = ModelWrapper(
        model_name=model_name,
        dtype="bfloat16",
        device="auto",
        trust_remote_code=True,
    )
    
    # FIX: Get ACTUAL device from loaded model (with safety!)
    try:
        model_device = next(model.model.parameters()).device
    except StopIteration:
        # Fallback if parameters() is empty (rare: offload/quantization)
        model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.warning(f"Could not infer device from model parameters; falling back to {model_device}")
    
    device = model_device
    logger.info(f"Model loaded on device: {device}")
    
    # Fix B: Raise on sharded model unless --allow_sharded is explicitly set.
    # With accelerate device_map, hooks may fire on wrong device/time.
    if hasattr(model.model, "hf_device_map"):
        unique_devices = set(model.model.hf_device_map.values())
        if len(unique_devices) > 1:
            if not getattr(args, "allow_sharded", False):
                raise RuntimeError(
                    f"Model is sharded across {len(unique_devices)} devices: {unique_devices}. "
                    f"Forward hooks may fire on the wrong device, making interventions unreliable. "
                    f"Either load on a single GPU (remove device_map) or pass --allow_sharded to override."
                )
            else:
                logger.warning(
                    f"--allow_sharded set: proceeding with sharded model ({unique_devices}). "
                    f"Intervention results may be unreliable."
                )
        else:
            logger.info(f"Model device_map uses single device {unique_devices} — OK.")
    
    # NOW load transcoders on the SAME device as model
    print("Loading transcoders...")
    transcoder_set = load_transcoder_set(
        model_size=model_size,
        device=device,  # Use actual model device!
        dtype=torch.bfloat16,
        lazy_load=True,
        layers=layers,
    )

    # Create experiment runner
    experiment = TranscoderInterventionExperiment(
        model=model,
        transcoder_set=transcoder_set,
        device=device,
        layers=layers,
    )

    # Process behaviours
    for behaviour in behaviours:
        print("\n" + "=" * 70)
        print(f"BEHAVIOUR: {behaviour}")
        print("=" * 70)

        # Load prompts
        prompt_path = Path(config["paths"]["prompts"])
        try:
            prompts = load_prompts(prompt_path, behaviour, args.split, prompts_file=args.prompts_file)
        except FileNotFoundError:
            print(f"Prompt file not found. Skipping {behaviour}.")
            continue

        if args.source_prompts_file:
            # Custom loader for specific file
            source_prompts_path = Path(args.source_prompts_file)
            print(f"Loading SOURCE prompts from {source_prompts_path}")
            # We can reuse load_prompts with prompts_file arg, ignoring behaviour path logic
            try:
                source_prompts = load_prompts(prompt_path, behaviour, args.split, prompts_file=args.source_prompts_file)
            except FileNotFoundError:
                logger.error(f"Source prompts file not found: {args.source_prompts_file}")
                raise
        else:
            source_prompts = None

        print(f"Loaded {len(prompts)} target prompts")
        
        def _count_numbers(ps):
            return {
                "singular": sum(1 for p in ps if p.get("number") == "singular"),
                "plural": sum(1 for p in ps if p.get("number") == "plural"),
                "unknown": sum(1 for p in ps if p.get("number") not in ("singular","plural")),
            }
            
        logger.info(f"Targets counts: {_count_numbers(prompts)}")

        if source_prompts:
            print(f"Loaded {len(source_prompts)} source prompts")
            logger.info(f"Sources counts: {_count_numbers(source_prompts)}")
            
        # Load attribution graph — use --graph_n_prompts if set, else --n_prompts
        graph_n = args.graph_n_prompts if args.graph_n_prompts is not None else args.n_prompts
        graph_data = load_attribution_graph(
            results_path, behaviour, args.split, n_prompts=graph_n
        )
        
        # Build per-layer feature index lists (unsigned and signed)
        top_features_by_layer: Dict[int, List[int]] = {}
        # (feature_idx, signed_mean_score_conditional) — for sign-aware ablation
        top_features_by_layer_signed: Dict[int, List[Tuple[int, float]]] = {}
        top_nodes = []

        if graph_data:
            meta = graph_data.get("metadata", {}) or {}
            gattrs = graph_data.get("graph_attrs", {}) or {}
            graph_type = meta.get("graph_type", None)
            if graph_type is None and "union_params" in gattrs:
                graph_type = "per_prompt_union"

            # Load enough features to cover all layers/signs after filtering.
            # Scale with top_k (we filter by sign and activation later), cap at 5000.
            n_load = max(200, args.top_k * 20)
            n_load = min(n_load, 5000)
            top_nodes = get_top_attributed_features(graph_data, n_features=n_load)
            print(f"Loaded {len(top_nodes)} top attributed features from graph (requested top {n_load})")

            for L, fidx, score, node in top_nodes:
                top_features_by_layer.setdefault(L, []).append(fidx)
                # SIGN source depends on graph type:
                # - per_prompt_union: prefer beta (stable sign), else mean_score_conditional
                # - correlation graphs: use corr
                signed = None
                if graph_type == "per_prompt_union":
                    if "beta" in node:
                        signed = node.get("beta", None)  # stable direction
                    if signed is None:
                        signed = node.get("mean_score_conditional", None)
                    if signed is None:
                        signed = node.get("mean_score_missing0", None)
                else:
                    signed = node.get("corr", None)

                if signed is None:
                    logger.warning(
                        f"Signed score missing in node L{L}_F{fidx} (graph_type={graph_type}); "
                        f"--ablation_sign pos/neg may be unreliable for this feature."
                    )
                    signed = 0.0
                top_features_by_layer_signed.setdefault(L, []).append((fidx, float(signed)))
        else:
            print("No attribution graph found, will use random/first features")

        # ====== UNIFIED PROMPT SUBSETTING ======
        # Slice ONCE, use everywhere. Prevents steering/feature_importance
        # from silently using more prompts than ablation/patching.
        sample_prompts = prompts[:args.n_prompts] if args.n_prompts else prompts
        logger.info(
            f"Prompt subsetting: {len(prompts)} total -> {len(sample_prompts)} selected "
            f"(--n_prompts={args.n_prompts})"
        )

        metadata = {
            "model_size": model_size,
            "transcoder_repo": tc_config["transcoders"][model_size]["repo_id"],
            "n_prompts": len(sample_prompts),  # actual count, not CLI arg
            "n_prompts_requested": args.n_prompts,
            "n_prompts_available": len(prompts),
            "top_k": args.top_k,
            "prompts_file": args.prompts_file,
            "source_prompts_file": args.source_prompts_file,
        }

        # Export token-feature examples if requested (before running experiments)
        if args.export_token_examples:
            out_dir = output_path / behaviour / "token_feature_examples"
            print(f"\nExporting token-feature examples to {out_dir}...")
            export_token_feature_examples(
                model=model,
                experiment=experiment,
                prompts=prompts,
                behaviour=behaviour,
                out_dir=out_dir,
                layers=layers,
                top_features_by_layer=top_features_by_layer,
                n_prompts=args.token_examples_n_prompts,
                last_n_tokens=args.last_n_tokens,
                top_k_per_layer=args.token_examples_topk,
            )

        # Run experiments
        # Determine experiments to run
        if args.experiment == "all":
            experiments_to_run = ["ablation", "patching", "feature_importance", "steering"]
        else:
             if args.experiment in ["importance", "feature_importance"]:
                experiments_to_run = ["feature_importance"]
             elif args.experiment == "steering":
                experiments_to_run = ["steering"]
             else:
                experiments_to_run = [args.experiment]

        logger.info(f"Experiments to run: {experiments_to_run}")

        for exp_type in experiments_to_run:
            print(f"\n--- Running {exp_type} experiments ---")
            logger.info(f"Using {len(sample_prompts)} prompts for {exp_type}")

            if exp_type == "feature_importance":
                # Feature importance sweep (per layer)
                for layer in layers:
                    print(f"\nLayer {layer} feature importance...")
                    
                    # Use graph candidates if available
                    layer_candidates = top_features_by_layer.get(layer, None)
                    
                    if layer_candidates:
                        print(f"  Using {len(layer_candidates)} candidate features from attribution graph")
                    else:
                        print("  No attribution graph; using first 50 features (baseline/control)")

                    importance_df = experiment.run_feature_importance_sweep(
                        sample_prompts,
                        layer=layer,
                        n_prompts=len(sample_prompts),
                        candidate_feature_indices=layer_candidates,
                        top_k_features=50,
                    )

                    # Save importance results (create directory first!)
                    imp_out_path = output_path / behaviour / "importance"
                    imp_out_path.mkdir(parents=True, exist_ok=True)
                    importance_df.to_csv(
                        imp_out_path / f"feature_importance_layer_{layer}.csv", 
                        index=False
                    )
                    print(f"  Saved importance results to {imp_out_path}")

            elif exp_type == "steering":
                # Steering handles its own layer/prompt loops
                steer_results = experiment.run_steering_experiment(
                    sample_prompts, layers, top_features_by_layer, behaviour,
                    coefficient=args.steering_coeff, top_k=args.top_k,
                    # Convert list of tuples to dict for lookup: layer -> {feat: sign}
                    signed_features={
                        L: {f: s for f, s in feats} 
                        for L, feats in top_features_by_layer_signed.items()
                    }
                )
                save_results(
                    steer_results, 
                    output_path / behaviour, 
                    behaviour, 
                    "steering", 
                    {**metadata, "steering_coeff": args.steering_coeff}
                )

            elif exp_type == "ablation":
                # Feature ablation
                all_results: List[InterventionResult] = []

                for i, prompt_data in enumerate(tqdm(sample_prompts, desc="Ablation")):
                    prompt = prompt_data["prompt"]
                    try:
                        correct, incorrect = get_answer_tokens(prompt_data)
                    except KeyError as e:
                        logger.warning(f"Skipping prompt {i} due to missing fields: {e}")
                        continue

                    for layer in layers:
                        # Select features by sign of mean_score_conditional
                        cands = top_features_by_layer_signed.get(layer, [])
                        if cands:
                            if args.ablation_sign == "pos":
                                # Positive drivers: ablating should decrease margin
                                layer_features = [fi for fi, s in cands if s > 0][:args.top_k]
                            elif args.ablation_sign == "neg":
                                # Negative drivers: ablating should increase margin
                                layer_features = [fi for fi, s in cands if s < 0][:args.top_k]
                            else:  # "all"
                                layer_features = [fi for fi, _ in cands][:args.top_k]

                            # If sign-filter yields nothing, fall back to top-k UNSIGNED from graph (not [0..k]).
                            if len(layer_features) == 0:
                                logger.warning(
                                    f"No features matched sign={args.ablation_sign} at layer {layer}. "
                                    f"Falling back to top-{args.top_k} unsigned features from graph."
                                )
                                layer_features = [fi for fi, _ in cands][:args.top_k]
                        else:
                            layer_features = top_features_by_layer.get(layer, [])[:args.top_k]

                        d = experiment.transcoder_set[layer].d_transcoder
                        layer_features = [fi for fi in layer_features if 0 <= fi < d]

                        if not layer_features:
                            logger.warning(
                                f"No valid features for layer {layer} after graph selection; "
                                f"using first {args.top_k} ONLY as last resort."
                            )
                            layer_features = list(range(min(args.top_k, d)))

                        # Fix 7: Activation-aware selection — prefer features active on THIS prompt
                        try:
                            abl_inputs = experiment.model.tokenize([prompt])
                            abl_inputs = {k: v.to(experiment.device) for k, v in abl_inputs.items()}
                            abl_mlp = get_mlp_input_activation(
                                experiment.model, abl_inputs, layer_idx=layer, token_pos=-1
                            )
                            tc = experiment.transcoder_set[layer]
                            abl_feats = tc.encode(abl_mlp.to(tc.dtype))[0]  # (d_tc,)
                            cand_t = torch.tensor(layer_features, device=abl_feats.device, dtype=torch.long)
                            # Use abs(): features might be signed (though rarely for JumpReLU), safeguards against negative values
                            act_scores = abl_feats[cand_t].abs()
                            k = min(args.top_k, len(layer_features))
                            topk_idx = torch.topk(act_scores, k=k).indices
                            layer_features = cand_t[topk_idx].tolist()
                        except Exception as e:
                            logger.warning(f"Activation-aware selection failed for layer {layer}: {e}. Using sign-filtered list.")
                            layer_features = layer_features[:args.top_k]

                        try:
                            result = experiment.run_ablation_experiment(
                                prompt=prompt,
                                prompt_idx=i,
                                correct_token=correct,
                                incorrect_token=incorrect,
                                layer=layer,
                                feature_indices=layer_features,
                                mode="zero",
                            )
                            all_results.append(result)
                        except ValueError as e:
                            logger.warning(f"Skipping prompt {i} layer {layer}: {e}")
                            continue

                save_results(all_results, output_path / behaviour, behaviour, "ablation", metadata)

            elif exp_type == "patching":
                # Activation patching
                results: List[InterventionResult] = []

                # Create pairs from the unified sample_prompts
                pairs = create_prompt_pairs(sample_prompts, behaviour, source_prompts=source_prompts)
                print(f"  Created {len(pairs)} pairs for patching")

                expected = len(sample_prompts)
                if len(pairs) < expected:
                    logger.warning(
                        f"Patching produced {len(pairs)} pairs but expected {expected} "
                        f"(one per target). {expected - len(pairs)} targets have no pair."
                    )

                if not pairs:
                    logger.warning("No pairs created! Check your prompts/margins.")
                    save_results(results, output_path / behaviour, behaviour, "patching", metadata)
                    continue

                for idx, (source, target) in enumerate(tqdm(pairs, desc="Patching")):
                    try:
                        source_correct, source_incorrect = get_answer_tokens(source)
                        target_correct, target_incorrect = get_answer_tokens(target)
                    except KeyError as e:
                        logger.warning(f"Skipping pair {idx} due to missing fields: {e}")
                        continue

                    # Compute source margin ONCE per pair
                    try:
                        source_margin = experiment.compute_logit_diff(
                            source["prompt"], source_correct, source_incorrect
                        )
                    except ValueError as e:
                        logger.warning(f"Skipping pair {idx} due to source margin error: {e}")
                        continue

                    for layer in layers:
                        # Candidate features from graph (pair-specific selection happens inside run_patching_experiment)
                        cands = top_features_by_layer.get(layer, [])
                        d = experiment.transcoder_set[layer].d_transcoder
                        cands = [fi for fi in cands if 0 <= fi < d]
                        if not cands:
                            logger.warning(f"No valid features for layer {layer}, using first {args.top_k}")
                            cands = list(range(min(args.top_k, d)))

                        try:
                            result = experiment.run_patching_experiment(
                                source_prompt=source["prompt"],
                                target_prompt=target["prompt"],
                                prompt_idx=idx,
                                source_correct=source_correct,
                                target_correct=target_correct,
                                target_incorrect=target_incorrect,
                                layer=layer,
                                candidate_features=cands,  # pair-specific top-k selected inside
                                top_k=args.top_k,
                            )

                            # Enrich metadata
                            result.metadata.update({
                                "source_margin": float(source_margin),
                                "target_margin": float(result.baseline_logit_diff),
                                "source_number": source.get("number", "unknown"),
                                "target_number": target.get("number", "unknown"),
                                "source_orig_idx": source.get("orig_idx", -1),
                                "target_orig_idx": target.get("orig_idx", -1),
                            })

                            results.append(result)

                        except (ValueError, KeyError) as e:
                            logger.warning(f"Skipping pair {idx} layer {layer}: {e}")
                            continue

                save_results(results, output_path / behaviour, behaviour, "patching", metadata)
                
    print("\n" + "=" * 70)
    print("INTERVENTION EXPERIMENTS COMPLETE")
    print("=" * 70)
    print(f"\nOutput: {output_path.absolute()}")
    print("\nNext step: python scripts/08_generate_figures.py")


if __name__ == "__main__":
    main()
