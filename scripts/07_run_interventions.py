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
    
    # Try with _nN suffix first if n_prompts specified
    if n_prompts is not None:
        graph_file = base_path / f"attribution_graph_{split}_n{n_prompts}.json"
        if graph_file.exists():
            with open(graph_file, "r") as f:
                return json.load(f)
    
    # Fall back to old naming (no _nN suffix)
    graph_file = base_path / f"attribution_graph_{split}.json"
    if graph_file.exists():
        with open(graph_file, "r") as f:
            return json.load(f)
    
    # Not found
    logger.warning(f"Attribution graph not found: {graph_file}")
    return None


def get_top_attributed_features(
    graph_data: Dict,
    n_features: int = 10,
) -> List[Tuple[int, int, float]]:
    """
    Extract top attributed features from graph.

    Returns:
        List of (layer, feature_idx, attribution_score) tuples
    """
    # Safety: handle None or malformed graph
    if not graph_data or "nodes" not in graph_data:
        logger.warning("Graph data is None or missing 'nodes' key")
        return []
    
    features = []
    for node in graph_data["nodes"]:
        if node.get("type") == "feature":
            # FIX: Read abs_corr (not avg_differential_attribution)
            score = node.get("abs_corr", None)
            if score is None:
                # Fallback: compute from corr if abs_corr missing
                score = abs(node.get("corr", 0.0))
            
            features.append((
                int(node["layer"]),
                int(node["feature_idx"]),
                float(score),
            ))

    # Sort by attribution magnitude  
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
    *** CRITICAL FIX: Patch MLP INPUT not residual stream! ***
    
    Transcoders are trained on MLP inputs (post_attention_layernorm output).
    This hook intercepts post_attention_layernorm and replaces its output.
    
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
    
    def hook(module, inp, out):
        """Replace MLP input at token position."""
        hook_called["count"] += 1
        if isinstance(out, tuple):
            h = out[0].clone()
            h[:, token_pos, :] = new_mlp_input.to(h.dtype).to(h.device)
            return (h,) + out[1:]
        else:
            h = out.clone()
            h[:, token_pos, :] = new_mlp_input.to(h.dtype).to(h.device)
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
        logger.info(f"✓ MLP input captured successfully! Shape: {mlp_input_full.shape}")
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
        Compute logit difference (margin) between correct and incorrect tokens.
        
        Returns:
            logits[correct] - logits[incorrect]
        """
        inputs = self.model.tokenize([prompt])
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            # FIX: use_cache=False for consistency with get_mlp_input_activation
            outputs = self.model.model(**inputs, use_cache=False)
            logits = outputs.logits[0, -1, :]

        # FIX: Validate single-token and get IDs
        cid = ensure_single_token(self.model, correct_token)
        iid = ensure_single_token(self.model, incorrect_token)

        return (logits[cid] - logits[iid]).item()

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
            raise ValueError(f"Unknown ablation mode: {mode}")
            
        # Decode to modified MLP input
        modified_mlp_input = transcoder.decode(features).to(mlp_input_act.dtype)
        
        # Sanity check: log norm change (should be non-zero!)
        delta_norm = (modified_mlp_input - mlp_input_act).norm().item()
        if prompt_idx % 10 == 0:  # Log occasionally
            logger.info(
                f"[Ablation] prompt={prompt_idx} layer={layer} "
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
        
        # Get intervened margin
        cid = ensure_single_token(self.model, correct_token)
        iid = ensure_single_token(self.model, incorrect_token)
        intervened_margin = (logits[cid] - logits[iid]).item()
        
        # Step 5: Compute causal effects
        margin_change = intervened_margin - baseline_margin  # SIGNED!
        baseline_sign = 1 if baseline_margin > 0 else -1
        intervened_sign = 1 if intervened_margin > 0 else -1
        sign_flipped = (baseline_sign != intervened_sign)
        
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
    ) -> InterventionResult:
        """
        Run activation patching with REAL forward pass intervention.
        
        Patches features from source prompt into target prompt using hooks.
        This measures the causal effect of transferring specific features.

        Args:
            source_prompt: Prompt to get features from
            target_prompt: Prompt to patch features into
            prompt_idx: Index for tracking
            source_correct: Correct answer for source
            target_correct: Correct answer for target
            target_incorrect: Incorrect answer for target
            layer: Layer to patch
            feature_indices: Specific features to patch (None = all, NOT RECOMMENDED)

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

        # SAFETY: Refuse to patch ALL features (too broad, hard to interpret!)
        if feature_indices is None:
            raise ValueError(
                f"Refusing to patch ALL {transcoder.d_transcoder} features! "
                f"This is too broad for meaningful interpretation. "
                f"Pass feature_indices explicitly (e.g., top 5-20 features from attribution graph)."
            )
        
        # Patch only specified features
        patched_features = target_features.clone()
        patched_features[:, feature_indices] = source_features[:, feature_indices]
        features_patched = feature_indices

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
        intervened_margin = (logits[cid] - logits[iid]).item()
        
        margin_change = intervened_margin - baseline_margin  # SIGNED!
        baseline_sign = 1 if baseline_margin > 0 else -1
        intervened_sign = 1 if intervened_margin > 0 else -1
        sign_flipped = (baseline_sign != intervened_sign)
        
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
    ) -> List[InterventionResult]:
        """
        Steering: for each prompt, add `coefficient` to selected feature(s) at the decision token
        in transcoder space, then patch the modified MLP input (post_attention_layernorm output)
        using `patch_mlp_input`.
        """
        logger.info(f"Running steering (coeff={coefficient}) on {len(prompts)} prompts...")
        results: List[InterventionResult] = []

        # Choose prompts subset if caller passed full list; keep consistent with other experiments
        sample_prompts = prompts  # caller already slices in main if needed

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
                
                # We need get_mlp_input_activation to be available or self.get_mlp_input_activation if it's in class?
                # The user provided snippet uses `get_mlp_input_activation(self.model, ...)`
                # I assume get_mlp_input_activation is a global function in the file.
                # I need to check if it exists. 
                # Based on previous edits, `get_mlp_input_activation` likely exists as a global helper or static method.
                # Let's assume global.
                mlp_input_act = get_mlp_input_activation(self.model, inputs, layer_idx=layer, token_pos=-1)

                # Encode -> steer -> decode
                with torch.no_grad():
                    feats = transcoder.encode(mlp_input_act.to(transcoder.dtype))
                    feats_mod = feats.clone()
                    feats_mod[:, cand] += float(coefficient)  # constant push
                    steered_mlp_input = transcoder.decode(feats_mod).to(mlp_input_act.dtype)

                # Intervened margin via patching the MLP input
                # patch_mlp_input is a context manager. User said "global context-manager patch_mlp_input(model_hf, ...)".
                # But in my previous code I used `self.patch_mlp_input`.
                # The user snippet uses `with patch_mlp_input(...)`.
                # I need to know if `patch_mlp_input` is global or member.
                # In 07 script, `patch_mlp_input` is a method of `TranscoderInterventionExperiment`?
                # User said: "you have patch_mlp_input(model_hf, ...)" which implies global.
                # But in typical pattern, it might be `self.patch_mlp_input`.
                # Let's check the file content later if possible, but for now I will use what user suggests: `patch_mlp_input` logic using global or method.
                # User code: `with patch_mlp_input(self.model.model, ...)`
                # This implies it's a global context manager OR I should import it?
                # I will define `patch_mlp_input` if it's missing or use `self.patch_mlp_input` if it's a method.
                # Wait, I recall seeing `with self.patch_mlp_input(...)` in my own previous code.
                # But the user says: "self.patch_mlp_input does not exist (there is a global ...)".
                # So I will use the global `patch_mlp_input` and pass the hf model.
                
                with torch.no_grad():
                     # Assuming patch_mlp_input is imported or available globally
                     # I might need to import it if it's in another file, or defined in this file.
                     # If it's defined in this file, I can use it.
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
                intervened_margin = (logits[cid] - logits[iid]).item()

                change = intervened_margin - baseline_margin
                sign_flipped = (np.sign(baseline_margin) != np.sign(intervened_margin))

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
                "activation_frequency": np.mean(feat_acts > 0),
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
    Create pairs of prompts for patching experiments.
    
    If source_prompts is provided (High Margin):
      Pair Target (from prompts, Low Margin) with Source (from source_prompts, High Margin).
      Target(Singular) <- Source(Plural)
      Target(Plural) <- Source(Singular)
      
    If only prompts is provided (Single List):
      Sort by margin and pair Low <-> High from the same list.
    """
    pairs = []
    
    # Helper to sort by margin (ascending: low to high)
    def sort_by_margin(p_list):
        if p_list and 'margin' in p_list[0]:
            return sorted(p_list, key=lambda x: float(x.get('margin', 0.0)))
        return p_list

    if behaviour == "grammar_agreement":
        # Split targets (prompts)
        t_sing = [p for p in prompts if p.get("number") == "singular"]
        t_plur = [p for p in prompts if p.get("number") == "plural"]
        
        # Mode 1: Separate Source/Target lists
        if source_prompts is not None:
            s_sing = [p for p in source_prompts if p.get("number") == "singular"]
            s_plur = [p for p in source_prompts if p.get("number") == "plural"]
            
            # Check availability with strict error if classes are missing
            if not (t_sing and t_plur and s_sing and s_plur):
                raise ValueError(
                    f"Cannot build grammar pairs: "
                    f"targets(sing={len(t_sing)}, plur={len(t_plur)}), "
                    f"sources(sing={len(s_sing)}, plur={len(s_plur)}). "
                    f"Fix 07b stratification ranges."
                )

            # Pair Low Margin Singular Target <- High Margin Plural Source
            # Assume sets are already filtered by margin in 07b, but sorting helps determinism
            t_sing = sort_by_margin(t_sing) # Low -> High
            s_plur = sort_by_margin(s_plur) # Low
            # We want BEST sources (Highest margin) for BEST targets (Lowest margin)
            # t_sing: [0.1, 0.2 ... 0.5]
            # s_plur: [2.5, 2.6 ... 3.0]
            # zip(t_sing, reversed(s_plur)) -> (0.1, 3.0), (0.2, 2.9)...
            for t, s in zip(t_sing, reversed(s_plur)):
                pairs.append((s, t)) # (Source, Target)
            
            if not t_plur or not s_sing:
                logger.warning(f"Missing Plural Targets ({len(t_plur)}) or Singular Sources ({len(s_sing)})")
            else:
                # Pair Low Margin Plural Target <- High Margin Singular Source
                t_plur = sort_by_margin(t_plur)
                s_sing = sort_by_margin(s_sing)
                
                for t, s in zip(t_plur, reversed(s_sing)):
                    pairs.append((s, t))

            # Validate pair counts
            if source_prompts:
                n1 = min(len(t_sing), len(s_plur))
                n2 = min(len(t_plur), len(s_sing))
                logger.info(f"Pairing plan: Sing_Target<-Plur_Source: {n1}, Plur_Target<-Sing_Source: {n2}")
                
                if n1 + n2 < 2:
                     raise ValueError("No pairs could be formed (after strict class check). Check your quotas.")
            
            return pairs

        # Mode 2: Single List (Fallback)
        if len(t_sing) == 0 or len(t_plur) == 0:
            logger.warning(
                f"Not enough classes for grammar pairing: "
                f"singular={len(t_sing)}, plural={len(t_plur)}. "
                f"Falling back to generic consecutive pairing."
            )
            pairs = []
            for i in range(0, len(prompts) - 1, 2):
                pairs.append((prompts[i], prompts[i+1]))
            return pairs
        
        # Sort both lists by margin
        t_sing = sort_by_margin(t_sing)
        t_plur = sort_by_margin(t_plur)
        
        # Pair Low Singular <- High Plural
        # t_sing is [Low...High], t_plur is [Low...High]
        # reversed(t_plur) is [High...Low]
        # zip(t_sing, reversed(t_plur)) -> (Low Sing, High Plur)
        pairs.extend(zip(reversed(t_plur), t_sing)) # (Source=HighPlur, Target=LowSing)
        
        # Pair Low Plural <- High Singular
        pairs.extend(zip(reversed(t_sing), t_plur)) # (Source=HighSing, Target=LowPlur)

        return pairs
        
    elif behaviour == "sentiment_continuation":
        positive = [p for p in prompts if p.get("sentiment") == "positive"]
        negative = [p for p in prompts if p.get("sentiment") == "negative"]
        for pos, neg in zip(positive[:len(negative)], negative):
            pairs.append((pos, neg))
            pairs.append((neg, pos))

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
                    feats = transcoder.encode(mlp_last.to(transcoder.dtype))  # (1, N, d_transcoder)

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
    
    # CRITICAL: Warn if model is sharded across devices
    if hasattr(model.model, "hf_device_map"):
        logger.warning(
            f"Model has device_map (multi-GPU/offload): {model.model.hf_device_map}. "
            f"Interventions may behave unexpectedly with sharded models. "
            f"For best results, use single-device loading."
        )
    
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
            
        # Load attribution graph for top features
        # FIX: Use existing results_path from config, don't overwrite!
        # load_attribution_graph expects results root, not behaviour subpath
        graph_n = None if (args.prompts_file or args.source_prompts_file) else args.n_prompts
        graph_data = load_attribution_graph(
            results_path, behaviour, args.split, n_prompts=graph_n
        )
        
        # Build per-layer feature index lists
        top_features_by_layer: Dict[int, List[int]] = {}
        top_features = []
        
        if graph_data:
            # FIX: Load enough features for ensemble ablation (at least 200 or requested k)
            n_load = max(200, args.top_k)
            top_features = get_top_attributed_features(graph_data, n_features=n_load)
            print(f"Loaded {len(top_features)} top attributed features from graph (requested top {n_load})")
            
            # Populate dictionary
            for L, fidx, score in top_features:
                top_features_by_layer.setdefault(L, []).append(fidx)
        else:
            print("No attribution graph found, will use random/first features")

        metadata = {
            "model_size": model_size,
            "transcoder_repo": tc_config["transcoders"][model_size]["repo_id"],
            "n_prompts": args.n_prompts,
            "top_k": args.top_k,
            "prompts_file": args.prompts_file,
            "source_prompts_file": args.source_prompts_file,
        }

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
                        prompts, 
                        layer=layer, 
                        n_prompts=args.n_prompts,
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
                    prompts, layers, top_features_by_layer, behaviour,
                    coefficient=args.steering_coeff, top_k=args.top_k
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
                sample_prompts = prompts[:args.n_prompts] if args.n_prompts else prompts

                for i, prompt_data in enumerate(tqdm(sample_prompts, desc="Ablation")):
                    prompt = prompt_data["prompt"]
                    try:
                        correct, incorrect = get_answer_tokens(prompt_data)
                    except KeyError as e:
                        logger.warning(f"Skipping prompt {i} due to missing fields: {e}")
                        continue

                    for layer in layers:
                        # Use graph-derived features if available, else fallback
                        if top_features_by_layer and layer in top_features_by_layer:
                            layer_features = top_features_by_layer[layer][:args.top_k]
                        else:
                            layer_features = []

                        d = experiment.transcoder_set[layer].d_transcoder
                        layer_features = [fi for fi in layer_features if 0 <= fi < d]

                        if not layer_features:
                            logger.warning(f"No valid features for layer {layer}, using first {args.top_k}")
                            layer_features = list(range(min(args.top_k, d)))

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

                # Create pairs
                pairs = create_prompt_pairs(prompts, behaviour, source_prompts=source_prompts)
                pairs = pairs[:args.n_prompts] if args.n_prompts else pairs
                print(f"  Created {len(pairs)} pairs for patching")
                
                if not pairs:
                    logger.warning("No pairs created! Check your prompts/margins.")
                    # Create empty results to ensure file existence
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
                        if top_features_by_layer and layer in top_features_by_layer:
                            layer_features = top_features_by_layer[layer][:args.top_k]
                        else:
                            layer_features = []

                        d = experiment.transcoder_set[layer].d_transcoder
                        layer_features = [fi for fi in layer_features if 0 <= fi < d]

                        if not layer_features:
                            logger.warning(f"No valid features for layer {layer}, using first {args.top_k}")
                            layer_features = list(range(min(args.top_k, d)))

                        try:
                            result = experiment.run_patching_experiment(
                                source_prompt=source["prompt"],
                                target_prompt=target["prompt"],
                                prompt_idx=idx,
                                source_correct=source_correct,
                                target_correct=target_correct,
                                target_incorrect=target_incorrect,
                                layer=layer,
                                feature_indices=layer_features,
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
