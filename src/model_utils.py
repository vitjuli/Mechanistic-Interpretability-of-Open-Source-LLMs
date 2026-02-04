"""
Model loading and inference utilities for Qwen3 models.

This wrapper handles model loading, tokenization, generation, and activation capture
for mechanistic interpretability experiments.

CRITICAL: When using with transcoders, MUST load the exact model variant that
transcoders were trained on (typically base model, NOT instruct variant).
Distribution mismatch between p_base and p_instruct will break feature alignment.

Based on Anthropic's "Scaling Monosemanticity" methodology.
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelWrapper:
    """Wrapper for Qwen model with activation capture capabilities."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-4B-Instruct-2507",
        dtype: str = "bfloat16",
        device: str = "auto",
        trust_remote_code: bool = True,
    ):
        """
        Initialize model and tokenizer.

        Args:
            model_name: HuggingFace model identifier
            dtype: Data type (bfloat16, float16, float32)
            device: Device placement (auto, cuda, cpu)
            trust_remote_code: Whether to trust remote code
        """
        self.model_name = model_name
        self.device = self._setup_device(device)
        self.dtype = self._parse_dtype(dtype)

        logger.info(f"Loading model: {model_name}")
        logger.info(f"Device: {self.device}, dtype: {self.dtype}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )
        
        # CRITICAL: Normalize pad_token_id
        # Some tokenizers (like Qwen) don't have pad_token_id by default
        # Use eos_token_id as fallback to ensure consistent padding behavior
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            logger.info(
                f"Tokenizer has no pad_token_id. Set pad_token_id = eos_token_id ({self.tokenizer.eos_token_id})"
            )
        
        # Ensure pad_token_id is set (fail fast if tokenizer has neither)
        if self.tokenizer.pad_token_id is None:
            raise ValueError(
                f"Tokenizer for {model_name} has neither pad_token_id nor eos_token_id. "
                "Cannot perform padding operations. Please configure the tokenizer properly."
            )

        # Load model
        load_kwargs = {
            "trust_remote_code": trust_remote_code,
        }

        if device == "auto":
            load_kwargs["device_map"] = "auto"
            load_kwargs["torch_dtype"] = self.dtype
        else:
            load_kwargs["torch_dtype"] = self.dtype

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **load_kwargs,
        )

        if device != "auto":
            self.model = self.model.to(self.device)

        self.model.eval()
        
        # Track if we're using device_map (critical for safe input placement)
        # Check if model actually has device_map (more robust than checking device arg)
        self.use_device_map = getattr(self.model, "hf_device_map", None) is not None

        # Store model config
        self.config = self.model.config
        self.num_layers = self.config.num_hidden_layers
        self.hidden_size = self.config.hidden_size

        logger.info(f"Model loaded: {self.num_layers} layers, hidden_size={self.hidden_size}")

    def _setup_device(self, device: str) -> torch.device:
        """Determine computation device."""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def _parse_dtype(self, dtype: str) -> torch.dtype:
        """Parse dtype string to torch.dtype."""
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        return dtype_map.get(dtype, torch.float32)
    
    def _move_inputs(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Move inputs to appropriate device, handling device_map='auto' safely."""
        if self.use_device_map:
            input_device = next(self.model.parameters()).device
            return {k: v.to(input_device) for k, v in inputs.items()}
        return {k: v.to(self.device) for k, v in inputs.items()}

    def _add_special_tokens(self, token_ids: List[int]) -> List[int]:
        """
        Add special tokens (BOS) to a list of token IDs.

        Handles tokenizers that lack build_inputs_with_special_tokens method
        (e.g., Qwen2Tokenizer).

        Args:
            token_ids: List of token IDs without special tokens

        Returns:
            List of token IDs with BOS prepended if applicable
        """
        # Try the standard method first
        if hasattr(self.tokenizer, 'build_inputs_with_special_tokens'):
            try:
                return self.tokenizer.build_inputs_with_special_tokens(token_ids)
            except (AttributeError, TypeError):
                pass

        # Fallback: manually prepend BOS token if it exists
        result = list(token_ids)
        if self.tokenizer.bos_token_id is not None:
            # Only prepend if not already there
            if len(result) == 0 or result[0] != self.tokenizer.bos_token_id:
                result = [self.tokenizer.bos_token_id] + result

        return result

    def tokenize(
        self,
        texts: List[str],
        add_special_tokens: bool = True,
        return_tensors: str = "pt",
        padding: bool = True,
        truncation: bool = False,  # Changed default: safer for interpretability
    ) -> Dict:
        """
        Tokenize input texts.

        Args:
            texts: List of input strings
            add_special_tokens: Whether to add BOS/EOS tokens
            return_tensors: Return format ('pt' for PyTorch tensors)
            padding: Whether to pad sequences to same length
            truncation: Whether to truncate long sequences (default False for safety)

        Returns:
            Dictionary with input_ids, attention_mask
        """
        return self.tokenizer(
            texts,
            add_special_tokens=add_special_tokens,
            return_tensors=return_tensors,
            padding=padding,
            truncation=truncation,
        )

    @torch.no_grad()
    def generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 10,
        temperature: float = 0.0,  # Greedy decoding
        return_dict_in_generate: bool = True,
        output_scores: bool = True,
    ) -> Dict:
        """
        Generate completions for prompts.

        Args:
            prompts: List of prompt strings
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 = greedy)
            return_dict_in_generate: Return detailed outputs
            output_scores: Include logit scores

        Returns:
            Dictionary with generated tokens and scores
        """
        inputs = self.tokenize(prompts)
        inputs = self._move_inputs(inputs)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if temperature > 0 else 1.0,
            do_sample=temperature > 0,
            return_dict_in_generate=return_dict_in_generate,
            output_scores=output_scores,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        return outputs

    @torch.no_grad()
    def get_logits(
        self,
        prompts: List[str],
        target_tokens: Optional[List[str]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Get next-token logits for prompts.

        Args:
            prompts: List of prompt strings
            target_tokens: Optional list of target tokens to extract logits for

        Returns:
            - Full logits tensor (batch, vocab_size)
            - Target token logits if target_tokens provided
        """
        inputs = self.tokenize(prompts)
        inputs = self._move_inputs(inputs)

        outputs = self.model(**inputs)  # Already has @torch.no_grad() decorator
        
        # CRITICAL: Get logits at LAST VALID position (not -1 which may be padding!)
        attention_mask = inputs["attention_mask"]
        last_positions = attention_mask.sum(dim=1) - 1  # (batch,)
        
        # Edge case: empty sequences
        if (last_positions < 0).any():
            raise ValueError("Found empty sequence after tokenization (attention_mask sum == 0).")
        
        # CRITICAL: Use logits device for indexing (safe with device_map)
        logits_all = outputs.logits
        dev = logits_all.device
        batch_indices = torch.arange(len(prompts), device=dev)
        last_positions = last_positions.to(dev)
        logits = logits_all[batch_indices, last_positions, :]  # (batch, vocab_size)

        if target_tokens is not None:
            target_ids = self.tokenizer(
                target_tokens,
                add_special_tokens=False,
            )["input_ids"]
            # Handle multi-token targets (take first token)
            target_ids = [ids[0] if isinstance(ids, list) else ids for ids in target_ids]
            target_logits = logits[:, target_ids]  # (batch, num_targets)
            return logits, target_logits

        return logits, None

    @torch.no_grad()
    def get_sequence_log_probs(
        self,
        prompts: List[str],
        target_sequences: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute log probabilities for target sequences using teacher forcing.
        
        CORRECT IMPLEMENTATION:
        - Tokenizes prompt and targets separately WITHOUT special tokens
        - Concatenates token IDs manually to avoid boundary tokenization issues
        - Adds special tokens once to full sequence
        - Single batched forward pass for efficiency
        
        Args:
            prompts: List of prompts (typically batch_size=1 in baseline)
            target_sequences: List of target sequences (e.g., [correct, incorrect])
            
        Returns:
            log_probs: Tensor of shape (batch_size, num_sequences) 
                       Sum of log probabilities for each sequence
            token_lengths: Tensor of shape (num_sequences,)
                          Number of tokens in each target sequence
        
        Example:
            prompts = ["Capital of France?"]
            targets = [" Paris", " London"]
            log_probs, lengths = model.get_sequence_log_probs(prompts, targets)
            # log_probs.shape = (1, 2), lengths.shape = (2,)
            # log_probs[0, 0] = sum of log P(Paris tokens | prompt)
            # log_probs[0, 1] = sum of log P(London tokens | prompt)
        """
        batch_size = len(prompts)
        num_sequences = len(target_sequences)
        
        # Step 1: Tokenize prompts WITHOUT special tokens
        # We'll add them later to the full sequence
        prompt_token_ids_list = []
        for prompt in prompts:
            ids = self.tokenizer(
                prompt,
                add_special_tokens=False,
                padding=False,
                truncation=False,
            )["input_ids"]
            prompt_token_ids_list.append(ids)
        
        # Step 2: Tokenize targets WITHOUT special tokens
        # This gives us clean token IDs for the target only
        target_token_data = []
        for target in target_sequences:
            ids = self.tokenizer(
                target,
                add_special_tokens=False,
                padding=False,
                truncation=False,
            )["input_ids"]
            target_token_data.append({
                'ids': ids,
                'length': len(ids)
            })
        
        # Store token lengths
        token_lengths = torch.tensor([t['length'] for t in target_token_data], dtype=torch.long)
        
        # Step 3: Manually concatenate prompt + target token IDs
        # Then add special tokens ONCE to the full sequence
        all_input_ids = []
        prompt_lengths = []  # Track where targets start
        
        for prompt_ids in prompt_token_ids_list:
            # Add special tokens (BOS) if tokenizer has them
            # Use helper to handle tokenizers that lack build_inputs_with_special_tokens
            prompt_with_special = self._add_special_tokens(prompt_ids)
            prompt_len = len(prompt_with_special)

            for target_data in target_token_data:
                target_ids = target_data['ids']

                # Concatenate token IDs manually
                combined_ids = prompt_ids + target_ids

                # Add special tokens to the COMBINED sequence
                final_ids = self._add_special_tokens(combined_ids)

                # Validate prompt_with_special is a prefix of final_ids
                if final_ids[:prompt_len] != prompt_with_special:
                    raise ValueError(
                        f"Tokenizer behavior violated assumption: prompt_with_special is not a prefix of final_ids."
                    )

                all_input_ids.append(final_ids)
                prompt_lengths.append(prompt_len)
        
        # Step 4: Convert to tensors and pad
        # Use pad_token_id (normalized in __init__)
        max_len = max(len(ids) for ids in all_input_ids)
        
        # CRITICAL: Use correct device (handles device_map='auto')
        device = next(self.model.parameters()).device if self.use_device_map else self.device
        
        n = batch_size * num_sequences
        input_ids_tensor = torch.full(
            (n, max_len),
            fill_value=self.tokenizer.pad_token_id,
            dtype=torch.long,
            device=device,
        )
        attention_mask_tensor = torch.zeros((n, max_len), dtype=torch.long, device=device)
        
        for i, ids in enumerate(all_input_ids):
            L = len(ids)
            input_ids_tensor[i, :L] = torch.tensor(ids, dtype=torch.long, device=device)
            attention_mask_tensor[i, :L] = 1
        
        # Step 5: Single batched forward pass
        outputs = self.model(
            input_ids=input_ids_tensor,
            attention_mask=attention_mask_tensor,
        )
        all_logits = outputs.logits  # (batch_size * num_sequences, seq_len, vocab_size)
        
        # Step 6: Extract log probabilities
        log_probs = torch.zeros(batch_size, num_sequences, device=device)
        
        idx = 0  # Index into flattened batch
        for batch_idx in range(batch_size):
            for seq_idx in range(num_sequences):
                logits = all_logits[idx]  # (seq_len, vocab_size)
                target_data = target_token_data[seq_idx]
                target_ids = target_data['ids']
                target_len = target_data['length']
                prompt_len = prompt_lengths[idx]
                
                if target_len == 0:
                    logger.warning(f"Target sequence '{target_sequences[seq_idx]}' has zero tokens")
                    log_probs[batch_idx, seq_idx] = float('-inf')
                    idx += 1
                    continue
                
                # Compute log probabilities using teacher forcing
                # CRITICAL: Logits shift in causal LM
                # logits[i] predicts token at position i+1
                #
                # Full sequence: [BOS] [prompt_tokens] [target_tokens] [PAD...]
                # Positions:      0     1...prompt_len-1  prompt_len...prompt_len+target_len-1
                #
                # Target token at position prompt_len+j is predicted by logits[prompt_len+j-1]
                
                log_softmax = torch.nn.functional.log_softmax(logits, dim=-1)
                
                total_log_prob = 0.0
                for j, target_token_id in enumerate(target_ids):
                    # Position in full sequence where this target token appears
                    token_position = prompt_len + j
                    
                    # Logits position that predicts this token (causal shift)
                    logit_position = token_position - 1
                    
                    if logit_position < 0 or logit_position >= len(log_softmax):
                        logger.warning(
                            f"Token position {logit_position} out of range for sequence length {len(log_softmax)}"
                        )
                        continue
                    
                    # Extract log probability for target token
                    token_log_prob = log_softmax[logit_position, target_token_id].item()
                    total_log_prob += token_log_prob
                
                log_probs[batch_idx, seq_idx] = total_log_prob
                idx += 1
        
        # Return token_lengths on CPU for compatibility with baseline script
        return log_probs, token_lengths.cpu()

    @torch.no_grad()
    def capture_mlp_inputs(
        self,
        prompts: List[str],
        layer_range: Tuple[int, int] = (10, 25),
        token_positions: str = "all",
    ) -> Dict:
        """
        Capture MLP input activations (post-LayerNorm, pre-MLP residual stream).
        
        *** CRITICAL FOR TRANSCODERS ***
        Transcoders are trained on MLP inputs, NOT block outputs!
        
        MLP input = LayerNorm(residual_after_attention)
        
        In Qwen3 architecture:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states)
        hidden_states = residual + hidden_states  # post-attention residual
        
        residual = hidden_states
        mlp_input = self.post_attention_layernorm(hidden_states)  ← WE CAPTURE THIS
        mlp_output = self.mlp(mlp_input)
        output = residual + mlp_output
        
        Args:
            prompts: List of prompt strings
            layer_range: (start_layer, end_layer) to capture
            token_positions: Token selection strategy ("all", "last", "last_N")
        
        Returns:
            Dictionary with keys:
                - "activations": Dict[str, np.ndarray]  # layer_{idx}_mlp_input -> activations
                - "metadata": Dict with input_ids, attention_mask, etc.
        """
        # Tokenize inputs
        inputs = self.tokenize(prompts, truncation=True)
        inputs = self._move_inputs(inputs)
        
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        batch_size, seq_len = input_ids.shape
        
        # Warning: if sequences truncated, token_positions="last_N" will be relative to truncated sequence
        max_len = getattr(self.tokenizer, "model_max_length", None)
        if isinstance(max_len, int) and max_len < 1_000_000:
            lengths = attention_mask.sum(dim=1)
            n_at_max = (lengths == max_len).sum().item()
            if n_at_max > 0:
                logger.warning(
                    f"{n_at_max}/{batch_size} sequences at model_max_length={max_len}. "
                    "With truncation=True and token_positions='last_N', last N tokens "
                    "will be relative to TRUNCATED sequence, not original prompt."
                )
        
        # Locate transformer blocks
        blocks = None
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            blocks = self.model.model.layers
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            blocks = self.model.transformer.h
        else:
            raise RuntimeError(
                "Could not locate transformer blocks. "
                "Expected self.model.model.layers or self.model.transformer.h."
            )
        
        start, end = layer_range
        end = min(end, len(blocks))
        if start < 0 or start >= end:
            raise ValueError(f"Invalid layer_range={layer_range} for n_layers={len(blocks)}")
        
        # Storage for captured MLP inputs
        captured_mlp_inputs: Dict[int, torch.Tensor] = {}
        
        def make_mlp_input_hook(li: int):
            def hook(module, inp, out):
                # Hook on post_attention_layernorm
                # out is the LayerNorm output = MLP input
                if isinstance(out, (tuple, list)):
                    out0 = out[0]
                else:
                    out0 = out
                captured_mlp_inputs[li] = out0.detach()
            return hook
        
        handles = []
        try:
            # Register hooks on post_attention_layernorm (MLP input point)
            for li in range(start, end):
                # Qwen3/Qwen2 architecture
                if hasattr(blocks[li], "post_attention_layernorm"):
                    hook_module = blocks[li].post_attention_layernorm
                elif hasattr(blocks[li], "ln_2"):  # GPT-2 style
                    hook_module = blocks[li].ln_2
                else:
                    raise RuntimeError(
                        f"Could not find post-attention layernorm in layer {li}. "
                        f"Available modules: {[name for name, _ in blocks[li].named_modules()][:10]}"
                    )
                
                handles.append(hook_module.register_forward_hook(make_mlp_input_hook(li)))
            
            # Forward pass
            outputs = self.model(**inputs)
        
        finally:
            # Always remove hooks
            for h in handles:
                h.remove()
        
        # Parse token_positions
        last_pos = None
        last_n = None
        
        if token_positions == "last":
            last_pos = attention_mask.sum(dim=1) - 1
        elif token_positions.startswith("last_"):
            try:
                last_n = int(token_positions.split("_")[1])
            except (IndexError, ValueError):
                raise ValueError(f"Invalid token_positions: {token_positions}")
        
        # Build activations dict
        activations = {}
        position_metadata = []
        
        for li in range(start, end):
            if li not in captured_mlp_inputs:
                raise RuntimeError(f"Hook for layer {li} did not fire.")
            
            layer_acts = captured_mlp_inputs[li]  # (batch, seq, hidden)
            
            # Select token positions
            if token_positions == "all":
                # Keep all VALID tokens
                acts_flat = layer_acts.reshape(-1, layer_acts.shape[-1])
                mask = attention_mask.bool().view(-1)
                selected_acts = acts_flat[mask]
                
                if li == start:
                    for i in range(batch_size):
                        valid_length = attention_mask[i].sum().item()
                        for j in range(valid_length):
                            position_metadata.append({
                                "prompt_idx": i,
                                "token_pos": j,
                                "token_id": input_ids[i, j].item(),
                            })
            
            elif token_positions == "last":
                # Last token only
                idx = torch.arange(batch_size, device=layer_acts.device)
                last_pos_device = last_pos.to(layer_acts.device)
                selected_acts = layer_acts[idx, last_pos_device, :]
                
                if li == start:
                    for i in range(batch_size):
                        position_metadata.append({
                            "prompt_idx": i,
                            "token_pos": last_pos[i].item(),
                            "token_id": input_ids[i, last_pos[i]].item(),
                        })
            
            elif token_positions.startswith("last_"):
                # Last N tokens
                # Parse "last_N" format (already done above in token position precompute)
                selected_list = []
                for i in range(batch_size):
                    seq_len_i = attention_mask[i].sum().item()
                    start_pos = max(0, seq_len_i - last_n)
                    end_pos = seq_len_i
                    selected_list.append(layer_acts[i, start_pos:end_pos, :])
                
                selected_acts = torch.cat(selected_list, dim=0)
                
                if li == start:
                    for i in range(batch_size):
                        seq_len_i = attention_mask[i].sum().item()
                        start_pos = max(0, seq_len_i - last_n)
                        end_pos = seq_len_i
                        for j in range(start_pos, end_pos):
                            position_metadata.append({
                                "prompt_idx": i,
                                "token_pos": j,
                                "token_id": input_ids[i, j].item(),
                            })
            
            else:
                raise ValueError(f"Unknown token_positions: {token_positions}")
            
            # Store as float16 for memory efficiency
            if selected_acts.dtype == torch.bfloat16:
                selected_acts = selected_acts.float()
            
            # Use descriptive key to indicate these are MLP inputs
            activations[f"layer_{li}_mlp_input"] = selected_acts.cpu().numpy().astype(np.float16)
        
        # Build metadata
        metadata = {
            "n_prompts": batch_size,
            "n_samples": len(position_metadata),
            "token_selection": token_positions,
            "layer_range": list(layer_range),
            "layer_range_inclusive": False,
            "shapes": {k: list(v.shape) for k, v in activations.items()},
            "position_map": position_metadata,
            "input_ids": input_ids.cpu().numpy(),
            "attention_mask": attention_mask.cpu().numpy(),
            "hook_point": "post_attention_layernorm_output",  # Explicitly document what we captured
            "note": "These are MLP inputs (post-LN, pre-MLP), required for transcoders",
        }
        
        return {
            "activations": activations,
            "metadata": metadata,
        }


    @torch.no_grad()
    def capture_activations(
        self,
        prompts: List[str],
        layer_range: Tuple[int, int] = (10, 25),
        token_positions: str = "all",
        include_logits: bool = False,
        target_answers: Optional[List[tuple]] = None,
    ) -> Dict:
        """
        Capture transformer block output hidden states (residual stream after each block).
        
        MEMORY OPTIMIZATION: Uses forward hooks instead of output_hidden_states=True
        to reduce memory usage by 10x+ (only stores requested layers, not all layers).
        
        Args:
            prompts: List of prompt strings
            layer_range: (start_layer, end_layer) to capture
            token_positions: Token selection strategy:
                - "all": All tokens (default, best for activation capture)
                - "next_token": Next-token prediction position (for attribution analysis)
                - "last": Last token only
                - "last_N" (e.g., "last_5"): Last N tokens
            include_logits: If True, compute logits for target answers
            target_answers: List of (correct_ans, incorrect_ans) tuples
                           Required if include_logits=True
        
        Returns:
            Dictionary with keys:
                - "activations": Dict[str, np.ndarray]  # layer_idx -> (n_samples, d_model)
                - "metadata": Dict with input_ids, attention_mask, token_positions, etc.
                - "logits": Dict (if include_logits=True) with correct, incorrect, delta
        """
        # Tokenize inputs
        # NOTE: Using truncation=True for activation capture to avoid OOM on long sequences
        inputs = self.tokenize(prompts, truncation=True)
        
        # CRITICAL: Safe device placement for device_map="auto"
        # When using device_map, model may be sharded across devices,
        # so we place inputs on the device of the first parameter
        if self.use_device_map:
            # Get device from first model parameter (works with device_map)
            input_device = next(self.model.parameters()).device
            inputs = {k: v.to(input_device) for k, v in inputs.items()}
        else:
            # Standard single-device placement
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        batch_size, seq_len = input_ids.shape
        
        # Info: sequences at model_max_length (honest, non-committal warning)
        max_len = getattr(self.tokenizer, "model_max_length", None)
        # Check for reasonable max_len (some tokenizers set to 1e30 or it's not an int)
        if isinstance(max_len, int) and max_len < 1_000_000:
            lengths = attention_mask.sum(dim=1)
            n_at_max = (lengths == max_len).sum().item()
            if n_at_max > 0:
                logger.warning(
                    f"{n_at_max}/{batch_size} sequences have length==model_max_length={max_len}. "
                    "With truncation=True, some examples may have been truncated."
                )
        
        # --- Locate transformer blocks robustly ---
        # Most HF causal LMs expose blocks at model.model.layers or model.transformer.h
        blocks = None
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            blocks = self.model.model.layers
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            blocks = self.model.transformer.h
        else:
            raise RuntimeError(
                "Could not locate transformer blocks. "
                "Expected self.model.model.layers or self.model.transformer.h. "
                "Inspect model architecture with print(self.model) / named_modules()."
            )
        
        start, end = layer_range
        end = min(end, len(blocks))
        if start < 0 or start >= end:
            raise ValueError(f"Invalid layer_range={layer_range} for n_layers={len(blocks)}")
        
        # Storage for captured outputs: layer_idx -> tensor(batch, seq, hidden)
        captured: Dict[int, torch.Tensor] = {}
        
        def make_hook(li: int):
            def hook(module, inp, out):
                # out can be Tensor or tuple; HF blocks often return Tensor or (Tensor, ...)
                if isinstance(out, (tuple, list)):
                    out0 = out[0]
                else:
                    out0 = out
                # Store detached tensor (keep on device for now)
                captured[li] = out0.detach()
            return hook
        
        handles = []
        try:
            # Register hooks on requested layers only
            for li in range(start, end):
                handles.append(blocks[li].register_forward_hook(make_hook(li)))
            
            # Forward pass (no output_hidden_states=True → saves memory!)
            outputs = self.model(**inputs)
        
        finally:
            # Always remove hooks
            for h in handles:
                h.remove()
        
        # Build activations dict
        activations = {}
        position_metadata = []  # NOTE: For very large datasets, can optimize to 3 numpy arrays
                                # (prompt_idx, token_pos, token_id) instead of list of dicts
        
        # Token position selection precompute
        last_pos = None  # Initialize to avoid UnboundLocalError
        last_n = None
        
        if token_positions == "next_token" or token_positions == "last":
            # Both next_token and last need last position
            last_pos = attention_mask.sum(dim=1) - 1
            if (last_pos < 0).any():
                raise ValueError("Found empty sequence after tokenization (attention_mask sum == 0).")
        
        elif token_positions.startswith("last_"):
            # Parse "last_N" format
            try:
                last_n = int(token_positions.split("_")[1])
            except (IndexError, ValueError):
                raise ValueError(f"Invalid token_positions format: {token_positions}. Expected 'last_N' where N is an integer.")
            
            # Get sequence lengths
            seq_lengths = attention_mask.sum(dim=1)
            if (seq_lengths < 1).any():
                raise ValueError("Found empty sequence after tokenization.")
        
        # Process each captured layer
        for li in range(start, end):
            if li not in captured:
                raise RuntimeError(f"Hook for layer {li} did not fire. Check block indexing.")
            
            layer_acts = captured[li]  # (batch, seq, hidden)
            
            # Select token positions
            if token_positions == "all":
                # Keep all VALID tokens (exclude padding)
                acts_flat = layer_acts.reshape(-1, layer_acts.shape[-1])
                mask = attention_mask.bool().view(-1)
                selected_acts = acts_flat[mask]
                
                # Build metadata only for first layer
                if li == start:
                    for i in range(batch_size):
                        valid_length = attention_mask[i].sum().item()
                        for j in range(valid_length):
                            position_metadata.append({
                                "prompt_idx": i,
                                "token_pos": j,
                                "token_id": input_ids[i, j].item(),
                            })
            
            elif token_positions == "next_token" or token_positions == "last":
                # Next-token prediction position or last token
                # CRITICAL: Use layer_acts.device for indexing (safe with device_map)
                idx = torch.arange(batch_size, device=layer_acts.device)
                last_pos_device = last_pos.to(layer_acts.device)
                selected_acts = layer_acts[idx, last_pos_device, :]
                
                if li == start:
                    for i in range(batch_size):
                        position_metadata.append({
                            "prompt_idx": i,
                            "token_pos": last_pos[i].item(),
                            "token_id": input_ids[i, last_pos[i]].item(),
                            "is_last_token": True,
                        })
            
            elif token_positions.startswith("last_"):
                # Last N tokens
                selected_list = []
                for i in range(batch_size):
                    seq_len = attention_mask[i].sum().item()
                    start_pos = max(0, seq_len - last_n)
                    end_pos = seq_len
                    selected_list.append(layer_acts[i, start_pos:end_pos, :])
                
                selected_acts = torch.cat(selected_list, dim=0)
                
                if li == start:
                    for i in range(batch_size):
                        seq_len = attention_mask[i].sum().item()
                        start_pos = max(0, seq_len - last_n)
                        end_pos = seq_len
                        for j in range(start_pos, end_pos):
                            position_metadata.append({
                                "prompt_idx": i,
                                "token_pos": j,
                                "token_id": input_ids[i, j].item(),
                            })
            
            else:
                raise ValueError(f"Unknown token_positions: {token_positions}")
            
            # Store as float16 for memory efficiency
            # NOTE: Convert to float32 first if bfloat16 (not compatible with numpy)
            if selected_acts.dtype == torch.bfloat16:
                selected_acts = selected_acts.float()
            
            activations[f"layer_{li}"] = selected_acts.cpu().numpy().astype(np.float16)
        
        # Build metadata
        metadata = {
            "n_prompts": batch_size,
            "n_samples": len(position_metadata),
            "token_selection": token_positions,
            "layer_range": list(layer_range),
            "layer_range_inclusive": False,  # range is [start, end) - half-open
            "shapes": {k: list(v.shape) for k, v in activations.items()},
            "position_map": position_metadata,
            "input_ids": input_ids.cpu().numpy(),
            "attention_mask": attention_mask.cpu().numpy(),
            "hook_point": "transformer_block_output",
        }
        
        # CRITICAL: Verify n_samples consistency
        first_layer = f"layer_{start}"
        if first_layer in activations:
            assert activations[first_layer].shape[0] == metadata["n_samples"], \
                f"n_samples mismatch: activations[{first_layer}].shape[0]={activations[first_layer].shape[0]} != len(position_map)={metadata['n_samples']}"
        
        result = {
            "activations": activations,
            "metadata": metadata,
        }
        
        # Optionally compute logits
        if include_logits:
            if target_answers is None:
                raise ValueError("target_answers required when include_logits=True")
            
            if len(target_answers) != batch_size:
                raise ValueError(f"Expected {batch_size} target_answers, got {len(target_answers)}")
            
            logits_data = self._compute_target_logits(
                target_answers, outputs, attention_mask  # Pass attention_mask
            )
            result["logits"] = logits_data
        
        return result
    
    def _compute_target_logits(
        self,
        target_answers: List[tuple],
        outputs,
        attention_mask: torch.Tensor,  # NEW parameter
    ) -> Dict[str, np.ndarray]:
        """
        Compute logits for correct/incorrect answers at next-token position.
        
        CRITICAL: Uses attention_mask to find last valid position (not -1!).
        With padding, logits[:, -1] would be on padding tokens (wrong).
        
        Args:
            target_answers: List of (correct, incorrect) tuples
            outputs: Model outputs from forward pass
            attention_mask: Attention mask to find last valid token position
        
        Returns:
            Dict with "correct", "incorrect", "delta" arrays (float32)
        """
        logits = outputs.logits  # (batch, seq_len, vocab_size)
        batch_size = len(target_answers)
        
        # Get last valid position for each example
        last_positions = attention_mask.sum(dim=1) - 1  # (batch_size,)
        
        # Edge case: empty sequences
        if (last_positions < 0).any():
            raise ValueError("Found empty sequence after tokenization (attention_mask sum == 0).")
        
        logits_correct = []
        logits_incorrect = []
        
        for i, (correct_ans, incorrect_ans) in enumerate(target_answers):
            # CRITICAL: Get logits at LAST VALID position (not -1!)
            # With padding: -1 = padding token (wrong)
            # Correct: last_positions[i] = last non-padding token
            last_pos = last_positions[i].item()
            last_logits = logits[i, last_pos, :]  # (vocab_size,)
            
            # CRITICAL: Tokenize answers WITHOUT special tokens
            # Answer tokenization is context-dependent (e.g., " A" vs "A")
            # For A/B tasks, typically need space: " A" and " B"
            correct_ids = self.tokenizer(correct_ans, add_special_tokens=False)["input_ids"]
            incorrect_ids = self.tokenizer(incorrect_ans, add_special_tokens=False)["input_ids"]
            
            # Warn if multi-token (using only first token)
            if len(correct_ids) > 1 or len(incorrect_ids) > 1:
                logger.warning(
                    f"Multi-token answer at prompt {i}: "
                    f"correct={correct_ans} ({len(correct_ids)} tokens), "
                    f"incorrect={incorrect_ans} ({len(incorrect_ids)} tokens). "
                    f"Using first token only for logit extraction."
                )
            
            # Take first token logit (for multi-token answers)
            correct_logit = last_logits[correct_ids[0]].item() if correct_ids else float("-inf")
            incorrect_logit = last_logits[incorrect_ids[0]].item() if incorrect_ids else float("-inf")
            
            logits_correct.append(correct_logit)
            logits_incorrect.append(incorrect_logit)
        
        logits_correct = np.array(logits_correct, dtype=np.float32)
        logits_incorrect = np.array(logits_incorrect, dtype=np.float32)
        
        return {
            "correct": logits_correct,
            "incorrect": logits_incorrect,
            "delta": logits_correct - logits_incorrect,
        }

    def decode(self, token_ids: torch.Tensor) -> List[str]:
        """Decode token IDs to strings."""
        return self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)
