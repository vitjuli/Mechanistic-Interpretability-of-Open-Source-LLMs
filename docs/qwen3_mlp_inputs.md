"""
Documentation for Qwen3 model structure and MLP input capture.

## Qwen3 Transformer Block Structure

For Qwen3, each transformer block has the following structure:

```python
class Qwen2DecoderLayer(nn.Module):
    def forward(self, hidden_states, ...):
        residual = hidden_states
        
        # Attention sublayer
        hidden_states = self.input_layernorm(hidden_states)  # LN before attn
        hidden_states = self.self_attn(hidden_states, ...)  # Attention
        hidden_states = residual + hidden_states  # Residual connection
        
        residual = hidden_states
        
        # MLP sublayer
        hidden_states = self.post_attention_layernorm(hidden_states)  # LN before MLP ← THIS IS MLP INPUT!
        hidden_states = self.mlp(hidden_states)  # MLP
        hidden_states = residual + hidden_states  # Residual connection
        
        return hidden_states  # This is block output
```

## What Transcoders Need

Transcoders are trained on MLP inputs, specifically:
```
MLP_input = LayerNorm(residual_after_attention)
          = self.post_attention_layernorm(hidden_states)
```

## Current vs Needed Hooks

**Current `capture_activations()`:**
- Hooks: `blocks[l].register_forward_hook()`
- Captures: `out[0]` = final block output = residual after MLP
- Formula: `r^(l+1) = r^(l) + Attn(...) + MLP(...)`
- **WRONG for transcoders!**

**Needed `capture_mlp_inputs()`:**
- Hooks: `blocks[l].post_attention_layernorm.register_forward_hook()`
- Captures: `out` = LayerNorm output = MLP input
- Formula: `h^(l) = LN(r^(l) + Attn(...))`
- **CORRECT for transcoders!**

## Module Names in Qwen3

Based on Qwen2 architecture (Qwen3 uses similar structure):
- Block: `model.layers[i]` (Qwen2DecoderLayer)
- Post-attention LN: `model.layers[i].post_attention_layernorm`
- MLP: `model.layers[i].mlp`

## Implementation Strategy

Add new method `capture_mlp_inputs()` that:
1. Hooks `post_attention_layernorm` output (not block output)
2. Returns pre-MLP activations
3. Otherwise same interface as `capture_activations()`
