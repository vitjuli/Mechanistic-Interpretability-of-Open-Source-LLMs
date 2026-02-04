"""
Transcoder module for loading and using pre-trained transcoders.

This module provides utilities for working with pre-trained transcoders from
the circuit-tracer project (https://github.com/safety-research/circuit-tracer).

Available transcoders for Qwen3 models:
- mwhanna/qwen3-0.6b-transcoders-lowl0
- mwhanna/qwen3-1.7b-transcoders-lowl0
- mwhanna/qwen3-4b-transcoders
- mwhanna/qwen3-8b-transcoders
- mwhanna/qwen3-14b-transcoders-lowl0
"""

from src.transcoder.activation_functions import JumpReLU, TopK
from src.transcoder.single_layer_transcoder import SingleLayerTranscoder
from src.transcoder.transcoder_loader import (
    TranscoderSet,
    load_transcoder_set,
    download_transcoder_weights,
)

__all__ = [
    "JumpReLU",
    "TopK",
    "SingleLayerTranscoder",
    "TranscoderSet",
    "load_transcoder_set",
    "download_transcoder_weights",
]
