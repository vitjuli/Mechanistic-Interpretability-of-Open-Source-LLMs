"""
Activation functions for transcoders.

Implements JumpReLU and other sparsity-inducing activation functions used
in per-layer transcoders (PLTs) from the circuit-tracer project.

Reference:
- https://github.com/safety-research/circuit-tracer
- Anthropic's "On the Biology of a Large Language Model" (2025)
"""

from typing import Any
import torch
import torch.nn as nn


def rectangle(x: torch.Tensor) -> torch.Tensor:
    """
    Rectangle function for JumpReLU gradient computation.

    Returns 1.0 where x is in (-0.5, 0.5), else 0.0.
    Used to compute the gradient of the threshold parameter.

    Args:
        x: Input tensor

    Returns:
        Binary tensor indicating values in the rectangle
    """
    return ((x > -0.5) & (x < 0.5)).to(x.dtype)


class _JumpReLUFunction(torch.autograd.Function):
    """
    Custom autograd function for JumpReLU with learnable threshold.

    JumpReLU(x; theta) = x if x > theta, else 0

    Unlike standard ReLU (theta=0), JumpReLU creates a hard gap between
    zero and the smallest non-zero activation, producing cleaner sparsity.
    """

    @staticmethod
    def forward(x: torch.Tensor, threshold: torch.Tensor, bandwidth: float) -> torch.Tensor:
        """
        Forward pass: apply thresholded ReLU.

        Args:
            x: Pre-activation values
            threshold: Per-feature thresholds (learnable)
            bandwidth: Bandwidth for gradient approximation

        Returns:
            x where x > threshold, else 0
        """
        return (x * (x > threshold)).to(x.dtype)

    @staticmethod
    def setup_context(ctx: Any, inputs: tuple, output: torch.Tensor) -> None:
        """Save tensors needed for backward pass."""
        x, threshold, bandwidth = inputs
        ctx.save_for_backward(x, threshold)
        ctx.bandwidth = bandwidth

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple:
        """
        Backward pass with straight-through estimator for threshold.

        The threshold gradient uses a rectangular kernel to approximate
        the derivative of the Heaviside step function.
        """
        x, threshold = ctx.saved_tensors
        bandwidth = ctx.bandwidth

        # Gradient w.r.t. x: pass through where x > threshold
        x_grad = (x > threshold) * grad_output

        # Gradient w.r.t. threshold: use rectangle approximation
        # This allows learning the optimal threshold per feature
        threshold_grad = torch.sum(
            -(threshold / bandwidth) * rectangle((x - threshold) / bandwidth) * grad_output,
            dim=0,
        )

        return x_grad, threshold_grad, None


class JumpReLU(nn.Module):
    """
    JumpReLU activation with learnable per-feature thresholds.

    JumpReLU(x; theta) = x if x > theta, else 0

    This activation function produces sparser representations than ReLU
    by requiring activations to exceed a positive threshold theta > 0.
    The threshold is learnable during training but fixed for pre-trained
    transcoders.

    Attributes:
        threshold: Per-feature threshold tensor (n_features,)
        bandwidth: Bandwidth for gradient approximation (default: 2.0)

    Reference:
        Originally introduced in Anthropic's transcoder work for cleaner
        feature decomposition than standard ReLU.
    """

    def __init__(
        self,
        threshold: float | torch.Tensor,
        bandwidth: float = 2.0,
    ) -> None:
        """
        Initialize JumpReLU activation.

        Args:
            threshold: Initial threshold value(s). Can be a scalar (broadcast
                       to all features) or a tensor (per-feature thresholds).
            bandwidth: Bandwidth for gradient approximation of the threshold.
                       Larger values give smoother gradients.
        """
        super().__init__()

        if not isinstance(threshold, torch.Tensor):
            threshold = torch.tensor(threshold)

        self.threshold = nn.Parameter(threshold)
        self.bandwidth = bandwidth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply JumpReLU activation.

        Args:
            x: Pre-activation values (..., n_features)

        Returns:
            Activated values with same shape as x
        """
        return _JumpReLUFunction.apply(x, self.threshold, self.bandwidth)

    def extra_repr(self) -> str:
        """String representation for printing."""
        return f"threshold_shape={tuple(self.threshold.shape)}, bandwidth={self.bandwidth}"


class TopK(nn.Module):
    """
    Top-K activation that keeps only the k largest values per sample.

    All values except the top k are set to zero. This provides an
    alternative sparsity mechanism to threshold-based activations.

    Attributes:
        k: Number of values to keep per sample
    """

    def __init__(self, k: int):
        """
        Initialize TopK activation.

        Args:
            k: Number of top values to keep per sample
        """
        super().__init__()
        self.k = k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply TopK activation.

        Args:
            x: Input tensor (..., n_features)

        Returns:
            Tensor with only top-k values per sample, others zeroed
        """
        _, indices = torch.topk(x, k=self.k, dim=-1)
        gate = torch.zeros_like(x)
        gate.scatter_(dim=-1, index=indices, value=1)
        return x * gate.to(x.dtype)

    def extra_repr(self) -> str:
        """String representation for printing."""
        return f"k={self.k}"


class ReLU(nn.Module):
    """
    Standard ReLU activation (JumpReLU with threshold=0).

    Included for compatibility with SAE-style transcoders that use
    ReLU instead of JumpReLU.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ReLU activation."""
        return torch.relu(x)
