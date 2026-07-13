"""Unit tests for transcoder activation functions (src/transcoder/activation_functions.py).

These are pure-tensor tests that run on CPU without the model or GPU, so they are
safe to run in CI.
"""

import torch

from src.transcoder.activation_functions import JumpReLU, ReLU, TopK, rectangle


def test_rectangle_indicator():
    x = torch.tensor([-1.0, -0.5, -0.4, 0.0, 0.4, 0.5, 1.0])
    out = rectangle(x)
    # 1.0 strictly inside (-0.5, 0.5), 0.0 on/outside the boundary
    assert torch.equal(out, torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0]))


def test_jumprelu_forward_thresholds():
    act = JumpReLU(threshold=1.0)
    x = torch.tensor([0.5, 1.0, 1.5, 2.0, -3.0])
    out = act(x)
    # keeps x where x > threshold, else 0 (note: x == threshold is dropped)
    assert torch.equal(out, torch.tensor([0.0, 0.0, 1.5, 2.0, 0.0]))


def test_jumprelu_per_feature_threshold():
    act = JumpReLU(threshold=torch.tensor([0.0, 2.0]))
    x = torch.tensor([[1.0, 1.0], [1.0, 3.0]])
    out = act(x)
    assert torch.equal(out, torch.tensor([[1.0, 0.0], [1.0, 3.0]]))


def test_topk_keeps_k_largest():
    act = TopK(k=2)
    x = torch.tensor([1.0, 3.0, 2.0, 0.0])
    out = act(x)
    # keeps the two largest (3, 2), zeroes the rest
    assert torch.equal(out, torch.tensor([0.0, 3.0, 2.0, 0.0]))


def test_topk_sparsity_count():
    act = TopK(k=3)
    x = torch.arange(10, dtype=torch.float32)
    out = act(x)
    assert int((out != 0).sum()) == 3


def test_relu_matches_torch():
    act = ReLU()
    x = torch.tensor([-2.0, -0.1, 0.0, 0.1, 5.0])
    assert torch.equal(act(x), torch.relu(x))
