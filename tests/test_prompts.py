"""Tests for the behavior prompts module."""

import pytest

from prompts import (
    get_prompts,
    get_all_prompts,
    BEHAVIOR_PROMPTS,
    FACTUAL_RECALL_PROMPTS,
    REASONING_PROMPTS,
    CODE_GENERATION_PROMPTS,
    MULTILINGUAL_PROMPTS,
)


class TestBehaviorPrompts:
    """Tests for behavior prompts module."""

    def test_factual_recall_prompts_exist(self):
        """Test that factual recall prompts are defined."""
        assert len(FACTUAL_RECALL_PROMPTS) > 0
        assert all(isinstance(p, str) for p in FACTUAL_RECALL_PROMPTS)

    def test_reasoning_prompts_exist(self):
        """Test that reasoning prompts are defined."""
        assert len(REASONING_PROMPTS) > 0
        assert all(isinstance(p, str) for p in REASONING_PROMPTS)

    def test_code_generation_prompts_exist(self):
        """Test that code generation prompts are defined."""
        assert len(CODE_GENERATION_PROMPTS) > 0
        assert all(isinstance(p, str) for p in CODE_GENERATION_PROMPTS)

    def test_multilingual_prompts_exist(self):
        """Test that multilingual prompts are defined."""
        assert len(MULTILINGUAL_PROMPTS) > 0
        assert all(isinstance(p, str) for p in MULTILINGUAL_PROMPTS)

    def test_get_prompts_valid_behavior(self):
        """Test getting prompts for valid behavior."""
        prompts = get_prompts("factual_recall")
        assert prompts == FACTUAL_RECALL_PROMPTS

    def test_get_prompts_invalid_behavior(self):
        """Test getting prompts for invalid behavior raises error."""
        with pytest.raises(ValueError, match="Unknown behavior"):
            get_prompts("nonexistent_behavior")

    def test_get_all_prompts(self):
        """Test getting all prompts."""
        all_prompts = get_all_prompts()
        assert "factual_recall" in all_prompts
        assert "reasoning" in all_prompts
        assert "code_generation" in all_prompts
        assert "multilingual" in all_prompts

    def test_behavior_prompts_dict(self):
        """Test BEHAVIOR_PROMPTS dictionary structure."""
        assert len(BEHAVIOR_PROMPTS) == 4
        for behavior, prompts in BEHAVIOR_PROMPTS.items():
            assert isinstance(behavior, str)
            assert isinstance(prompts, list)
            assert len(prompts) > 0
