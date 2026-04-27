"""
Tests for ACT (Action Chunking with Transformers) policy.

Tests forward pass, prediction, gradients, and checkpoint round-trip
using a small config with mocked DINOv2 for speed.
"""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
import torch
import torch.nn as nn

from glove.policy.act import ACTPolicy, ACTConfig, DINOv2Encoder, CVAEEncoder


# DINOv2 is large and needs internet — mock it for unit tests
class FakeDINO(nn.Module):
    """Minimal DINOv2 stand-in that outputs deterministic patch tokens."""

    def __init__(self, embed_dim=384, patch_size=14):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        # Deterministic linear projection instead of random noise
        self.proj = nn.Linear(3, embed_dim, bias=False)

    def forward_features(self, x):
        B, C, H, W = x.shape
        n_h, n_w = H // self.patch_size, W // self.patch_size
        # Average pool each patch to get a deterministic representation
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.contiguous().view(B, C, n_h, n_w, -1).mean(dim=-1)  # (B, C, n_h, n_w)
        x = x.permute(0, 2, 3, 1).reshape(B, n_h * n_w, C)  # (B, N, C)
        patch_tokens = self.proj(x)  # (B, N, embed_dim)
        return {"x_norm_patchtokens": patch_tokens}


@pytest.fixture
def small_act_config():
    """Small ACT config for fast tests."""
    return ACTConfig(
        obs_dim=23,
        action_dim=21,
        image_size=98,  # 98/14 = 7 → 49 patches
        obs_horizon=2,
        action_horizon=8,
        latent_dim=8,
        hidden_dim=32,
        nhead=2,
        num_encoder_layers=1,
        num_decoder_layers=1,
        feedforward_dim=64,
        dropout=0.0,
        num_tasks=2,
    )


@pytest.fixture
def act_policy(small_act_config):
    """ACT policy with mocked DINOv2."""
    with patch("glove.policy.act._load_dinov2", return_value=FakeDINO()):
        policy = ACTPolicy(small_act_config)
    return policy


@pytest.fixture
def sample_batch(small_act_config):
    """Random batch matching expected tensor shapes."""
    B = 2
    T = small_act_config.obs_horizon
    H = small_act_config.action_horizon
    images = torch.randn(B, T, 3, small_act_config.image_size, small_act_config.image_size)
    proprio = torch.randn(B, T, small_act_config.obs_dim)
    actions = torch.randn(B, H, small_act_config.action_dim)
    return images, proprio, actions


class TestACTForward:
    def test_forward_returns_loss_dict(self, act_policy, sample_batch):
        images, proprio, actions = sample_batch
        result = act_policy(images, proprio, actions)

        assert "loss" in result
        assert "l1_loss" in result
        assert "kl_loss" in result

    def test_loss_is_scalar(self, act_policy, sample_batch):
        images, proprio, actions = sample_batch
        result = act_policy(images, proprio, actions)

        assert result["loss"].dim() == 0
        assert result["l1_loss"].dim() == 0
        assert result["kl_loss"].dim() == 0

    def test_kl_loss_nonnegative(self, act_policy, sample_batch):
        images, proprio, actions = sample_batch
        result = act_policy(images, proprio, actions)

        assert result["kl_loss"].item() >= 0

    def test_loss_is_finite(self, act_policy, sample_batch):
        images, proprio, actions = sample_batch
        result = act_policy(images, proprio, actions)

        assert torch.isfinite(result["loss"])


class TestACTPredict:
    def test_predict_shape(self, act_policy, sample_batch, small_act_config):
        images, proprio, _ = sample_batch
        pred = act_policy.predict(images, proprio)

        assert pred.shape == (2, small_act_config.action_horizon, small_act_config.action_dim)

    def test_predict_is_deterministic(self, act_policy, sample_batch):
        """With z=0, predict should be deterministic."""
        images, proprio, _ = sample_batch
        pred1 = act_policy.predict(images, proprio)
        pred2 = act_policy.predict(images, proprio)

        torch.testing.assert_close(pred1, pred2)

    def test_predict_is_finite(self, act_policy, sample_batch):
        images, proprio, _ = sample_batch
        pred = act_policy.predict(images, proprio)

        assert torch.isfinite(pred).all()


class TestACTGradients:
    def test_gradients_flow(self, act_policy, sample_batch):
        images, proprio, actions = sample_batch
        result = act_policy(images, proprio, actions)
        result["loss"].backward()

        # Check that non-frozen parameters have gradients
        trainable_with_grad = 0
        trainable_total = 0
        for name, param in act_policy.named_parameters():
            if param.requires_grad:
                trainable_total += 1
                if param.grad is not None:
                    trainable_with_grad += 1

        assert trainable_with_grad > 0, "No trainable parameters received gradients"
        # task_embed has num_tasks entries but only task_id=0 is used,
        # so unused embedding rows won't get gradients. Allow small gap.
        assert trainable_with_grad >= trainable_total - 1, (
            f"Only {trainable_with_grad}/{trainable_total} trainable params got gradients"
        )

    def test_dino_frozen(self, act_policy, sample_batch):
        """DINOv2 backbone should not have gradients."""
        images, proprio, actions = sample_batch
        result = act_policy(images, proprio, actions)
        result["loss"].backward()

        for param in act_policy.dino_encoder.dino.parameters():
            assert param.grad is None or (param.grad == 0).all()


class TestACTCheckpoint:
    def test_save_load_roundtrip(self, act_policy, small_act_config, sample_batch):
        images, proprio, _ = sample_batch

        # Get predictions before save
        pred_before = act_policy.predict(images, proprio)

        # Save
        tmp = Path(tempfile.mkdtemp())
        try:
            ckpt_path = tmp / "test_ckpt.pt"
            torch.save({
                "model": act_policy.state_dict(),
                "config": vars(small_act_config),
                "policy_type": "act",
            }, ckpt_path)

            # Load into new policy
            with patch("glove.policy.act._load_dinov2", return_value=FakeDINO()):
                new_policy = ACTPolicy(small_act_config)
            checkpoint = torch.load(ckpt_path, weights_only=False)
            new_policy.load_state_dict(checkpoint["model"])

            pred_after = new_policy.predict(images, proprio)
            torch.testing.assert_close(pred_before, pred_after)
        finally:
            shutil.rmtree(tmp)


class TestACTConfig:
    def test_default_config(self):
        config = ACTConfig()
        assert config.action_horizon == 300
        assert config.latent_dim == 32
        assert config.kl_weight == 10.0
        assert config.num_tasks == 1

    def test_config_to_dict(self):
        config = ACTConfig(hidden_dim=128)
        d = vars(config)
        assert d["hidden_dim"] == 128
        assert "latent_dim" in d


class TestCVAEEncoder:
    def test_output_shapes(self, small_act_config):
        encoder = CVAEEncoder(small_act_config)
        B, H = 2, small_act_config.action_horizon
        actions = torch.randn(B, H, small_act_config.action_dim)
        context = torch.randn(B, 10, small_act_config.hidden_dim)

        mu, logvar = encoder(actions, context)

        assert mu.shape == (B, small_act_config.latent_dim)
        assert logvar.shape == (B, small_act_config.latent_dim)
