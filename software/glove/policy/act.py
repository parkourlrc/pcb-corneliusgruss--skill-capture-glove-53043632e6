"""ACT (Action Chunking with Transformers) — glove adaptation.

Based on Zhao et al. "Learning Fine-Grained Bimanual Manipulation with Low-Cost
Hardware" (RSS 2023). Sunday Robotics' foundation model ACT-1 is a direct
descendant of this architecture — training on their glove corpus instead of
teleop trajectories.

Glove specifics vs original ACT:
- **Single camera** (GoPro Hero 10), not bimanual stereo pairs.
- **Hand configuration is raw Hall Bx/By/Bz** per sensor (12D), not atan2
  joint angles. Bx/By naturally encodes angle as (cos, sin) with no 2π seam;
  Bz carries contact-force information — both useful to the policy.
- **Action = relative 6-DoF pose delta + absolute Hall target**, each
  future step expressed in the body frame at the current observation time.
  Relativization bounds the network input range and keeps the policy
  invariant to where the glove happens to be in world frame.

Dimensions:
    obs_dim    = 23  =  rel_pos (3)  +  rot6d (6)  +  hall_raw (4×3=12)  +  tof (2)
    action_dim = 21  =  rel_pos (3)  +  rot6d (6)  +  hall_raw (4×3=12)

Components:
- Visual encoder: DINOv2 ViT-S/14 (frozen) → patch tokens, projected to hidden_dim
- CVAE encoder (training only): Transformer over [CLS] + action tokens + obs context → latent z
- Policy decoder: DETR-style Transformer decoder; action queries cross-attend to context
- Task embedding + temporal embedding, no camera embedding (single cam)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


DINO_PATCH_SIZE = 14
DINO_EMBED_DIM = 384  # ViT-S


@dataclass
class ACTConfig:
    """ACT hyperparameters for the glove setup."""

    # Observation / action dimensions (glove schema, see module docstring)
    obs_dim: int = 23
    action_dim: int = 21
    image_size: int = 224           # multiple of DINO_PATCH_SIZE (14)
    # 300 frames = 5s at 60fps. Long horizon gives the policy implicit "memory":
    # for a pick-and-place where the target may go out of frame mid-task, a
    # single query when both objects are visible plans the entire trajectory
    # up front, so nothing depends on later frames seeing the target again.
    action_horizon: int = 300
    obs_horizon: int = 2            # past observation frames conditioned on
    # Pose target semantics for action[:, :9]:
    # - "relative": each future pose is expressed relative to the current obs pose.
    # - "delta": each step is the SE(3) increment from the previous frame to the
    #   current frame, with step 0 = identity. This makes long-horizon targets
    #   local-motion-like instead of asking one decoder query to regress the full
    #   5 s endpoint magnitude directly.
    action_pose_mode: str = "relative"

    # CVAE
    latent_dim: int = 32
    kl_weight: float = 10.0
    # Free-bits trick (v7+): floor each latent dim's KL contribution at this
    # threshold (in nats) before applying kl_weight. Standard fix for posterior
    # collapse — guarantees the encoder is allowed to use ≥`free_bits * D`
    # nats "for free" without paying KL pressure. Set 0 to disable.
    # v4/v5 collapsed to total KL ≈ 0.0007 across 32 dims ≈ 2e-5 per dim. With
    # free_bits=0.01 we force ≥0.32 nats total → 450× more capacity available.
    free_bits: float = 0.0

    # Magnitude-undershoot fix (v6+): the per-frame L1 baseline averages
    # over the chunk and produces median-magnitude predictions when the
    # CVAE collapses. These two terms add explicit pressure to commit to
    # the *endpoint* of each motion (where the goal lives in mug-coaster:
    # "mug placed on coaster") and to weight late-chunk position L1
    # higher, since that's where the undershoot empirically lives.
    # Position = first 3 dims of the action vector.
    endpoint_pos_weight: float = 0.0   # multiplier on |pred[end] - gt[end]|_pos
    ramp_pos_weight: float = 0.0       # multiplier on linspace(0,1,H) * |pred-gt|_pos

    # Transformer backbone sizes (mirror ACT paper defaults)
    hidden_dim: int = 256
    nhead: int = 8
    num_encoder_layers: int = 4     # CVAE encoder depth
    num_decoder_layers: int = 7     # DETR decoder depth
    feedforward_dim: int = 2048
    dropout: float = 0.1

    # Multi-task support: per-task learned embedding prepended to context
    num_tasks: int = 1

    # Visual backbone
    freeze_backbone: bool = True


DINOV2_VITS14_URL = (
    "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth"
)


def _build_vit_small() -> nn.Module:
    """Build DINOv2 ViT-S/14 architecture. Uses cached torch.hub repo when available."""
    import importlib
    import pathlib
    import sys

    hub_dir = torch.hub.get_dir()
    cached = list(pathlib.Path(hub_dir).glob("facebookresearch_dinov2_*"))
    if cached:
        repo_path = str(sorted(cached)[-1])
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)
        hub_module = importlib.import_module("hubconf")
        return hub_module.dinov2_vits14(pretrained=False)

    return torch.hub.load(
        "facebookresearch/dinov2", "dinov2_vits14",
        pretrained=False, trust_repo=True,
    )


def _load_dinov2(freeze: bool = True) -> nn.Module:
    """Load DINOv2 ViT-S/14 with pretrained weights. Weights cached under $TORCH_HOME."""
    model = _build_vit_small()
    state = torch.hub.load_state_dict_from_url(
        DINOV2_VITS14_URL, map_location="cpu", check_hash=False,
    )
    model.load_state_dict(state, strict=True)
    if freeze:
        for param in model.parameters():
            param.requires_grad_(False)
        model.eval()
    return model


def prefetch_dinov2() -> None:
    """Warm the DINOv2 cache. Run once on a node with internet before submitting GPU jobs."""
    print("Prefetching DINOv2 ViT-S/14 repo …")
    torch.hub.load("facebookresearch/dinov2", "dinov2_vits14",
                   pretrained=False, trust_repo=True)
    print("Prefetching DINOv2 ViT-S/14 weights …")
    torch.hub.load_state_dict_from_url(DINOV2_VITS14_URL, map_location="cpu")
    print("Done — cached under", torch.hub.get_dir())


class DINOv2Encoder(nn.Module):
    """Projects DINOv2 patch tokens to policy hidden_dim."""

    def __init__(self, hidden_dim: int, freeze: bool = True):
        super().__init__()
        self.dino = _load_dinov2(freeze=freeze)
        self.proj = nn.Linear(DINO_EMBED_DIM, hidden_dim)
        self.freeze = freeze

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) → (B, num_patches, hidden_dim)."""
        if self.freeze:
            with torch.no_grad():
                features = self.dino.forward_features(x)
                patch_tokens = features["x_norm_patchtokens"]
        else:
            features = self.dino.forward_features(x)
            patch_tokens = features["x_norm_patchtokens"]
        return self.proj(patch_tokens)


class CVAEEncoder(nn.Module):
    """Encodes (actions, obs_context) → latent z for style conditioning.

    Training: true action chunk is encoded, z sampled via reparameterization.
    Inference: z = 0 (prior mean), no posterior needed.
    """

    def __init__(self, config: ACTConfig):
        super().__init__()
        self.latent_dim = config.latent_dim
        self.action_proj = nn.Linear(config.action_dim, config.hidden_dim)
        self.action_pos = nn.Embedding(config.action_horizon, config.hidden_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim, nhead=config.nhead,
            dim_feedforward=config.feedforward_dim, dropout=config.dropout,
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_encoder_layers)
        self.mu_head = nn.Linear(config.hidden_dim, config.latent_dim)
        self.logvar_head = nn.Linear(config.hidden_dim, config.latent_dim)

    def forward(self, actions: torch.Tensor, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, H = actions.shape[:2]
        action_tokens = self.action_proj(actions)
        positions = torch.arange(H, device=actions.device)
        action_tokens = action_tokens + self.action_pos(positions).unsqueeze(0)
        cls = self.cls_token.expand(B, -1, -1)
        sequence = torch.cat([cls, action_tokens, context], dim=1)
        encoded = self.encoder(sequence)
        cls_out = encoded[:, 0]
        return self.mu_head(cls_out), self.logvar_head(cls_out)


class ACTPolicy(nn.Module):
    """ACT visuomotor policy for the glove. Predicts `action_horizon` future actions
    from `obs_horizon` past observations (image + proprio).

    Uses CVAE for style-conditioned multimodal action distributions and a
    DETR-style transformer decoder with learned action queries.
    """

    def __init__(self, config: ACTConfig):
        super().__init__()
        if config.action_pose_mode not in ("relative", "delta"):
            raise ValueError(f"Unsupported action_pose_mode={config.action_pose_mode!r}")
        if config.action_pose_mode != "relative" and (
            config.endpoint_pos_weight > 0.0 or config.ramp_pos_weight > 0.0
        ):
            raise ValueError("endpoint/ramp position losses are only implemented for relative action_pose_mode")
        self.config = config
        num_patches = (config.image_size // DINO_PATCH_SIZE) ** 2

        self.dino_encoder = DINOv2Encoder(config.hidden_dim, freeze=config.freeze_backbone)
        self.temporal_embed = nn.Embedding(config.obs_horizon, config.hidden_dim)
        self.task_embed = nn.Embedding(config.num_tasks, config.hidden_dim)
        self.proprio_proj = nn.Linear(config.obs_dim, config.hidden_dim)

        self.cvae_encoder = CVAEEncoder(config)
        self.z_proj = nn.Linear(config.latent_dim, config.hidden_dim)
        self.action_queries = nn.Embedding(config.action_horizon, config.hidden_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.hidden_dim, nhead=config.nhead,
            dim_feedforward=config.feedforward_dim, dropout=config.dropout,
            batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.num_decoder_layers)
        self.action_head = nn.Linear(config.hidden_dim, config.action_dim)

        self._num_patches = num_patches
        self._context_len = (
            1                                              # task token
            + config.obs_horizon                           # proprio tokens
            + config.obs_horizon * num_patches             # visual tokens (single camera)
        )

    def _encode_visual(self, images: torch.Tensor) -> torch.Tensor:
        """(B, T, 3, H, W) → (B, T*num_patches, hidden_dim). Temporal embedding added per step."""
        B, T = images.shape[:2]
        tokens_per_t = []
        for t in range(T):
            tokens = self.dino_encoder(images[:, t])            # (B, N, hidden_dim)
            tokens = tokens + self.temporal_embed.weight[t]
            tokens_per_t.append(tokens)
        return torch.cat(tokens_per_t, dim=1)

    def _encode_context(
        self,
        images: torch.Tensor,      # (B, T, 3, H, W)
        proprio: torch.Tensor,     # (B, T, obs_dim)
        task_id: int = 0,
    ) -> torch.Tensor:
        """Build full context: [task_token, proprio_0..T-1, visual_0..T-1]."""
        B = images.shape[0]
        task_token = self.task_embed.weight[task_id].view(1, 1, -1).expand(B, -1, -1)
        proprio_tokens = self.proprio_proj(proprio)
        visual_tokens = self._encode_visual(images)
        return torch.cat([task_token, proprio_tokens, visual_tokens], dim=1)

    @staticmethod
    def _reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    @staticmethod
    def _kl_divergence(
        mu: torch.Tensor, logvar: torch.Tensor, free_bits: float = 0.0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """KL(N(mu, var) || N(0, I)).

        Returns (kl_for_loss, kl_for_reporting). They differ only when
        free_bits > 0: the loss term floors per-dim mean KL at `free_bits`
        (gradients zero out below the floor), while the reported metric is
        the true unfloored KL.
        """
        # Per-sample, per-dim KL: 0.5 * (mu^2 + exp(logvar) - logvar - 1)
        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())  # (B, D)
        kl_report = kl_per_dim.sum(dim=-1).mean()
        if free_bits <= 0:
            return kl_report, kl_report
        kl_per_dim_mean = kl_per_dim.mean(dim=0)  # (D,)
        kl_loss = torch.clamp(kl_per_dim_mean, min=free_bits).sum()
        return kl_loss, kl_report

    def _decode_actions(self, z: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B = z.shape[0]
        z_token = self.z_proj(z).unsqueeze(1)                   # (B, 1, hidden_dim)
        queries = self.action_queries.weight.unsqueeze(0).expand(B, -1, -1)
        tgt = torch.cat([z_token, queries], dim=1)
        decoded = self.decoder(tgt, context)
        return self.action_head(decoded[:, 1:, :])              # drop z token

    def forward(
        self,
        images: torch.Tensor,      # (B, T, 3, H, W)
        proprio: torch.Tensor,     # (B, T, obs_dim)
        actions: torch.Tensor,     # (B, H, action_dim)
        is_pad: torch.Tensor | None = None,  # (B, H) bool — True = padded frame, exclude from loss
        task_id: int = 0,
    ) -> dict[str, torch.Tensor]:
        """Training step. Returns dict with `loss`, `l1_loss`, `kl_loss`.

        `is_pad` masks out frames that are padding (action horizon exceeded
        episode length) or SLAM-lost — the L1 loss ignores them, so the model
        doesn't waste capacity predicting fake zero-vectors.
        """
        context = self._encode_context(images, proprio, task_id=task_id)
        mu, logvar = self.cvae_encoder(actions, context)
        z = self._reparameterize(mu, logvar)
        pred = self._decode_actions(z, context)

        if is_pad is None:
            l1_per = (pred - actions).abs()                    # (B, H, D)
            l1 = l1_per.mean()
            keep = torch.ones_like(actions[..., :1])           # (B, H, 1) — all real
        else:
            # ACT-plus-plus style: per-element L1, zero out padded positions,
            # then mean over real elements only.
            l1_per = (pred - actions).abs()                    # (B, H, D)
            keep = (~is_pad).float().unsqueeze(-1)             # (B, H, 1)
            denom = keep.sum() * actions.shape[-1] + 1e-6
            l1 = (l1_per * keep).sum() / denom
        kl_for_loss, kl = self._kl_divergence(mu, logvar, self.config.free_bits)

        # Position-only auxiliary losses to fight magnitude undershoot.
        # Both default to 0 weight, so this is a no-op unless enabled.
        endpoint_l1 = pred.new_zeros(())
        ramp_l1 = pred.new_zeros(())
        if self.config.endpoint_pos_weight > 0.0 or self.config.ramp_pos_weight > 0.0:
            B, H, _ = pred.shape
            pos_l1 = (pred[..., :3] - actions[..., :3]).abs()  # (B, H, 3)
            keep_pos = keep                                     # (B, H, 1)

            if self.config.endpoint_pos_weight > 0.0:
                # Last non-padded frame index per batch item (pad is contiguous at end).
                if is_pad is None:
                    last_idx = torch.full((B,), H - 1, device=pred.device, dtype=torch.long)
                else:
                    last_idx = ((~is_pad).long().sum(dim=1) - 1).clamp(min=0)
                batch_idx = torch.arange(B, device=pred.device)
                endpoint_l1 = pos_l1[batch_idx, last_idx, :].mean()

            if self.config.ramp_pos_weight > 0.0:
                # Up-ramp: 0 at chunk start, 1 at chunk end. Empirical undershoot
                # lives late in the chunk; weight position L1 toward the tail.
                ramp = torch.linspace(0.0, 1.0, H, device=pred.device).view(1, H, 1)
                weighted = pos_l1 * ramp * keep_pos             # (B, H, 3)
                norm = (ramp * keep_pos).sum().clamp(min=1e-6) * 3.0
                ramp_l1 = weighted.sum() / norm

        loss = (
            l1
            + self.config.kl_weight * kl_for_loss
            + self.config.endpoint_pos_weight * endpoint_l1
            + self.config.ramp_pos_weight * ramp_l1
        )
        return {
            "loss": loss,
            "l1_loss": l1,
            "kl_loss": kl,
            "endpoint_pos_l1": endpoint_l1.detach(),
            "ramp_pos_l1": ramp_l1.detach(),
        }

    @torch.no_grad()
    def predict(
        self,
        images: torch.Tensor,      # (B, T, 3, H, W)
        proprio: torch.Tensor,     # (B, T, obs_dim)
        task_id: int = 0,
    ) -> torch.Tensor:
        """Inference: predict action chunk with z = 0 (prior mean). Returns (B, H, action_dim)."""
        context = self._encode_context(images, proprio, task_id=task_id)
        B = images.shape[0]
        z = torch.zeros(B, self.config.latent_dim, device=images.device)
        return self._decode_actions(z, context)


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return (trainable, total) parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total
