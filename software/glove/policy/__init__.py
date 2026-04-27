"""Policy architectures for manipulation.

`ACTPolicy` (Zhao et al. 2023, with a frozen DINOv2 backbone) is the production
checkpoint used for the headline results. `DiffusionPolicy` (Chi et al. 2023)
is kept as the architecture comparison baseline reported in the paper.
"""

from glove.policy.diffusion import DiffusionPolicy, PolicyConfig
from glove.policy.act import ACTPolicy, ACTConfig

__all__ = ["DiffusionPolicy", "PolicyConfig", "ACTPolicy", "ACTConfig"]
