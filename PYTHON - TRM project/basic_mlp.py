import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPBaseline(nn.Module):
    """
    MLP baseline roughly matched to TRM-no-rec:

    - input:  (B, D)
    - output: (B, 1)

    Uses:
      * hidden size D (same as TRM hidden_size)
      * expansion factor (like TRM's MLP expansion)
      * depth ~ 2 * num_layers  (to mimic 2 f_θ calls)
    """

    def __init__(self, hidden_size: int, num_layers: int,
                 expansion: float = 2.0, out_dim: int = 1):
        super().__init__()
        D = hidden_size
        exp_D = int(expansion * D)

        blocks = []
        # 2 * num_layers "blocks", each: Linear(D -> exp_D) -> GELU -> Linear(exp_D -> D) -> GELU
        for _ in range(2 * num_layers):
            blocks.extend([
                nn.Linear(D, exp_D),
                nn.GELU(),
                nn.Linear(exp_D, D),
                nn.GELU(),
            ])

        self.feature = nn.Sequential(*blocks)
        self.reg_head = nn.Linear(D, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, D)
        returns: (B, 1)
        """
        h = self.feature(x)
        y_pred = self.reg_head(h)
        return y_pred
