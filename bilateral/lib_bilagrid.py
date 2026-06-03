from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def color_affine_transform(affine_mats: torch.Tensor, rgb: torch.Tensor) -> torch.Tensor:
    """Apply per-pixel 3x4 affine color transforms."""
    return torch.matmul(affine_mats[..., :3], rgb.unsqueeze(-1)).squeeze(-1) + affine_mats[..., 3]


def total_variation_loss(x: torch.Tensor) -> torch.Tensor:
    """Compute average squared total variation over grid spatial dimensions."""
    total = x.new_zeros(())
    for dim in range(2, x.ndim):
        if x.shape[dim] <= 1:
            continue
        total = total + torch.diff(x, dim=dim).square().mean()
    return total


class BilateralGrid(nn.Module):
    """Per-image bilateral grids initialized as identity color transforms."""

    def __init__(self, num: int, grid_X: int = 16, grid_Y: int = 16, grid_W: int = 8):
        super().__init__()
        identity = torch.tensor(
            [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        )
        grid = identity.view(1, 12, 1, 1, 1).expand(1, 12, grid_W, grid_Y, grid_X).clone()
        self.grids = nn.Parameter(grid.expand(num, -1, -1, -1, -1).clone())
        self.register_buffer("rgb2gray_weight", torch.tensor([0.299, 0.587, 0.114]))

    def forward(
        self,
        xy: torch.Tensor,
        rgb: torch.Tensor,
        grid_idx: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if grid_idx is None:
            grids = self.grids.mean(dim=0, keepdim=True)
        else:
            grids = self.grids[grid_idx.reshape(-1)]

        xy = xy * 2.0 - 1.0
        guidance = torch.tensordot(rgb, self.rgb2gray_weight, dims=([-1], [0])).unsqueeze(-1)
        guidance = guidance * 2.0 - 1.0
        sample_xyz = torch.cat([xy, guidance], dim=-1).unsqueeze(1)
        affine = F.grid_sample(
            grids,
            sample_xyz,
            mode="bilinear",
            align_corners=True,
            padding_mode="border",
        )
        affine = affine.squeeze(2).permute(0, 2, 3, 1)
        return affine.reshape(*affine.shape[:-1], 3, 4)


def slice_grid(
    bil_grid: BilateralGrid,
    xy: torch.Tensor,
    rgb: torch.Tensor,
    grid_idx: Optional[torch.Tensor],
) -> torch.Tensor:
    return bil_grid(xy, rgb, grid_idx)
