import torch
from torch.nn import functional as F




def grid_sample_wrapper(grid: torch.Tensor, coords: torch.Tensor, align_corners: bool = True) -> torch.Tensor:
    grid_dim = coords.shape[-1]     # 需要grid_dim=2 即：pts=[n,2]

    if coords.dim() == 2:
        coords = coords.unsqueeze(0)

    if grid_dim == 2 or grid_dim == 3:
        grid_sampler = F.grid_sample

    coords = coords.view([coords.shape[0]] + [1] * (grid_dim - 1) + list(coords.shape[1:]))
    B, feature_dim = grid.shape[:2]
    n = coords.shape[-2]
    interp = grid_sampler(
        grid,  # [B, feature_dim, reso, ...]
        coords,  # [B, 1, ..., n, grid_dim]
        align_corners=align_corners,
        mode='bilinear', padding_mode='border')
    interp = interp.view(B, feature_dim, n).transpose(-1, -2)  # [B, n, feature_dim]        torch.Size([1, 2097152, 32])
    interp = interp.squeeze()  # [B?, n, feature_dim?]      torch.Size([2097152, 32])
    return interp
