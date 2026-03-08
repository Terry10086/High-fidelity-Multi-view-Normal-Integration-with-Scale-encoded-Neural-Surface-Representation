import itertools
import logging as log
from typing import Optional, Union, List, Dict, Sequence, Iterable, Collection, Callable

import torch
import torch.nn as nn
# import tinycudann as tcnn

from encoding.ops.interpolation import grid_sample_wrapper
from third_parties.ops import grid_sample

# from plenoxels.raymarching.spatial_distortions import SpatialDistortion

class KPlaneField(nn.Module):
    def __init__(
        self,
        aabb: torch.Tensor,
        grid_config: Union[str, List[Dict]],
        concat_features_across_scales: bool,
        multiscale_res: Optional[Sequence[int]],
        density_activation: Callable,
    ) -> None:
        super().__init__()

        if isinstance(aabb, list):
            aabb = torch.tensor(aabb)

        self.aabb = nn.Parameter(aabb, requires_grad=False)
        self.spatial_distortion = False
        self.grid_config = grid_config

        self.multiscale_res_multipliers: List[int] = multiscale_res or [1]
        self.concat_features = concat_features_across_scales
        self.density_activation = density_activation
        self.linear_decoder = False

        # 1. 初始化 planes
        self.grids = nn.ModuleList()
        self.feature_dim = 0
        for res in self.multiscale_res_multipliers: #[1, 2, 4, 8]  用于乘以base_res，生成高&多分辨率的plane
            # initialize coordinate grid
            config = self.grid_config[0].copy()
            # Resolution fix: multi-res only on spatial planes
            config["resolution"] = [r * res for r in config["resolution"][:3]] + config["resolution"][3:]   # [1*64, 2*64, 4*64, 8*64]  得到[64,64,64,25(t_res)], [128,128,128,25(t_res)]... 最后一维是t_res不需要多分辨率

            gp = init_grid_param(
                grid_nd=config["grid_dimensions"],
                in_dim=config["input_coordinate_dim"],
                out_dim=config["output_coordinate_dim"],
                reso=config["resolution"],
            )       # 初始化plane参数

            # shape[1] is out-dim - Concatenate over feature len for each scale
            if self.concat_features:        # 是否concat多分辨率的feature
                self.feature_dim += gp[-1].shape[1]
            else:
                self.feature_dim = gp[-1].shape[1]
            self.grids.append(gp)
        
        from colorama import Fore, Style
        print(Fore.MAGENTA)
        print('Resolution of scale:', self.grids[0][2].shape)
        print(Style.RESET_ALL)
        # log.info(f"Initialized model grids: {self.grids}")


    def forward(self, pts, radius):
        # camera_indices = None

        if radius is not None:
            # timestamps = timestamps[:, None].expand(-1, n_samples)[..., None]  # [n_rays, n_samples, 1]
            pts = torch.cat((pts, radius), dim=-1)  # [n_rays, n_samples, 4]

        features = interpolate_ms_features(
            pts, ms_grids=self.grids,  # noqa
            grid_dimensions=self.grid_config[0]["grid_dimensions"],
            concat_features=self.concat_features, num_levels=None)
        
        return features

    def get_params(self):
        field_params = {k: v for k, v in self.grids.named_parameters(prefix="grids")}
        nn_params = [
            self.sigma_net.named_parameters(prefix="sigma_net"),
            self.direction_encoder.named_parameters(prefix="direction_encoder"),
        ]
        if self.linear_decoder:
            nn_params.append(self.color_basis.named_parameters(prefix="color_basis"))
        else:
            nn_params.append(self.color_net.named_parameters(prefix="color_net"))
        nn_params = {k: v for plist in nn_params for k, v in plist}
        other_params = {k: v for k, v in self.named_parameters() if (
            k not in nn_params.keys() and k not in field_params.keys()
        )}
        return {
            "nn": list(nn_params.values()),
            "field": list(field_params.values()),
            "other": list(other_params.values()),
        }
    

def get_normalized_directions(directions):
    """SH encoding must be in the range [0, 1]

    Args:
        directions: batch of directions
    """
    return (directions + 1.0) / 2.0


def normalize_aabb(pts, aabb):
    return (pts - aabb[0]) * (2.0 / (aabb[1] - aabb[0])) - 1.0


def init_grid_param(
        grid_nd: int,
        in_dim: int,
        out_dim: int,
        reso: Sequence[int],
        a: float = 0.1,
        b: float = 0.5):


    coo_combs = [(0, 3), (1, 3), (2, 3)]

    grid_coefs = nn.ParameterList()
    for ci, coo_comb in enumerate(coo_combs):
        new_grid_coef = nn.Parameter(torch.empty([1, out_dim] + [reso[cc] for cc in coo_comb]))
        nn.init.ones_(new_grid_coef)
        grid_coefs.append(new_grid_coef)

    return grid_coefs


def interpolate_ms_features(pts,
                            ms_grids,
                            grid_dimensions,
                            concat_features,
                            num_levels,
                            ) -> torch.Tensor:
    
    coo_combs = list(itertools.combinations(range(3), grid_dimensions))   # [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    coo_combs = [(0, 3), (1, 3), (2, 3)]

    if num_levels is None:
        num_levels = len(ms_grids)      # 1
    multi_scale_interp = [] if concat_features else 0.

    single_scale_list = []
    mask_list = []
    grid: nn.ParameterList
    for scale_id, grid in enumerate(ms_grids[:num_levels]):
        interp_space = 1.
        for ci, coo_comb in enumerate(coo_combs):
            # interpolate in plane
            feature_dim = grid[ci].shape[1]  # shape of grid[ci]: 1, out_dim, *reso  [batch, channel, res of x/y/z/t, x/y/z/t]

            interp_out_plane2 = grid_sample.grid_sample_2d(grid[ci], pts[..., coo_comb])        # torch.Size([2097152, 32])
            interp_out_plane2 = interp_out_plane2.view(1, feature_dim, -1).transpose(-1, -2).squeeze()  # torch.Size([2097152, 32])
            interp_out_plane2 = (interp_out_plane2.view(-1, feature_dim))
            single_scale_list.append(interp_out_plane2)     # 32*6 = 72           

            # np.save('xy_4w_128',grid[0].cpu().numpy().transpose(2, 3, 1, 0).squeeze(-1).mean(axis=-1))

        
    if concat_features:
        multi_scale_interp = torch.cat(single_scale_list, dim=-1)
    return multi_scale_interp


def fix_grid(grid, pts):
    pts_new = (pts + 1) / 2 * grid.shape[-1]
    
    mask = torch.zeros_like(grid[0,0,:,:], dtype=torch.bool)
    x_indices = pts_new[:,0].round().long().clamp(0, grid.shape[2]-1)
    y_indices = pts_new[:,1].round().long().clamp(0, grid.shape[2]-1)
    mask[y_indices,x_indices] = True

    return mask
