import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import mcubes
from icecream import ic
from tqdm import tqdm

def compute_ball_radii(distance, radiis, cos):
    inverse_cos = 1.0 / cos
    tmp = (inverse_cos * inverse_cos - 1).sqrt() - radiis
    sample_ball_radii = distance * radiis * cos / (tmp * tmp + 1.0).sqrt()
    return sample_ball_radii

def extract_fields(bound_min, bound_max, radii_grid, resolution, query_func, encoding):
    N = 64
    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in tqdm(enumerate(X)):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs)
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)
                    # level =  torch.ones_like(pts[..., 0]).unsqueeze(-1) * 0.0001
                    pts2idx = map_to_grid_index(pts.cpu().numpy(),radii_grid.shape[0])
                    level = torch.from_numpy(radii_grid[pts2idx[:, 0], pts2idx[:, 1], pts2idx[:, 2]]).to(pts.device).unsqueeze(-1).float()   # pts.device
                    val = query_func(pts, encoding, level).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy()
                    u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
    return u

def map_to_grid_index(val, resolution):
    val = ((val + 1) / 2 * (resolution - 1)).round()
    val = np.clip(val.astype(int), 0, resolution - 1)
    return val


def extract_geometry(bound_min, bound_max, radii_grid, resolution, threshold, query_func, encoding):
    u = extract_fields(bound_min, bound_max, radii_grid, resolution, query_func, encoding)
    print('threshold: {}'.format(threshold), u.min(), u.max())
    vertices, triangles = mcubes.marching_cubes(u, threshold)
    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()

    vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    return vertices, triangles


def sample_pdf(bins, weights, n_samples, det=False):
    # This implementation is from NeRF
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    # Take uniform samples
    if det:
        u = torch.linspace(0. + 0.5 / n_samples, 1. - 0.5 / n_samples, steps=n_samples)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples])

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


class NeuSRenderer:
    def __init__(self, sdf_network,deviation_network,encoding,n_samples,
                 n_importance,
                 n_outside,
                 up_sample_steps,
                 perturb):
        
        self.sdf_network = sdf_network
        self.deviation_network = deviation_network
        self.encoding = encoding
        
        self.n_samples = n_samples
        self.n_importance = n_importance
        self.n_outside = n_outside
        self.up_sample_steps = up_sample_steps
        self.perturb = perturb


    def up_sample(self, rays_o, rays_d, z_vals, sdf, n_importance, inv_s):
        """
        Up sampling give a fixed inv_s
        """
        batch_size, n_samples = z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]  # n_rays, n_samples, 3
        radius = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=False)
        inside_sphere = (radius[:, :-1] < 1.0) | (radius[:, 1:] < 1.0)
        sdf = sdf.reshape(batch_size, n_samples)
        prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]
        prev_z_vals, next_z_vals = z_vals[:, :-1], z_vals[:, 1:]
        mid_sdf = (prev_sdf + next_sdf) * 0.5
        cos_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5)

        # ----------------------------------------------------------------------------------------------------------
        # Use min value of [ cos, prev_cos ]
        # Though it makes the sampling (not rendering) a little bit biased, this strategy can make the sampling more
        # robust when meeting situations like below:
        #
        # SDF
        # ^
        # |\          -----x----...
        # | \        /
        # |  x      x
        # |---\----/-------------> 0 level
        # |    \  /
        # |     \/
        # |
        # ----------------------------------------------------------------------------------------------------------
        prev_cos_val = torch.cat([torch.zeros([batch_size, 1]), cos_val[:, :-1]], dim=-1)
        cos_val = torch.stack([prev_cos_val, cos_val], dim=-1)
        cos_val, _ = torch.min(cos_val, dim=-1, keepdim=False)
        cos_val = cos_val.clip(-1e3, 0.0) * inside_sphere

        dist = (next_z_vals - prev_z_vals)
        prev_esti_sdf = mid_sdf - cos_val * dist * 0.5
        next_esti_sdf = mid_sdf + cos_val * dist * 0.5
        prev_cdf = torch.sigmoid(prev_esti_sdf * inv_s)
        next_cdf = torch.sigmoid(next_esti_sdf * inv_s)
        alpha = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)
        weights = alpha * torch.cumprod(
            torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]

        z_samples = sample_pdf(z_vals, weights, n_importance, det=True).detach()
        return z_samples

    def cat_z_vals(self, rays_o, rays_d, z_vals, new_z_vals, sdf, radiis_select, ray_cos_select, z_vals_importance_list, last=False):
        batch_size, n_samples = z_vals.shape
        _, n_importance = new_z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * new_z_vals[..., :, None]
        z_vals = torch.cat([z_vals, new_z_vals], dim=-1)
        z_vals_importance_list.append(new_z_vals)
        z_vals, index = torch.sort(z_vals, dim=-1)

        radiis = radiis_select.expand(new_z_vals.size()).reshape(-1, 1)  # (n,1)
        ray_cos = ray_cos_select.expand(new_z_vals.size()).reshape(-1, 1)    # (n,1)
        sample_ball_radii = compute_ball_radii(new_z_vals.reshape(-1, 1), radiis, ray_cos)

        if not last:
            new_sdf = self.sdf_network.sdf(pts.reshape(-1, 3), self.encoding, sample_ball_radii).reshape(batch_size, n_importance)
            sdf = torch.cat([sdf, new_sdf], dim=-1)
            xx = torch.arange(batch_size)[:, None].expand(batch_size, n_samples + n_importance).reshape(-1)
            index = index.reshape(-1)
            sdf = sdf[(xx, index)].reshape(batch_size, n_samples + n_importance)

        return z_vals, sdf, z_vals_importance_list

    def render_core(self,
                    rays_o,
                    rays_d,
                    z_vals,
                    sample_dist,
                    radiis_select,
                    ray_cos_select,
                    sdf_network,
                    deviation_network,
                    cos_anneal_ratio=0.0):
        batch_size, n_samples = z_vals.shape

        # Section length
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.Tensor([sample_dist]).expand(dists[..., :1].shape)], -1)
        mid_z_vals = z_vals + dists * 0.5

        # Section midpoints
        pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]  # n_rays, n_samples, 3
        dirs = rays_d[:, None, :].expand(pts.shape)

        pts = pts.reshape(-1, 3)
        dirs = dirs.reshape(-1, 3)

        radiis = radiis_select.expand(mid_z_vals.size()).reshape(-1, 1)  # (n,1)
        ray_cos = ray_cos_select.expand(mid_z_vals.size()).reshape(-1, 1)    # (n,1)
        sample_ball_radii = compute_ball_radii(mid_z_vals.reshape(-1, 1), radiis, ray_cos)

        radius_loss = None

        sdf, radius_loss = sdf_network(pts, self.encoding, sample_ball_radii)        # , feature, mask_list

        gradients = sdf_network.gradient(pts, self.encoding, sample_ball_radii).squeeze()
        

        inv_s = deviation_network(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)           # Single parameter
        inv_s = inv_s.expand(batch_size * n_samples, 1)

        true_cos = (dirs * gradients).sum(-1, keepdim=True)

        # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
        # the cos value "not dead" at the beginning training iterations, for better convergence.
        iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
                     F.relu(-true_cos) * cos_anneal_ratio)  # always non-positive

        # Estimate signed distances at section points
        estimated_next_sdf = sdf + iter_cos * dists.reshape(-1, 1) * 0.5
        estimated_prev_sdf = sdf - iter_cos * dists.reshape(-1, 1) * 0.5

        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

        p = prev_cdf - next_cdf
        c = prev_cdf

        alpha = ((p + 1e-5) / (c + 1e-5)).reshape(batch_size, n_samples).clip(0.0, 1.0)

        pts_norm = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=True).reshape(batch_size, n_samples)
        inside_sphere = (pts_norm < 1.0).float().detach()
        relax_inside_sphere = (pts_norm < 1.2).float().detach()

        weights = alpha * torch.cumprod(torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]
        
        comp_normals = (gradients.reshape(batch_size, n_samples, 3) * weights[:, :, None]).sum(dim=1)
        depth = (mid_z_vals.reshape(batch_size, n_samples, 1) * weights[:, :, None]).sum(dim=1)

        # Eikonal loss
        gradient_error = (torch.linalg.norm(gradients.reshape(batch_size, n_samples, 3), ord=2,
                                            dim=-1) - 1.0) ** 2
        gradient_error = (relax_inside_sphere * gradient_error).sum() / (relax_inside_sphere.sum() + 1e-5)

        return {
            'color': None,
            'sdf': sdf,
            'dists': dists,
            'gradients': gradients.reshape(batch_size, n_samples, 3),
            # 's_val': 1.0 / inv_s,
            'mid_z_vals': mid_z_vals,
            'weights': weights,
            'cdf': c.reshape(batch_size, n_samples),
            'gradient_error': gradient_error,
            'inside_sphere': inside_sphere,
            'comp_normal':comp_normals,
            'depth':depth,
            # 'feature': feature,
            'pts': pts,
            'sample_ball_radii':sample_ball_radii,
            'radius_loss': radius_loss,
            # 'mask_list': mask_list
        }

    def render(self, rays_o, rays_d, near, far, radiis_select, ray_cos_select, patch_size, img_idx, use_plane_fitting, cos_anneal_ratio=0.0):
        batch_size = len(rays_o)
        sample_dist = 2.0 / self.n_samples   # Assuming the region of interest is a unit sphere
        z_vals = torch.linspace(0.0, 1.0, self.n_samples)
        z_vals = near + (far - near) * z_vals[None, :]

        n_samples = self.n_samples      # 64
        perturb = self.perturb      # 1

        if perturb > 0:
            t_rand = (torch.rand([batch_size, 1]) - 0.5)        # -0.4999~0.5
            z_vals = z_vals + t_rand * 2.0 / self.n_samples

        # Up sample
        if self.n_importance > 0:
            with torch.no_grad():
                pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]
                radiis = radiis_select.expand(z_vals.size()).reshape(-1, 1)  # (n,1)
                ray_cos = ray_cos_select.expand(z_vals.size()).reshape(-1, 1)    # (n,1)
                sample_ball_radii = compute_ball_radii(z_vals.reshape(-1, 1), radiis, ray_cos)
                
                sdf = self.sdf_network.sdf(pts.reshape(-1, 3), self.encoding, sample_ball_radii).reshape(batch_size, self.n_samples)
                z_vals_importance_list = []

                for i in range(self.up_sample_steps):
                    new_z_vals = self.up_sample(rays_o,
                                                rays_d,
                                                z_vals,
                                                sdf,
                                                self.n_importance // self.up_sample_steps,
                                                64 * 2**i)
                    z_vals, sdf, z_vals_importance_list  = self.cat_z_vals(rays_o,
                                                  rays_d,
                                                  z_vals,
                                                  new_z_vals,
                                                  sdf,
                                                  radiis_select,
                                                  ray_cos_select,
                                                  z_vals_importance_list,
                                                  last=(i + 1 == self.up_sample_steps),)

            z_vals_importance = torch.cat(z_vals_importance_list,axis=-1)   # z_vals_importance_list[-1]        # torch.cat(z_vals_importance_list,axis=-1)
            # z_vals = z_vals_importance
            z_vals_importance, _= torch.sort(z_vals_importance, dim=-1)

        # Render core
        ret_fine = self.render_core(rays_o,
                                    rays_d,
                                    z_vals,
                                    sample_dist,
                                    radiis_select,
                                    ray_cos_select,
                                    self.sdf_network,
                                    self.deviation_network,                                   
                                    cos_anneal_ratio=cos_anneal_ratio)

        color_fine = ret_fine['color']
        weights = ret_fine['weights']
        weights_sum = weights.sum(dim=-1, keepdim=True)
        gradients = ret_fine['gradients']
        # s_val = ret_fine['s_val'].reshape(batch_size, n_samples).mean(dim=-1, keepdim=True)

        if img_idx%2==1 and use_plane_fitting:       
            normal_avg = self.get_normal(rays_o,
                                        rays_d,
                                        z_vals_importance,
                                        sample_dist,
                                        patch_size,
                                        radiis_select,
                                        ray_cos_select,
                                        self.sdf_network,
                                        self.deviation_network,                                   
                                        cos_anneal_ratio=cos_anneal_ratio)
        else:
            normal_avg = None

        return {
            'color_fine': color_fine,
            # 's_val': s_val,
            'cdf_fine': ret_fine['cdf'],
            'weight_sum': weights_sum,
            'weight_max': torch.max(weights, dim=-1, keepdim=True)[0],
            'gradients': gradients,
            'weights': weights,
            'gradient_error': ret_fine['gradient_error'],
            'inside_sphere': ret_fine['inside_sphere'],
            'comp_normal':ret_fine['comp_normal'],
            'depth':ret_fine['depth'],
            # 'feature':ret_fine['feature'],
            'pts':ret_fine['pts'],
            'sample_ball_radii': ret_fine['sample_ball_radii'],
            'radius_loss': ret_fine['radius_loss'],
            'normal_avg': normal_avg
            # 'mask_list': ret_fine['mask_list']
        }
    
    def get_normal(self,
                   rays_o,
                   rays_d,
                   z_vals,
                   sample_dist,
                   patch_size,
                   radiis_select,
                   ray_cos_select,
                   sdf_network,
                   deviation_network,                  
                   cos_anneal_ratio=0.0):
        
        batch_size, n_samples = z_vals.shape

        # Section length
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.Tensor([sample_dist]).expand(dists[..., :1].shape)], -1)
        mid_z_vals = z_vals + dists * 0.5

        pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]  # n_rays, n_samples, 3
        dirs = rays_d[:, None, :].expand(pts.shape)

        pts = pts.reshape(-1, 3)
        dirs = dirs.reshape(-1, 3)

        radiis = radiis_select.expand(mid_z_vals.size()).reshape(-1, 1)  # (n,1)
        ray_cos = ray_cos_select.expand(mid_z_vals.size()).reshape(-1, 1)    # (n,1)
        sample_ball_radii = compute_ball_radii(mid_z_vals.reshape(-1, 1), radiis * patch_size, ray_cos)

        # sample_ball_radii = sample_ball_radii * patch_size
        # print(sample_ball_radii.max())
        sdf, _ = sdf_network(pts, self.encoding, sample_ball_radii)        # , feature, mask_list

        gradients = sdf_network.gradient(pts, self.encoding, sample_ball_radii).squeeze()
        # sampled_color = color_network(pts, gradients, dirs, feature_vector).reshape(batch_size, n_samples, 3)

        inv_s = deviation_network(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)           # Single parameter
        inv_s = inv_s.expand(batch_size * n_samples, 1)

        true_cos = (dirs * gradients).sum(-1, keepdim=True)

        # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
        # the cos value "not dead" at the beginning training iterations, for better convergence.
        iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
                     F.relu(-true_cos) * cos_anneal_ratio)  # always non-positive

        # Estimate signed distances at section points
        estimated_next_sdf = sdf + iter_cos * dists.reshape(-1, 1) * 0.5
        estimated_prev_sdf = sdf - iter_cos * dists.reshape(-1, 1) * 0.5

        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

        p = prev_cdf - next_cdf
        c = prev_cdf

        alpha = ((p + 1e-5) / (c + 1e-5)).reshape(batch_size, n_samples).clip(0.0, 1.0)
        weights = alpha * torch.cumprod(torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]
        
        normal_avg = (gradients.reshape(batch_size, n_samples, 3) * weights[:, :, None]).sum(dim=1)
        
        return normal_avg


    def extract_geometry(self, bound_min, bound_max, radii_grid, resolution, threshold=0.0):
        return extract_geometry(bound_min,
                                bound_max,
                                radii_grid,
                                resolution=resolution,
                                threshold=threshold,
                                query_func=lambda pts, encoding, level: -self.sdf_network.sdf(pts, self.encoding, level, sampling=True),
                                encoding = self.encoding)
