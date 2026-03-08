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

def deal_map(encoding):
    maps = []
    for i in range(3):
        map = encoding.grids[0][i].permute(2, 3, 1, 0).squeeze(-1).mean(axis=-1)
        map = F.interpolate(map.unsqueeze(0).unsqueeze(0), scale_factor=2, mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
        maps.append(map)

    return maps[0],maps[1],maps[2]


def extract_fields(radius, radius_min, radius_max, bound_min, bound_max, radii_grid, resolution, query_func, encoding):
    N = 64
    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)

    mapx,mapy,mapz = deal_map(encoding)

    radius_pc = []

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    color = np.zeros([resolution, resolution, resolution], dtype=np.float32)

    with torch.no_grad():
        for xi, xs in tqdm(enumerate(X)):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs)
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)

                    min_val = -1
                    max_val = 1
                    num_parts = mapx.shape[1]
                    interval_width = (max_val - min_val) / num_parts
                    indices = torch.round((pts - min_val) / interval_width).clamp(0, num_parts - 1).long()
                    
                    
                    result_sum = mapx[:, indices[:, 0]] * mapy[:, indices[:, 1]] * mapz[:, indices[:, 2]]
                    
                    max_values, max_indices = torch.max(result_sum.sum(axis=-1),dim=0)
                    # top_values, top_indices = torch.topk(result_sum.sum(axis=-1), k=5, dim=0)
                                                                                  
                    # if top_indices[0]>32:
                    #     max_indices = top_indices[0]
                    # else:
                    #     close_count = (top_indices - top_indices[0]).abs() <= 5
                    #     majority_count = close_count.sum().item()
                    #     if majority_count>=4:
                    #         max_indices = top_indices[0]
                    #     else:
                    #         max_indices = max(top_indices)

                    radii = torch.linspace(radius_min, radius_max, result_sum.shape[0])
                    corresponding_radii = radii[max_indices]
                    
                    level = (torch.ones_like(pts[..., 0]) * corresponding_radii).unsqueeze(-1)       # 0.0016
                    val = query_func(pts, encoding, level).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy()
                    u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
                    color[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = level.reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy()

                    ###################################################
                    has_positive = (val > 0).any()
                    has_negative = (val < 0).any()

                    if has_positive and has_negative:
                        avg = (pts[0]+pts[-1])/2
                        
                    ###################################################
        

    
    #np.save('./exp/pc.npy',torch.stack(radius_pc).detach().cpu().numpy())
    return u,color

def map_to_grid_index(val, resolution):
    val = ((val + 1) / 2 * (resolution - 1)).round()
    val = np.clip(val.astype(int), 0, resolution - 1)
    return val


def _iter_tiles(D, H, W, chunk, overlap):
    # 带重叠遍历，保证最后一个块覆盖到末尾
    for z0 in range(0, D, chunk - overlap):
        z1 = min(z0 + chunk, D)
        if z1 - z0 <= overlap and z1 != D:  # 末块太小则与前块合并
            continue
        for y0 in range(0, H, chunk - overlap):
            y1 = min(y0 + chunk, H)
            if y1 - y0 <= overlap and y1 != H:
                continue
            for x0 in range(0, W, chunk - overlap):
                x1 = min(x0 + chunk, W)
                if x1 - x0 <= overlap and x1 != W:
                    continue
                yield z0, z1, y0, y1, x0, x1

# def marching_cubes_chunked_from_grid(u, threshold, chunk_size=512, overlap=1, dedup_tol=1e-6):
#     """对已有 dense 体素 u 进行分块 Marching Cubes（带边界重叠与顶点去重）"""
#     assert u.ndim == 3
#     D, H, W = u.shape
# 
#     # 顶点哈希表：量化坐标 -> 全局索引
#     quant_inv = 1.0 / dedup_tol
#     def quantize(v):
#         return tuple(np.round(v * quant_inv).astype(np.int64))
# 
#     vertex_map = {}
#     vertices_all = []
#     faces_all = []
#     vcount = 0
# 
#     tiles = list(_iter_tiles(D, H, W, chunk_size, overlap))
#     pbar = tqdm(total=len(tiles), desc="Marching Cubes (chunked)", unit="block")
# 
#     for z0, z1, y0, y1, x0, x1 in tiles:
#         sub = u[z0:z1, y0:y1, x0:x1]
# 
#         # 跳过无交界块
#         smin, smax = sub.min(), sub.max()
#         if smin > threshold or smax < threshold:
#             pbar.update(1)
#             continue
# 
#         # 跑本块 MC
#         v, f = mcubes.marching_cubes(sub, threshold)
# 
#         # 将本块坐标偏移回全局体素坐标系
#         v[:, 0] += z0
#         v[:, 1] += y0
#         v[:, 2] += x0
# 
#         # 顶点去重：对每个顶点进行量化，复用已存在索引
#         idx_map = np.empty(len(v), dtype=np.int64)
#         for i in range(len(v)):
#             key = quantize(v[i])
#             if key in vertex_map:
#                 idx_map[i] = vertex_map[key]
#             else:
#                 vertex_map[key] = vcount
#                 idx_map[i] = vcount
#                 vertices_all.append(v[i])
#                 vcount += 1
# 
#         # 重映射本块面索引到全局
#         faces_all.append(idx_map[f])
# 
#         pbar.update(1)
# 
#     pbar.close()
# 
#     if len(vertices_all) == 0:
#         return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int32)
# 
#     V = np.vstack(vertices_all).astype(np.float32)
#     F = np.vstack(faces_all).astype(np.int32)
#     return V, F

def marching_cubes_chunked_safe(u, threshold=0.0, chunk_size=128, overlap=1, dedup_tol=1e-6):
    D, H, W = u.shape
    vertices_all, faces_all = [], []
    vcount = 0

    # 遍历所有 chunk
    for z0 in range(0, D, chunk_size-overlap):
        z1 = min(z0 + chunk_size, D)
        for y0 in range(0, H, chunk_size-overlap):
            y1 = min(y0 + chunk_size, H)
            for x0 in range(0, W, chunk_size-overlap):
                x1 = min(x0 + chunk_size, W)
                sub = u[z0:z1, y0:y1, x0:x1]
                
                smin, smax = sub.min(), sub.max()
                if smin > threshold or smax < threshold:
                    continue

                # 每块 marching cubes
                v, f = mcubes.marching_cubes(sub, threshold)

                # 坐标偏移到全局
                v[:, 0] += z0
                v[:, 1] += y0
                v[:, 2] += x0

                # 拼接
                faces_all.append(f + vcount)
                vertices_all.append(v)
                vcount += len(v)

    if len(vertices_all) == 0:
        return np.zeros((0,3), dtype=np.float32), np.zeros((0,3), dtype=np.int32)

    V = np.vstack(vertices_all).astype(np.float32)
    F = np.vstack(faces_all).astype(np.int32)

    # 全局去重 (numpy vectorized)
    quant = np.round(V / dedup_tol).astype(np.int64)
    _, idx, inv = np.unique(quant, axis=0, return_index=True, return_inverse=True)
    V = V[idx]
    F = inv[F]

    return V, F

def extract_geometry(radius, radius_min, radius_max, bound_min, bound_max, radii_grid, resolution, threshold, query_func, encoding):
    u,color = extract_fields(radius, radius_min, radius_max, bound_min, bound_max, radii_grid, resolution, query_func, encoding)
    print('threshold: {}'.format(threshold), u.min(), u.max())
    vertices, triangles = marching_cubes_chunked_safe(
        u, threshold=threshold, chunk_size=512, overlap=1, dedup_tol=1e-6
    )
    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()

    from matplotlib import cm
    import open3d as o3d
    color_normalized = (color - color.min()) / (color.max()-color.min())
    vertex_colors = color_normalized[vertices[:, 0].astype(int),
                                 vertices[:, 1].astype(int),
                                 vertices[:, 2].astype(int)]

    # Apply a color map (e.g., 'viridis') to the normalized colors
    cmap = cm.get_cmap('viridis')
    vertex_colors_mapped = cmap(vertex_colors)[:, :3]  # Ignore alpha channel

    # Step 3: Create Open3D TriangleMesh with vertices and colors
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors_mapped)

    # Save or visualize
    o3d.io.write_triangle_mesh("./exp/colored_mesh_exp2.ply", mesh)
    print('succeed!')

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
    def __init__(self, radius_min,radius_max, sdf_network,deviation_network, encoding, f2w, n_samples,
                 n_importance,
                 n_outside,
                 up_sample_steps,
                 set_radius,
                 perturb):
        self.radius_min = radius_min
        self.radius_max = radius_max
        
        self.sdf_network = sdf_network
        self.deviation_network = deviation_network
        self.encoding = encoding
        self.f2w = f2w
        
        self.n_samples = n_samples
        self.n_importance = n_importance
        self.n_outside = n_outside
        self.up_sample_steps = up_sample_steps
        self.perturb = perturb
        self.set_radius = set_radius


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

    def cat_z_vals(self, rays_o, rays_d, z_vals, new_z_vals, sdf, radiis_select, ray_cos_select, z_vals_importance_list, last=False, is_validate=False):
        batch_size, n_samples = z_vals.shape
        _, n_importance = new_z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * new_z_vals[..., :, None]
        z_vals = torch.cat([z_vals, new_z_vals], dim=-1)
        z_vals_importance_list.append(new_z_vals)
        z_vals, index = torch.sort(z_vals, dim=-1)

        radiis = radiis_select.expand(new_z_vals.size()).reshape(-1, 1)  # (n,1)
        ray_cos = ray_cos_select.expand(new_z_vals.size()).reshape(-1, 1)    # (n,1)
        sample_ball_radii = compute_ball_radii(new_z_vals.reshape(-1, 1), radiis, ray_cos)
        if is_validate==True and (sample_ball_radii.min()+sample_ball_radii.max())/2 < self.radius_min:
                sample_ball_radii = torch.ones_like(sample_ball_radii)*self.set_radius

        if not last:
            new_sdf = self.sdf_network.sdf(pts.reshape(-1, 3), self.encoding, self.f2w, sample_ball_radii).reshape(batch_size, n_importance)
            sdf = torch.cat([sdf, new_sdf], dim=-1)
            xx = torch.arange(batch_size)[:, None].expand(batch_size, n_samples + n_importance).reshape(-1)
            index = index.reshape(-1)
            sdf = sdf[(xx, index)].reshape(batch_size, n_samples + n_importance)

        return z_vals, sdf, z_vals_importance_list

    def render_core(self,
                    rays_o,
                    rays_d,
                    z_vals,
                    mask,scales,
                    sample_dist,
                    radiis_select,
                    ray_cos_select,
                    sdf_network,
                    deviation_network,
                    is_validate = False,
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

        mask = mask.expand(mid_z_vals.size()).reshape(-1, 1)
        radiis = radiis_select.expand(mid_z_vals.size()).reshape(-1, 1)  # (n,1)
        ray_cos = ray_cos_select.expand(mid_z_vals.size()).reshape(-1, 1)    # (n,1)
        sample_ball_radii = compute_ball_radii(mid_z_vals.reshape(-1, 1), radiis, ray_cos)

        if is_validate ==True and (sample_ball_radii.min()+sample_ball_radii.max())/2 < self.radius_min:
            sample_ball_radii = torch.ones_like(sample_ball_radii)*self.set_radius

        sdf, radius_loss = sdf_network(pts, self.encoding, self.f2w, sample_ball_radii, set_radius=self.set_radius, cal_sdf_loss = True)        # , feature, mask_list

        gradients = sdf_network.gradient(pts, self.encoding, self.f2w, sample_ball_radii).squeeze()

        ## --------------------------------------- plane loss ---------------------------------------------

        # # for plane fitting loss
        # factors = scales.transpose(0,1)     # from 1.1 to 1.9
        # radiis_scaled = radiis * factors 
        # radius_scaled = compute_ball_radii(mid_z_vals.reshape(-1, 1), radiis_scaled, ray_cos)

        # gradients_scale = []
        # for radius_num in range(factors.shape[-1]):
        #     gradients_scale_mask = torch.zeros_like(gradients)
        #     gi = sdf_network.gradient(pts[mask.bool().squeeze()], self.encoding, self.f2w, radius_scaled[:, [radius_num]][mask.bool().squeeze()]).squeeze()
        #     gradients_scale_mask[mask.bool().squeeze()] = gi
        #     gradients_scale.append(gradients_scale_mask)
        # gradients_scale = torch.stack(gradients_scale, dim=0).reshape(factors.shape[-1],batch_size, n_samples, 3)

        ## --------------------------------------- plane loss ---------------------------------------------

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

        # ones_weights = torch.ones_like(weights)
        
        comp_normals = (gradients.reshape(batch_size, n_samples, 3) * weights[:, :, None]).sum(dim=1)
        depth = (mid_z_vals.reshape(batch_size, n_samples, 1) * weights[:, :, None]).sum(dim=1)
        radius = (sample_ball_radii.reshape(batch_size, n_samples, 1) * weights[:, :, None]).sum(dim=1)
        # comp_normals_scale = (gradients_scale * weights[:, :, None].unsqueeze(0)).sum(dim=2)
        comp_normals_scale = None

        

        # Eikonal loss
        gradient_error = (torch.linalg.norm(gradients.reshape(batch_size, n_samples, 3), ord=2,
                                            dim=-1) - 1.0) ** 2
        gradient_error = (relax_inside_sphere * gradient_error).sum() / (relax_inside_sphere.sum() + 1e-5)

        return {
            'sdf': sdf,
            'dists': dists,
            'gradients': gradients.reshape(batch_size, n_samples, 3),
            'mid_z_vals': mid_z_vals,
            'weights': weights,
            'cdf': c.reshape(batch_size, n_samples),
            'gradient_error': gradient_error,
            'inside_sphere': inside_sphere,
            'comp_normal':comp_normals,
            'depth':depth,
            'pts': pts,
            'sample_ball_radii':sample_ball_radii,
            'radius_loss': radius_loss,
            'radius':radius,       
            'mesh_loss': None,
            'comp_normals_scale': comp_normals_scale    
        }

    def render(self, rays_o, rays_d, near, far, radiis_select, ray_cos_select, patch_size, mask, scales, use_plane_fitting, is_validate = False, cos_anneal_ratio=0.0):
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

                if is_validate==True and (sample_ball_radii.min()+sample_ball_radii.max())/2 < self.radius_min:
                    sample_ball_radii = torch.ones_like(sample_ball_radii)*self.set_radius
                
                sdf = self.sdf_network.sdf(pts.reshape(-1, 3), self.encoding, self.f2w, sample_ball_radii).reshape(batch_size, self.n_samples)
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
                                                  last=(i + 1 == self.up_sample_steps), is_validate=is_validate)

            # z_vals_importance = z_vals_importance_list[-1]   # z_vals_importance_list[-1]        # torch.cat(z_vals_importance_list,axis=-1)
            # z_vals_importance, _= torch.sort(z_vals_importance, dim=-1)

        # Render core
        ret_fine = self.render_core(rays_o,
                                    rays_d,
                                    z_vals,
                                    mask,
                                    scales,
                                    sample_dist,
                                    radiis_select,
                                    ray_cos_select,
                                    self.sdf_network,
                                    self.deviation_network,    
                                    is_validate = is_validate,                               
                                    cos_anneal_ratio=cos_anneal_ratio)

        
        weights = ret_fine['weights']
        weights_sum = weights.sum(dim=-1, keepdim=True)
        gradients = ret_fine['gradients']
        


        # dists = z_vals_importance[..., 1:] - z_vals_importance[..., :-1]
        # dists = torch.cat([dists, torch.Tensor([sample_dist]).expand(dists[..., :1].shape)], -1)
        # mid_z_vals = z_vals_importance + dists * 0.5
        # pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]

        # sample_ball_radii = compute_ball_radii(mid_z_vals, radiis_select, ray_cos_select)
        


        return {
            'cdf_fine': ret_fine['cdf'],
            'weight_sum': weights_sum,
            'weight_max': torch.max(weights, dim=-1, keepdim=True)[0],
            'gradients': gradients,
            'weights': weights,
            'gradient_error': ret_fine['gradient_error'],
            'inside_sphere': ret_fine['inside_sphere'],
            'comp_normal':ret_fine['comp_normal'],
            'depth':ret_fine['depth'],
            'pts':pts,
            'sample_ball_radii': None,
            'radius_loss': ret_fine['radius_loss'],
            'radius':ret_fine['radius'],
            'normal_avg': None,
            'mesh_loss': ret_fine['mesh_loss'],
            'sdf': ret_fine['sdf'],
            'comp_normals_scale': ret_fine['comp_normals_scale']  
        }

    def extract_geometry(self, radius, bound_min, bound_max, radii_grid, resolution, threshold=0.0):
        return extract_geometry(radius,
                                self.radius_min,
                                self.radius_max,
                                bound_min,
                                bound_max,
                                self.set_radius,           # radii_grid
                                resolution=resolution,
                                threshold=threshold,
                                query_func=lambda pts, encoding, level: -self.sdf_network.sdf(pts, self.encoding, self.f2w, level),
                                encoding = self.encoding)
