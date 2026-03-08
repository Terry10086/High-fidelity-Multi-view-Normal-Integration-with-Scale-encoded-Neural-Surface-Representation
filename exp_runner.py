import os
import time
import logging
import argparse
import numpy as np
import cv2 as cv
import trimesh
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from shutil import copyfile
from icecream import ic
from tqdm import tqdm
from pyhocon import ConfigFactory
from models.dataset import Dataset
from models.fields import SDFNetwork, SingleVarianceNetwork, WeightMLP
from models.renderer import NeuSRenderer
from models.cd_and_fscore import chamfer_distance_and_f1_score

from encoding.kplanes import KPlaneField
from colorama import Fore, Style
import matplotlib.pyplot as plt
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)

import open3d as o3d
from glob import glob
import pyexr
from utilities.utils import crop_image_by_mask, toRGBA
import logging
logging.basicConfig(level=logging.WARNING)

class Runner:
    def __init__(self, conf_path, mode='train', case='CASE_NAME', is_continue=False):
        self.device = torch.device('cuda')
        # Configuration
        self.conf_path = conf_path
        f = open(self.conf_path)
        conf_text = f.read()
        conf_text = conf_text.replace('CASE_NAME', case)
        f.close()
        self.conf = ConfigFactory.parse_string(conf_text)

        exp_time_dir = self.conf['dataset.exp_time_dir']    # exp_2024_04_20_22_59_10
                
        self.conf['dataset.data_dir'] = self.conf['dataset.data_dir'].replace('CASE_NAME', case)
        print(Fore.MAGENTA)
        print('save file name:',exp_time_dir)
        print('running dataset:',self.conf['dataset.data_dir'])
        print(Style.RESET_ALL)
        
        self.base_exp_dir = os.path.join(self.conf['general.base_exp_dir'], exp_time_dir)
        os.makedirs(self.base_exp_dir, exist_ok=True)
        self.dataset = Dataset(self.conf['dataset'])
        self.iter_step = 0

        # Training parameters
        self.end_iter = self.conf.get_int('train.end_iter')
        self.save_freq = self.conf.get_int('train.save_freq')
        self.report_freq = self.conf.get_int('train.report_freq')
        self.val_freq = self.conf.get_int('val.val_freq')
        self.val_mesh_freq = self.conf.get_int('val.val_mesh_freq')
        self.val_mesh_res = self.conf.get_int('val.val_mesh_res')
        self.batch_size = self.conf.get_int('train.batch_size')
        self.validate_resolution_level = self.conf.get_int('train.validate_resolution_level')
        self.learning_rate = self.conf.get_float('train.learning_rate')
        self.learning_rate_alpha = self.conf.get_float('train.learning_rate_alpha')
        self.use_white_bkgd = self.conf.get_bool('train.use_white_bkgd')
        self.warm_up_end = self.conf.get_float('train.warm_up_end', default=0.0)
        self.anneal_end = self.conf.get_float('train.anneal_end', default=0.0)
        self.patch_size = self.conf.get_int('train.patch_size', default=3)

        self.increase_bindwidth_every = self.conf.get_int('train.increase_bindwidth_every', default=350)

        # Validation parameters
        
        self.val_normal_resolution_level = self.conf.get_int('val.val_normal_resolution_level')
        self.val_gradient_method = self.conf.get('val.gradient_method', 'dfd')

        self.val_mesh_freq = self.conf.get_int('val.val_mesh_freq')
        self.val_mesh_res = self.conf.get_int('val.val_mesh_res')

        self.eval_metric_freq = self.conf.get_int('val.eval_metric_freq')
        self.report_freq = self.conf.get_int('val.report_freq')
        self.save_freq = self.conf.get_int('val.save_freq')

        # Weights
        self.igr_weight = self.conf.get_float('train.igr_weight')
        self.mask_weight = self.conf.get_float('train.mask_weight')
        self.normal_weight = self.conf.get_float('train.normal_weight')
        self.sdf_weight = self.conf.get_float('train.sdf_weight')
        
        self.is_continue = is_continue
        self.mode = mode
        self.model_list = []
        self.writer = None
        self.radius_min = self.conf.get_float('model.sdf_network.radius_min', default=0.1)
        self.radius_max = self.conf.get_float('model.sdf_network.radius_max', default=0.1)
        self.radius_max = self.conf.get_float('model.sdf_network.radius_max', default=0.1)
        self.radii_grid = np.full((128, 128, 128), 0.00111)    # self.radius_max

        # Networks   
        self.sdf_network = SDFNetwork(self.dataset.real_data, **self.conf['model.sdf_network'], encoding_config=self.conf['model.encoding']).to(self.device)
        self.deviation_network = SingleVarianceNetwork(**self.conf['model.variance_network']).to(self.device)
        self.encoding = KPlaneField(**self.conf['model.kplanes']).to(self.device)
        self.f2w = None # WeightMLP(input_dim = self.conf['model.kplanes.grid_config'][0]['output_coordinate_dim'], output_dim = self.conf['model.encoding.n_levels']* self.conf['model.encoding.n_features_per_level'], hidden_dim = self.conf['model.f2w.hidden_dim']).to(self.device)
        
        params_to_train = []
        params_to_train.append(dict(params=self.sdf_network.parameters(), lr=self.learning_rate))
        params_to_train.append(dict(params=self.deviation_network.parameters(), lr=self.learning_rate))
        params_to_train.append(dict(params=self.encoding.parameters(), lr=0.01, weight_decay=1e-5))        # 0.002
        # params_to_train.append(dict(params=self.f2w.parameters(), lr=self.learning_rate))
        self.optimizer = torch.optim.Adam(params_to_train, lr=self.learning_rate)

        self.renderer = NeuSRenderer(
             self.radius_min,self.radius_max,
                                     self.sdf_network,
                                     self.deviation_network,
                                     self.encoding, self.f2w, **self.conf['model.neus_renderer']
                                     )

        # Load checkpoint
        latest_model_name = None
        if is_continue:
            model_list_raw = os.listdir(os.path.join(self.base_exp_dir, 'checkpoints'))
            model_list = []
            for model_name in model_list_raw:
                if model_name[-3:] == 'pth' and int(model_name[5:-4]) <= self.end_iter:
                    model_list.append(model_name)
            model_list.sort()
            latest_model_name = model_list[-1]

        if latest_model_name is not None:
            logging.info('Find checkpoint: {}'.format(latest_model_name))
            self.load_checkpoint(latest_model_name)

        # Backup codes and configs for debug
        if self.mode[:5] == 'train':
            self.file_backup()

    def train(self):
        self.writer = SummaryWriter(log_dir=os.path.join(self.base_exp_dir, 'logs'))
        self.update_learning_rate()
        res_step = self.end_iter - self.iter_step
        loss_list = []
        iter_list = []
        pbar = tqdm(range(res_step))


        for iter_i in pbar:
            data, idx, pixel_idx = self.dataset.gen_random_rays_at(self.batch_size)

            rays_o, rays_d, true_normal, mask, radii, ray_cos = data[:, :3], data[:, 3: 6], data[:, 6: 9], data[:, 9: 10], data[:, 10: 11], data[:, 11:12]   
            near, far = self.dataset.near_far_from_sphere(rays_o, rays_d)
            near = torch.clamp(near, min=0.0001)

            if self.mask_weight > 0.0:
                mask = (mask > 0.5).float()
            else:
                mask = torch.ones_like(mask)
            mask_sum = mask.sum() + 1e-5
            
            self.use_plane_fitting = False
            patch_size = None
            scales = None

            render_out = self.renderer.render(rays_o, rays_d, near, far, radii, ray_cos, patch_size, mask, scales, use_plane_fitting = self.use_plane_fitting, 
                                            cos_anneal_ratio=self.get_cos_anneal_ratio())
            
            if self.iter_step % self.increase_bindwidth_every == 0:
                self.renderer.sdf_network.increase_bandwidth()
            
            comp_normal = render_out['comp_normal']
            gradient_error = render_out['gradient_error']
            weight_sum = render_out['weight_sum']
            radius_error = render_out['radius_loss']
            sdf = render_out['sdf']
            mesh_loss = render_out['mesh_loss']
            comp_normals_scale = render_out['comp_normals_scale']



            normal_error = (comp_normal - true_normal) * mask
            # normal_loss = F.mse_loss(normal_error, torch.zeros_like(normal_error), reduction='sum') / mask_sum
            normal_loss = F.l1_loss(normal_error, torch.zeros_like(normal_error), reduction='sum') / mask_sum

            eikonal_loss = gradient_error
            mask_loss = F.binary_cross_entropy(weight_sum.clip(1e-3, 1.0 - 1e-3), mask)
            sdf_loss = radius_error.std(dim=1).sum()/radius_error.shape[0]

            # force1 = self.regularize_l1_time_planes(self.encoding)
            loss = self.normal_weight * normal_loss + mask_loss * self.mask_weight + eikonal_loss * self.igr_weight + self.sdf_weight * sdf_loss # + 0.0001 * force1   # + normal_scale_loss + time_smooth 
                                  
            self.optimizer.zero_grad()      # 把所有梯度置为0
            loss.backward()
            self.optimizer.step()

            self.iter_step += 1

            if self.iter_step % self.report_freq == 0:
                from collections import OrderedDict
                message_postfix = OrderedDict(loss=f"{loss:.3e}")
                pbar.set_postfix(ordered_dict=message_postfix)

                if self.iter_step>5000:
                    loss_list.append(loss.item())
                    iter_list.append(self.iter_step)
                if self.iter_step%10000==0:
                    print('iter:{:8>d} loss = {} lr={}'.format(self.iter_step, loss, self.optimizer.param_groups[0]['lr']))
            
            if self.iter_step % 10000 == 0:
                
                os.makedirs(os.path.join(self.base_exp_dir, 'fig'), exist_ok=True)
                plt.figure()
                plt.xlabel('Iteration')
                plt.ylabel('loss')
                plt.plot(iter_list,loss_list)
                path = os.path.join(self.base_exp_dir,'fig','{:0>8d}_{}.png'.format(self.iter_step, 'loss_value'))
                plt.savefig(path)

            if self.iter_step % self.save_freq == 0:
                self.save_checkpoint()
                print(Fore.RED)
                print('saved in',self.base_exp_dir)
                print(Style.RESET_ALL)
                

            if self.iter_step % self.val_freq == 0: # and self.iter_step>20000:
                self.validate_image()

            if self.iter_step % self.val_mesh_freq == 0:
                self.validate_mesh(0,resolution=self.val_mesh_res)
            
            if self.iter_step % self.eval_metric_freq == 0 :               # and self.iter_step>20000
                # if self.dataset.type == 'super':
                #     if self.dataset.exclude_view_list == []:
                #         mae_allview = self.eval_mae_sn(gradient_method="dfd")
                #         print('MAE (all views): {:.4f}'.format(mae_allview))
                #     else:
                #         mae_allview, mae_test_view = self.eval_mae_sn(gradient_method="dfd")
                #         print('MAE (all views): {:.4f}'.format(mae_allview))
                #         print('MAE (test views) {:.4f}'.format(mae_test_view))
                # else:

                # MAE
                # if self.dataset.exclude_view_list == []:
                #     mae_allview = self.eval_mae()
                #     print('MAE (all views): {:.4f}'.format(mae_allview))
                # else:
                #     mae_allview, mae_test_view = self.eval_mae()
                #     print('MAE (all views): {:.4f}'.format(mae_allview))
                #     print('MAE (test views) {:.4f}'.format(mae_test_view))

                if self.dataset.mesh_gt is not None:
                    save_dir = os.path.join(self.base_exp_dir, 'points_val')
                    gt_points_path = os.path.join(save_dir, "pcd_gt.ply")           

                    if os.path.exists(gt_points_path):   # 
                        # Load the existing ground truth points from the file
                        pcd_gt = o3d.io.read_point_cloud(gt_points_path)
                        self.dataset.points_gt = np.asarray(pcd_gt.points)
                    else:                        
                        self.dataset.mesh_gt.vertices = o3d.utility.Vector3dVector(
                        (np.asarray(self.dataset.mesh_gt.vertices) -
                         self.dataset.scale_mats_np[:3, 3][None]) /
                        self.dataset.scale_mats_np[0, 0])
                        mesh = trimesh.Trimesh(np.asarray(self.dataset.mesh_gt.vertices),
                                            np.asarray(self.dataset.mesh_gt.triangles), process=False)
                        self.dataset.points_gt = self.find_visible_points(mesh) * self.dataset.scale_mats_np[0, 0] + self.dataset.scale_mats_np[:3, 3][None]
                        save_dir = os.path.join(self.base_exp_dir, 'points_val')
                        os.makedirs(save_dir, exist_ok=True)
                        # save gt points
                        pcd_gt = o3d.geometry.PointCloud()
                        pcd_gt.points = o3d.utility.Vector3dVector(self.dataset.points_gt)
                        o3d.io.write_point_cloud(os.path.join(save_dir, f"pcd_gt.ply"), pcd_gt)     # 把mesh保存为点云文件


                    # ---------------- 导入mesh ------------------------------
                    trimip = o3d.io.read_triangle_mesh(r'/media/yangtongyu/T9/code3/Tri-MipRF-main/exp/mic/close/trimip_512.ply')
                    trimip_mesh = trimesh.Trimesh(np.asarray(trimip.vertices), np.asarray(trimip.triangles), process=False)
                    self.dataset.points_eval = self.find_visible_points(trimip_mesh)
                    pcd_eval = o3d.geometry.PointCloud()
                    pcd_eval.points = o3d.utility.Vector3dVector(self.dataset.points_eval)
                    o3d.io.write_point_cloud(os.path.join(save_dir, f"pcd_eval.ply"), pcd_eval)     # 把mesh保存为点云文件
                    
                    # ---------------- 导入mesh ------------------------------
                    

                cd, fscore = self.eval_geo(resolution=512)
                print(f'iter: {self.iter_step} cd: {cd:.3e}, fscore: {fscore:.3e}')

            self.update_learning_rate()
    
    

    def calculate_normal_avg(self, pixel_idx, img_idx, patch_size):
        # 对pixel_idx做筛选，保证不在边缘
        p = patch_size
        height, width = self.dataset.H, self.dataset.W
        x_indices, y_indices = pixel_idx
        half_patch = p // 2
        mask = ((x_indices >= half_patch) & (x_indices < height - half_patch) & (y_indices >= half_patch) & (y_indices < width - half_patch)).cuda()

        x_indices_all = x_indices[:, None, None] + torch.arange(-p // 2+1, p // 2+1, device=self.device).repeat(p, 1).cuda()   # (num_patch, patch_H, patch_W)  torch.Size([4096, 7, 7])
        y_indices_all = y_indices[:, None, None] + torch.arange(-p // 2+1, p // 2+1, device=self.device).reshape(-1, 1).repeat(1, p).cuda()   # (num_patch, patch_H, patch_W)
        
        # 填充图像
        padded_images = F.pad(self.dataset.images[img_idx], (0, 0, half_patch, half_patch, half_patch, half_patch), mode='constant', value=0).cuda()     # torch.Size([806, 806, 9])
        
        # 调整索引
        x_indices_all_clamped = x_indices_all + half_patch
        y_indices_all_clamped = y_indices_all + half_patch
        comp_normal = padded_images[(y_indices_all_clamped, x_indices_all_clamped)]     # torch.Size([4096, 7, 7, 3])

        # # 方式1：求平均值
        # comp_normal_reshape = comp_normal.reshape(comp_normal.shape[0],-1,3)        # torch.Size([2048, 9, 3])
        # # eigenvectors_select = torch.mean(comp_normal_reshape,dim=1)

        # # 方式2：特征值
        # comp_normal_unit = F.normalize(comp_normal_reshape, p=2, dim=-1)            # 归一化后, 当使用完全一样的9个向量求均值, 下面的方法获得的结果会完全一样
        # normals_fitting = torch.matmul(comp_normal_unit.transpose(1, 2), comp_normal_unit)
        # eigenvalues, eigenvectors = torch.linalg.eigh(normals_fitting)      # eigenvalues由小到大排序好了
        # eigenvectors_select = eigenvectors[:,:,2].detach()           # 选特征值最大的

        # 方式3：双线性插值
        eigenvectors_select = F.interpolate(comp_normal.permute(0, 3, 1, 2), size=(1, 1), mode='bilinear', align_corners=False).squeeze()


        return eigenvectors_select, mask
    
    def regularize(self, model):
        from typing import Sequence
        multi_res_grids: Sequence[torch.nn.ParameterList]
        multi_res_grids = model.grids
        
        total = 0
        # model.grids is 6 x [1, rank * F_dim, reso, reso]
        for grids in multi_res_grids:
            if len(grids) == 3:
                time_grids = []
            else:
                time_grids = [2, 4, 5]
            for grid_id in time_grids:
                total += self.compute_plane_smoothness(grids[grid_id])
        return torch.as_tensor(total)

    def compute_plane_smoothness(self,t):
        batch_size, c, h, w = t.shape
        # Convolve with a second derivative filter, in the time dimension which is dimension 2
        first_difference = t[..., 1:, :] - t[..., :h-1, :]  # [batch, c, h-1, w]
        second_difference = first_difference[..., 1:, :] - first_difference[..., :h-2, :]  # [batch, c, h-2, w]
        # Take the L2 norm of the result
        return torch.square(second_difference).mean()
    
    def regularize_l1_time_planes(self, model):
        from typing import Sequence
        multi_res_grids: Sequence[torch.nn.ParameterList]
        multi_res_grids = model.grids
        
        total = 0.0
        for grids in multi_res_grids:
            time_grids = [0, 1, 2]
            for grid_id in time_grids:
                total += torch.abs(1 - grids[grid_id]).mean()
        return torch.as_tensor(total)

    def get_image_perm(self):
        return torch.randperm(self.dataset.n_images)

    def get_cos_anneal_ratio(self):
        if self.anneal_end == 0.0:
            return 1.0
        else:
            return np.min([1.0, self.iter_step / self.anneal_end])

    def update_learning_rate(self):
        if self.iter_step < self.warm_up_end:
            learning_factor = self.iter_step / self.warm_up_end
        else:
            alpha = self.learning_rate_alpha
            progress = (self.iter_step - self.warm_up_end) / (self.end_iter - self.warm_up_end)
            learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha

        for g in self.optimizer.param_groups:
            g['lr'] = self.learning_rate * learning_factor

    def file_backup(self):
        dir_lis = self.conf['general.recording']
        os.makedirs(os.path.join(self.base_exp_dir, 'recording'), exist_ok=True)
        for dir_name in dir_lis:
            cur_dir = os.path.join(self.base_exp_dir, 'recording', dir_name)
            os.makedirs(cur_dir, exist_ok=True)
            files = os.listdir(dir_name)
            for f_name in files:
                if f_name[-3:] == '.py':
                    copyfile(os.path.join(dir_name, f_name), os.path.join(cur_dir, f_name))

        copyfile(self.conf_path, os.path.join(self.base_exp_dir, 'recording', 'config.conf'))

    def load_checkpoint(self, checkpoint_name):
        checkpoint = torch.load(os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name), map_location=self.device)
        
        self.sdf_network.load_state_dict(checkpoint['sdf_network_fine'])
        self.deviation_network.load_state_dict(checkpoint['variance_network_fine'])
        self.encoding.load_state_dict(checkpoint['encoding'])
        # self.f2w.load_state_dict(checkpoint['f2w'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.iter_step = checkpoint['iter_step']

        logging.info('End')

    def save_checkpoint(self):
        checkpoint = {
            'sdf_network_fine': self.sdf_network.state_dict(),
            'variance_network_fine': self.deviation_network.state_dict(),
            'encoding':self.encoding.state_dict(),
            # 'f2w':self.f2w.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'iter_step': self.iter_step,
        }

        os.makedirs(os.path.join(self.base_exp_dir, 'checkpoints'), exist_ok=True)
        torch.save(checkpoint, os.path.join(self.base_exp_dir, 'checkpoints', 'ckpt_{:0>6d}.pth'.format(self.iter_step)))

    def validate_image(self, idx=-1, resolution_level=-1):
        close_view = [1,2,5,6,7,10,12,15,16,17,18,21,22,24,36,48]


        os.makedirs(os.path.join(self.base_exp_dir, 'normals'), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir, 'depth'), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir, 'radius'), exist_ok=True)

        for i in range(0,self.dataset.n_images,1):      # self.dataset.n_images
            idx = i
            # if i in close_view:
            #     continue
            print('Validate: iter: {}, camera: {}'.format(self.iter_step, idx))

            if resolution_level < 0:
                resolution_level = self.validate_resolution_level
            rays_o, rays_d, radiis_select, ray_cos_select = self.dataset.gen_rays_at(idx, resolution_level=resolution_level)
            H, W, _ = rays_o.shape
            rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
            rays_d = rays_d.reshape(-1, 3).split(self.batch_size)
            # mask = self.dataset.masks[idx].reshape(-1, self.dataset.masks[idx].shape[-1]).split(self.batch_size)
            if self.dataset.real_data==True:
                mask = self.dataset.masks[idx]
            else:
                mask = self.dataset.masks[idx][:,:,0]
            mask = mask.reshape(-1, 1).split(self.batch_size)

            radiis_select = radiis_select.reshape(-1, 1).split(self.batch_size)
            ray_cos_select = ray_cos_select.reshape(-1, 1).split(self.batch_size)

            comp_normal_list = []
            depth_list = []
            radii_distb_list = []

            for rays_o_batch, rays_d_batch, radiis_select_batch, ray_cos_select_batch, mask_batch in zip(rays_o, rays_d,radiis_select,ray_cos_select, mask):
                near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)

                
                render_out = self.renderer.render(rays_o_batch,
                                                rays_d_batch,                                              
                                                near,
                                                far,
                                                radiis_select_batch,
                                                ray_cos_select_batch,
                                                3,
                                                mask_batch,
                                                scales = torch.arange(1.01, 1.02, 0.01).unsqueeze(-1),
                                                use_plane_fitting = False,
                                                is_validate = True,
                                                cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                                )
                
                comp_normal = render_out['comp_normal']
                depth = render_out['depth']
                radius_distribution = render_out['radius']
                
                comp_normal_list.append(comp_normal.detach().cpu().numpy())
                depth_list.append(depth.detach().cpu().numpy())
                radii_distb_list.append(radius_distribution.detach().cpu().numpy())
                
                del render_out

            comp_normal_list = np.concatenate(comp_normal_list, axis=0)
            depth_list = np.concatenate(depth_list, axis=0)
            radii_distb_list = np.concatenate(radii_distb_list,axis=0)

            np.save(os.path.join(self.base_exp_dir,'radius','{:0>8d}_{}.npy'.format(self.iter_step, self.dataset.idx[idx])),radii_distb_list.reshape([H,W, 1]))
            np.save(os.path.join(self.base_exp_dir,'normals','{:0>8d}_{}.npy'.format(self.iter_step, self.dataset.idx[idx])),comp_normal_list.reshape([H,W, 3]))
            np.save(os.path.join(self.base_exp_dir,'depth','{:0>8d}_{}.npy'.format(self.iter_step, self.dataset.idx[idx])),depth_list.reshape([H,W, 1]))

            

    def pc2volume(self, depth, radius, idx):
        depth = depth.squeeze()
        radius = radius.squeeze()
        mask = self.dataset.masks[idx].cpu().numpy().astype(bool)
        if mask.shape[-1]==3:
            mask = mask[:,:,0]

        _, bin_edges, _ = plt.hist(depth[mask], bins=10, edgecolor='black')       
        depth[mask & (depth < bin_edges[-2])] = 0
        shrink_pixels = 2

        
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (2 * shrink_pixels + 1, 1))
        shrinked_mask = cv.erode(mask.astype(np.uint8), kernel, iterations=1)

        shrinked_mask = shrinked_mask.astype(np.bool_)
        depth[~shrinked_mask] = 0
        K = self.dataset.intrinsics_all[0].cpu().numpy()
        point_cloud, radius = self.depth_to_point_cloud(depth, K, radius)
        point_cloud_homogeneous = np.hstack((point_cloud, np.ones((point_cloud.shape[0], 1))))
        c2w = self.dataset.pose_all[idx].cpu().numpy()
        transformed_points_homogeneous = (c2w @ point_cloud_homogeneous.T).T
        transformed_points = transformed_points_homogeneous[:, :3]      # 把点云转到世界坐标系下


        return transformed_points, radius
    
    def depth_to_point_cloud(self, depth, K, radius):

        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        height, width = depth.shape
        # 创建像素网格
        u, v = np.meshgrid(np.arange(width), np.arange(height))

        # 计算 3D 点
        X = (u - cx) * depth / fx
        Y = (v - cy) * depth / fy
        Z = depth


        # 将 3D 点组合成点云
        point_cloud = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)

        valid_points = point_cloud[:, 2] > 0  # 保留 Z 大于 0 的点

        radii = radius.flatten()
        radii = radii[valid_points]
        point_cloud = point_cloud[valid_points]


        return point_cloud.reshape(-1, 3),radii.reshape(-1, 1)  # 返回 (N, 3) 形状的点云


    def map_to_grid_index(self, val, resolution):
        val = ((val + 1) / 2 * (resolution - 1)).round()
        val = np.clip(val.astype(int), 0, resolution - 1)
        return val

    def render_novel_image(self, idx_0, idx_1, ratio, resolution_level):
        """
        Interpolate view between two cameras.
        """
        rays_o, rays_d = self.dataset.gen_rays_between(idx_0, idx_1, ratio, resolution_level=resolution_level)
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)

        out_rgb_fine = []
        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None

            render_out = self.renderer.render(rays_o_batch,
                                              rays_d_batch,
                                              near,
                                              far,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                              background_rgb=background_rgb)

            out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())

            del render_out

        img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3]) * 256).clip(0, 255).astype(np.uint8)
        return img_fine

    def render_normal(self, idx):
        rays_o, rays_d, radiis_select, ray_cos_select = self.dataset.gen_rays_at(idx, resolution_level=1)
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)
        # mask = self.dataset.masks[idx].reshape(-1, self.dataset.masks[idx].shape[-1]).split(self.batch_size)
        if self.dataset.real_data==True:
            mask = self.dataset.masks[idx]
        else:
            mask = self.dataset.masks[idx][:,:,0]
        mask = mask.reshape(-1, 1).split(self.batch_size)

        radiis_select = radiis_select.reshape(-1, 1).split(self.batch_size)
        ray_cos_select = ray_cos_select.reshape(-1, 1).split(self.batch_size)

        comp_normal_list = []
        depth_list = []
        radii_distb_list = []

        for rays_o_batch, rays_d_batch, radiis_select_batch, ray_cos_select_batch, mask_batch in zip(rays_o, rays_d,radiis_select,ray_cos_select, mask):
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            
            render_out = self.renderer.render(rays_o_batch,
                                            rays_d_batch,                                              
                                            near,
                                            far,
                                            radiis_select_batch,
                                            ray_cos_select_batch,
                                            3,
                                            mask_batch,
                                            scales = torch.arange(1.01, 1.02, 0.01).unsqueeze(-1),
                                            use_plane_fitting = False,
                                            is_validate = True,
                                            cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                            )
            
            comp_normal = render_out['comp_normal']
            depth = render_out['depth']
            radius_distribution = render_out['radius']
            
            comp_normal_list.append(comp_normal.detach().cpu().numpy())
            depth_list.append(depth.detach().cpu().numpy())
            radii_distb_list.append(radius_distribution.detach().cpu().numpy())
            
            del render_out

        comp_normal_list = np.concatenate(comp_normal_list, axis=0)
        return comp_normal_list, self.base_exp_dir

        
    def validate_mesh(self, radius, world_space=False, resolution=1024, threshold=0.0):
        print('mesh resolution:',resolution)
        bound_min = torch.tensor(self.dataset.object_bbox_min, dtype=torch.float32)
        bound_max = torch.tensor(self.dataset.object_bbox_max, dtype=torch.float32)

        vertices, triangles =\
            self.renderer.extract_geometry(radius, bound_min, bound_max, self.radii_grid, resolution=resolution, threshold=threshold)
        os.makedirs(os.path.join(self.base_exp_dir, 'meshes'), exist_ok=True)

        if True:    # world_space
            vertices = vertices * self.dataset.scale_mats_np[0, 0] + self.dataset.scale_mats_np[:3, 3][None]

        mesh = trimesh.Trimesh(vertices, triangles)
        mesh.export(os.path.join(self.base_exp_dir, 'meshes', 'ours_exp2_same_{:0>8d}.ply'.format(self.iter_step)))

        logging.info('End')

    def interpolate_view(self, img_idx_0, img_idx_1):
        images = []
        n_frames = 60
        for i in range(n_frames):
            print(i)
            images.append(self.render_novel_image(img_idx_0,
                                                  img_idx_1,
                                                  np.sin(((i / n_frames) - 0.5) * np.pi) * 0.5 + 0.5,
                          resolution_level=4))
        for i in range(n_frames):
            images.append(images[n_frames - i - 1])

        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        video_dir = os.path.join(self.base_exp_dir, 'render')
        os.makedirs(video_dir, exist_ok=True)
        h, w, _ = images[0].shape
        writer = cv.VideoWriter(os.path.join(video_dir,
                                             '{:0>8d}_{}_{}.mp4'.format(self.iter_step, img_idx_0, img_idx_1)),
                                fourcc, 30, (w, h))

        for image in images:
            writer.write(image)

        writer.release()
    
    
    @torch.no_grad()
    def eval_geo(self, resolution):
#         
# 
#         # marching cubes
#         bound_min = torch.tensor(self.dataset.object_bbox_min, dtype=torch.float32)
#         bound_max = torch.tensor(self.dataset.object_bbox_max, dtype=torch.float32)
# 
#         vertices, triangles = \
#             self.renderer.extract_geometry(0, bound_min, bound_max, self.radii_grid, resolution=resolution, threshold=0)
#         mesh = trimesh.Trimesh(vertices, triangles)
# 
#         output_dir = os.path.join(self.base_exp_dir, 'meshes')
#         os.makedirs(output_dir, exist_ok=True)  # 如果不存在就创建
#         mesh.export(os.path.join(self.base_exp_dir, 'meshes', 'ours_exp.ply'.format(self.iter_step)))
#                 
#         # vertices = np.load(os.path.join(self.base_exp_dir,'meshes', 'vertices.npy'))
#         # triangles = np.load(os.path.join(self.base_exp_dir,'meshes', 'triangles.npy'))
# 
#         save_dir = os.path.join(self.base_exp_dir, 'points_val')
#         os.makedirs(save_dir, exist_ok=True)
# 
#         mesh = trimesh.Trimesh(np.asarray(vertices), np.asarray(triangles), process=False)
#         vertices_world = vertices * self.dataset.scale_mats_np[0, 0] + self.dataset.scale_mats_np[:3, 3][None]
#         mesh_world = trimesh.Trimesh(np.asarray(vertices_world), np.asarray(triangles), process=False)
#         mesh_world_path = os.path.join(save_dir, f"{self.iter_step}_world.obj")
#         mesh_world.export(mesh_world_path)
# 
#         points_eval = self.find_visible_points(mesh)*self.dataset.scale_mats_np[0, 0] + self.dataset.scale_mats_np[:3, 3][None]
# 
#         # save the sampled points
#         sampled_points_path = os.path.join(save_dir, f"{self.iter_step}_points_eval.ply")
#         pcd_eval = o3d.geometry.PointCloud()
#         pcd_eval.points = o3d.utility.Vector3dVector(points_eval)        # points_inside
#         o3d.io.write_point_cloud(sampled_points_path, pcd_eval)
# 
#         cd, fscore = chamfer_distance_and_f1_score(points_eval, self.dataset.points_gt)
        cd, fscore = chamfer_distance_and_f1_score(self.dataset.points_eval, self.dataset.points_gt)
        return cd, fscore
    
    def find_visible_points(self, mesh):
        num_view = self.dataset.n_images
        points_list = []
        for view_idx in range(num_view):
            rays_o, rays_v,_,_ = self.dataset.gen_rays_at(view_idx, resolution_level=1)

            mask_np = self.dataset.masks[view_idx].cpu().numpy().astype(bool)
            if mask_np.shape[-1] != 3:
                mask_np = cv.resize(mask_np.astype(np.uint8)*255, (int(mask_np.shape[1] // 1), int(mask_np.shape[0] // 1)), interpolation=cv.INTER_NEAREST).astype(bool)
            else:
                mask_np = cv.resize(mask_np.astype(np.uint8)*255, (int(mask_np.shape[1] // 1), int(mask_np.shape[0] // 1)), interpolation=cv.INTER_NEAREST).astype(bool)[:,:,1]     # 必须加
            # mask_np[:] = True

            rays_o = rays_o[mask_np]
            rays_v = rays_v[mask_np]

            rays_o, rays_v = rays_o.cpu().detach().numpy(), rays_v.cpu().detach().numpy()
            rays_v = rays_v / np.linalg.norm(rays_v, axis=-1, keepdims=True)
            locations, index_ray, index_tri = mesh.ray.intersects_location(
                ray_origins=rays_o,
                ray_directions=rays_v,
                multiple_hits=False)
            points_list.append(locations)
        return np.concatenate(points_list, axis=0)
    
    def eval_mae(self):
        print("Computing mean angular errors...")
        
        normal_dir = os.path.join(self.dataset.data_dir, "normal_world_space_sdmunips")        
        ae_map_list = []
        normal_map_eval_list = []
        ae_map_eval_list = []
        ae_map_test_list = []
        
        normal_lis = sorted(glob(os.path.join(normal_dir, '*.exr'))) 
        normal_eval_list = os.path.join(self.base_exp_dir,'normals')   
        save_dir = os.path.join(self.base_exp_dir,'error_map')   
        os.makedirs(save_dir, exist_ok=True)


        for idx in range(0,self.dataset.n_images,1):
            normal_gt = pyexr.read(normal_lis[idx])[..., :3]
            normal_gt = normal_gt / (1e-10 + np.linalg.norm(normal_gt, axis=-1, keepdims=True))
            if self.dataset.real_data==True:
                mask_np = self.dataset.masks[idx].cpu().numpy().astype(bool)
            else:
                mask_np = self.dataset.masks[idx].cpu().numpy().astype(bool)[:,:,0]
            
            if self.is_continue==True and self.eval_metric_freq == 1:
                index = self.iter_step
            else:
                index = self.iter_step

            file_path_exist = os.path.exists(os.path.join(normal_eval_list, '{:0>8d}_{}.npy'.format(index, self.dataset.idx[idx])))
            if not file_path_exist:
                break

            normal_map_world = np.load(os.path.join(normal_eval_list,'{:0>8d}_{}.npy'.format(index, self.dataset.idx[idx]))  )

            normal_map_world = normal_map_world / (1e-10 + np.linalg.norm(normal_map_world, axis=-1, keepdims=True))

            normal_eval = np.zeros((self.dataset.masks[idx].shape[0], self.dataset.masks[idx].shape[1], 3))
            normal_eval[:normal_map_world.shape[0], :normal_map_world.shape[1]] = normal_map_world
            normal_eval[~mask_np] = np.nan
            normal_map_eval_list.append(normal_eval)
            
            angular_error_map = np.rad2deg(np.arccos(np.clip(np.sum(normal_gt * normal_eval, axis=-1), -1, 1)))
            # save angular error map

            ae_map_list.append(angular_error_map.copy())
            if idx in self.dataset.exclude_view_list:
                ae_map_test_list.append(angular_error_map.copy())

            # apply jet to angular error map
            angular_error_map[~mask_np] = 0
            angular_error_map_jet = cv.applyColorMap((angular_error_map / 20 * 255).clip(0, 255).astype(np.uint8),
                                                     cv.COLORMAP_JET)
            angular_error_map_jet[~mask_np] = 255
            angular_error_map_jet = crop_image_by_mask(toRGBA(angular_error_map_jet, mask_np), mask_np)
            
            cv.imwrite(os.path.join(save_dir, '{:0>8d}_{}_{}_ae_up_{}.png'.format(index, 0, idx, 20)), angular_error_map_jet)

            print('iter:',index,'idx:',idx,'MAE:',np.nanmean(np.stack(ae_map_list[-1], axis=0)))


            ae_map_eval_list.append(angular_error_map_jet)

        ae_map_list = [np.nanmean(arr) for arr in ae_map_list]
        ae_map_test_list = [np.nanmean(arr) for arr in ae_map_test_list]
        
        mae = np.nanmean(np.stack(ae_map_list, axis=0))
        self.writer.add_scalar('Statistics/mae_allview', mae, self.iter_step)

        if len(ae_map_test_list) > 0:
            mae_test = np.nanmean(np.stack(ae_map_test_list, axis=0))
            self.writer.add_scalar('Statistics/mae_testview', mae_test, self.iter_step)
            return mae, mae_test

        return mae

    @torch.no_grad()
    def eval_mae_sn(self, gradient_method):
        print("Computing mean angular errors...")
        normal_gt_dir = os.path.join(self.dataset.data_dir, "normal_world_space_GT")

        ae_map_list = []
        normal_map_eval_list = []
        ae_map_eval_list = []
        ae_map_test_list = []
        for idx in range(self.dataset.n_images):
            normal_gt = pyexr.read(os.path.join(normal_gt_dir, "{:02d}.exr".format(idx)))[..., :3]

            mask_np = self.dataset.masks_np[idx].astype(bool)

            normal_map_world, save_dir = self.render_normal(idx)

            normal_map_world = normal_map_world / (1e-10 + np.linalg.norm(normal_map_world, axis=-1, keepdims=True))

            normal_eval = np.zeros((self.dataset.H, self.dataset.W, 3))
            normal_eval = normal_map_world.reshape(self.dataset.H, self.dataset.W, 3)
            normal_eval[~mask_np] = np.nan
            normal_map_eval_list.append(normal_eval)
            # self.writer.add_image(step=self.iter_step, data=(normal_eval + 1) / 2, name=("normal_eval_{:02d}".format(idx)))
            # pyexr.write(os.path.join(normal_save_dir, "{:02d}.exr".format(idx)), normal_img)

            angular_error_map = np.rad2deg(np.arccos(np.clip(np.sum(normal_gt * normal_eval, axis=-1), -1, 1)))
            # save angular error map

            ae_map_list.append(angular_error_map.copy())
            if idx in self.dataset.exclude_view_list:
                ae_map_test_list.append(angular_error_map.copy())

            # apply jet to angular error map
            angular_error_map[~mask_np] = 0
            angular_error_map_jet = cv.applyColorMap((angular_error_map / 20 * 255).clip(0, 255).astype(np.uint8),
                                                     cv.COLORMAP_JET)
            angular_error_map_jet[~mask_np] = 255
            angular_error_map_jet = crop_image_by_mask(toRGBA(angular_error_map_jet, mask_np), mask_np)
            cv.imwrite(os.path.join(save_dir, '{:0>8d}_{}_{}_ae_up_{}.png'.format(self.iter_step, 0, idx, 20)), angular_error_map_jet)


            ae_map_eval_list.append(angular_error_map_jet)

        mae = np.nanmean(np.stack(ae_map_list, axis=0))
        self.writer.add_scalar('Statistics/mae_allview', mae, self.iter_step)

        if len(ae_map_test_list) > 0:
            mae_test = np.nanmean(np.stack(ae_map_test_list, axis=0))
            self.writer.add_scalar('Statistics/mae_testview', mae_test, self.iter_step)
            return mae, mae_test

        return mae

    def validate_normal_patch_based(self, idx=-1, resolution_level=-1, gradient_method="dfd"):
        if idx < 0:
            idx = np.random.randint(self.dataset.n_images)

        print('Rendering normal maps...  iter: {}, camera: {}'.format(self.iter_step, idx))

        if resolution_level < 0:
            resolution_level = self.validate_resolution_level
        rays_o_patch_center, \
            rays_d_patch_center, \
            rays_o_patches_all, \
            rays_v_patches_all, \
            rays_ez, \
            rays_A_inverse, horizontal_num_patch, vertical_num_patch = self.dataset.gen_patches_at(idx, resolution_level=resolution_level,
                                                                                                   patch_H=self.patch_size,
                                                                                                   patch_W=self.patch_size)
        mask_np = self.dataset.masks_np[idx].astype(bool)  # (H, W)

        img_w = horizontal_num_patch * self.patch_size
        img_h = vertical_num_patch * self.patch_size
        # resize mask to the size of the image
        mask_np = cv.resize(mask_np.astype(np.uint8),
                            ((int(img_w), int(img_h))),
                            interpolation=cv.INTER_NEAREST).astype(bool)

        num_patches = rays_o_patches_all.shape[0]
        eval_patch_size = 1024
        comp_normal_map = np.zeros([img_h, img_w, 3])
        comp_normal_list = []

        for patch_idx in range(0, num_patches, eval_patch_size):
            rays_o_patch_center_batch = rays_o_patch_center[patch_idx:patch_idx+eval_patch_size]
            rays_d_patch_center_batch = rays_d_patch_center[patch_idx:patch_idx+eval_patch_size]
            rays_o_patches_all_batch = rays_o_patches_all[patch_idx:patch_idx+eval_patch_size]
            rays_v_patches_all_batch = rays_v_patches_all[patch_idx:patch_idx+eval_patch_size]
            rays_ez_batch = rays_ez[patch_idx:patch_idx+eval_patch_size]
            rays_A_inverse_batch = rays_A_inverse[patch_idx:patch_idx+eval_patch_size]

            near, far = self.dataset.near_far_from_sphere(rays_o_patch_center_batch,
                                                          rays_d_patch_center_batch)
            render_out = self.renderer.render(rays_o_patches_all_batch,
                                                    rays_v_patches_all_batch,
                                                    rays_ez_batch,
                                                    near, far,
                                                    rays_A_inverse_batch, gradient_method)

            comp_normal = render_out['comp_normal']
            comp_normal = comp_normal.detach().cpu().numpy()
            comp_normal_list.append(comp_normal)

        comp_normal_list = np.concatenate(comp_normal_list, axis=0)

        count = 0
        for i in range(0, img_h, self.patch_size):
            for j in range(0, img_w, self.patch_size):
                comp_normal_map[i:i+self.patch_size, j:j+self.patch_size] = comp_normal_list[count]
                count += 1
        normal_img_world = comp_normal_map

        rot = np.linalg.inv(self.dataset.pose_all[idx, :3, :3].detach().cpu().numpy())  # W2C rotation

        normal_img = np.matmul(rot, normal_img_world[..., None]).squeeze()
        normal_img[..., [1, 2]] *= -1
        normal_img_png = (np.squeeze(normal_img) * 128 + 128).clip(0, 255)
        normal_img_norm = np.linalg.norm(np.squeeze(normal_img), axis=2, keepdims=True)
        normal_dir = os.path.join(self.base_exp_dir, f'normals_validation_{gradient_method}', 'iter_{:0>6d}'.format(self.iter_step))
        os.makedirs(normal_dir, exist_ok=True)

        normal_img_normalized = np.squeeze(normal_img) / (normal_img_norm + 1e-7)
        normal_img_normalized = (np.squeeze(normal_img_normalized) * 128 + 128).clip(0, 255)

        normal_eval = np.zeros((img_h, img_w, 3))
        normal_eval[:normal_img_png.shape[0], :normal_img_png.shape[1]] = normal_img_png

        normal_eval_normalized = np.zeros((img_h, img_w, 3))
        normal_eval_normalized[:normal_img_normalized.shape[0], :normal_img_normalized.shape[1]] = normal_img_normalized

        normal_img_normalized = crop_image_by_mask(toRGBA(normal_eval_normalized.astype(np.uint8)[...,::-1], mask_np), mask_np)

        cv.imwrite(os.path.join(normal_dir, '{:0>8d}_{}_{}_rendered.png'.format(self.iter_step, 0, idx)),
                           normal_eval[..., ::-1])

        cv.imwrite(os.path.join(normal_dir, '{:0>8d}_{}_{}_normalized.png'.format(self.iter_step, 0, idx)),
                            normal_img_normalized)
        return normal_img_world, normal_dir

if __name__ == '__main__':
    print('Hello Wooden')

    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=FORMAT)

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/ele2.conf')  # real_close
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--mcube_threshold', type=float, default=0.0)
    parser.add_argument('--is_continue', default=False)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--case', type=str, default='mic')

    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)

    print(Fore.MAGENTA)
    # print(f'Running on the object: {args.case}')
    print(f'Using config: {args.conf[8:]}')
    print(Style.RESET_ALL)

    runner = Runner(args.conf, args.mode, args.case, args.is_continue)

    if args.mode == 'train':
        runner.train()
    elif args.mode == 'validate_mesh':
        runner.validate_mesh(0, resolution=2048)
    elif args.mode.startswith('interpolate'):  # Interpolate views given two image indices
        _, img_idx_0, img_idx_1 = args.mode.split('_')
        img_idx_0 = int(img_idx_0)
        img_idx_1 = int(img_idx_1)
        runner.interpolate_view(img_idx_0, img_idx_1)
