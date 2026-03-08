import torch
import torch.nn.functional as F
import cv2 as cv
import numpy as np
import os
from glob import glob
from icecream import ic
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp
import pyexr
import open3d as o3d
from .cameras import PinholeCamera
from icecream import ic
import pickle

# This function is borrowed from IDR: https://github.com/lioryariv/idr
def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose


class Dataset:
    def __init__(self, conf):
        super(Dataset, self).__init__()
        print('Load data: Begin')
        self.device = torch.device('cuda')
        self.conf = conf

        self.data_dir = conf.get_string('data_dir')
        normal_dir = conf.get_string('normal_dir')
        self.render_cameras_name = conf.get_string('render_cameras_name')
        self.object_cameras_name = conf.get_string('object_cameras_name')

        self.camera_outside_sphere = conf.get_bool('camera_outside_sphere', default=True)
        self.scale_mat_scale = conf.get_float('scale_mat_scale', default=1.1)
        self.type = conf.get_string('type')
        self.H = conf.get_int('height')
        self.W = conf.get_int('width')
        self.real_data = conf.get_bool('real_data')

        self.exclude_view_list = conf['exclude_views']
        ic(self.exclude_view_list)
        mesh_path = os.path.join(self.data_dir, 'mesh_Gt.ply')
        if os.path.exists(mesh_path):
            self.mesh_gt = o3d.io.read_triangle_mesh(mesh_path)
            o3d.io.write_triangle_mesh('mesh_Gt_fixed.ply', self.mesh_gt)
        else:
            self.mesh_gt = None


        if self.type == 'multi':
            self.pose_all = []
            self.intrinsics_all = []
            self.scale_mats_np = []
            
            self.masks = []
            self.scale_factor = 1.0

            self.normal_lis = sorted(glob(os.path.join(self.data_dir, normal_dir, '*.exr')))
            self.n_images = len(self.normal_lis)

            self.normal_np = [pyexr.read(im_name)[..., :3].astype(np.float32) / (1e-10 + np.linalg.norm(pyexr.read(im_name)[..., :3].astype(np.float32), axis=-1, keepdims=True)) for im_name in self.normal_lis]
            self.images = [torch.from_numpy(im_name).to(self.device) for im_name in self.normal_np]
            self.train_images = set(range(self.n_images)) - set(self.exclude_view_list)

            self.pose_path_list = sorted(glob(os.path.join(self.data_dir, '*.pkl')))
            self.mask_path_list = sorted(glob(os.path.join(self.data_dir, '*-depth.png')))

            
       
            self.cams = []

            for i in range(self.n_images):
                with open(self.pose_path_list[i], 'rb') as f:
                    self.cams.append(pickle.load(f))

                # self.images.append(np.array(cv.imread(f'{self.data_dir}/{i}.png') / 256.0).astype(np.float32))
                normal_np = pyexr.read(self.normal_lis[i])[..., :3].astype(np.float32) / (1e-10 + np.linalg.norm(pyexr.read(self.normal_lis[i])[..., :3].astype(np.float32), axis=-1, keepdims=True))
                self.images.append(normal_np)

                depth = cv.imread(self.mask_path_list[i],cv.IMREAD_UNCHANGED)
                depth = depth.astype(np.float32) / 65535 * 15
                mask = depth < 13
                mask = mask.astype(np.float32)
                self.masks.append(torch.from_numpy(np.repeat(mask[:, :, np.newaxis], 3, axis=2)).cpu())   # [n_images, H, W, 3]
                
                # pose
                pose = np.eye(4, dtype=np.float32)
                pose[:3, :] = self.cams[i][0].copy().astype(np.float32)
                intrinsics = np.eye(4).astype(np.float32)
                intrinsics[:3, :3] = self.cams[i][1].astype(np.float32)

                P = intrinsics@pose
                P = P[:3, :4]
                intrinsics, pose = load_K_Rt_from_P(None, P)

                # pose[:,3:] *= self.scale_factor
                self.pose_all.append(torch.from_numpy(pose).float())
                # intrinsics
                self.intrinsics_all.append(torch.from_numpy(intrinsics).float())

            self.pose_all = torch.stack(self.pose_all).to(self.device)  # [n_images, 4, 4]


            self.intrinsics_all = torch.stack(self.intrinsics_all).to(self.device)   # [n_images, 4, 4]
            self.intrinsics_all_inv = torch.inverse(self.intrinsics_all)
            
            self.focal = self.intrinsics_all[0][0, 0]
            
            self.object_bbox_min = np.array([-1.01, -1.01, -1.01])
            self.object_bbox_max = np.array([ 1.01,  1.01,  1.01])
            
            self.scale_mats_np = np.eye(4) # self.scale_mat
        
        elif self.type == 'normal':
            self.pose_all = []
            self.intrinsics_all = []
            self.scale_mats_np = []
            self.images = []
            self.masks = []
            self.scale_factor = 1.0
        
            # from skimage.io import imread
            self.n_images = len(glob(f'{self.data_dir}/*.pkl'))
            self.train_images = set(range(self.n_images)) - set(self.exclude_view_list)

            self.idx = [idx for idx in range(self.n_images)]
            self.cams = []
            self.normal_lis = sorted(glob(os.path.join(self.data_dir, normal_dir, '*.exr')))
            self.pose_path_list = sorted(glob(os.path.join(self.data_dir, '*.pkl')))
            self.mask_path_list = sorted(glob(os.path.join(self.data_dir, '*-depth.png')))

            # self.normal_np = np.stack([pyexr.read(im_name)[..., :3].astype(np.float32) / (1e-10 + np.linalg.norm(pyexr.read(im_name)[..., :3].astype(np.float32), axis=-1, keepdims=True)) for im_name in self.normal_lis])

            for i in range(self.n_images):
                with open(self.pose_path_list[i], 'rb') as f:
                    self.cams.append(pickle.load(f))

                # self.images.append(np.array(cv.imread(f'{self.data_dir}/{i}.png') / 256.0).astype(np.float32))
                normal_np = pyexr.read(self.normal_lis[i])[..., :3].astype(np.float32) / (1e-10 + np.linalg.norm(pyexr.read(self.normal_lis[i])[..., :3].astype(np.float32), axis=-1, keepdims=True))
                self.images.append(normal_np)

                depth = cv.imread(self.mask_path_list[i],cv.IMREAD_UNCHANGED)
                depth = depth.astype(np.float32) / 65535 * 15
                mask = depth < 13
                mask = mask.astype(np.float32)
                self.masks.append(torch.from_numpy(np.repeat(mask[:, :, np.newaxis], 3, axis=2)).cpu())   # [n_images, H, W, 3]
                
                # pose
                pose = np.eye(4, dtype=np.float32)
                pose[:3, :] = self.cams[i][0].copy().astype(np.float32)
                intrinsics = np.eye(4).astype(np.float32)
                intrinsics[:3, :3] = self.cams[i][1].astype(np.float32)

                # scale = 4
                # scale_mat = np.diag([scale, scale, scale, 1.0]).astype(np.float32)
                # P = intrinsics@pose@scale_mat

                P = intrinsics@pose
                P = P[:3, :4]
                intrinsics, pose = load_K_Rt_from_P(None, P)

                # pose[:,3:] *= self.scale_factor
                self.pose_all.append(torch.from_numpy(pose).float())
                # intrinsics
                self.intrinsics_all.append(torch.from_numpy(intrinsics).float())

            self.images = torch.from_numpy(np.stack(self.images)).cpu()
            self.masks = torch.from_numpy(np.stack(self.masks)).cpu()
            self.pose_all = torch.stack(self.pose_all).to(self.device)  # [n_images, 4, 4]

            self.intrinsics_all = torch.stack(self.intrinsics_all).to(self.device)   # [n_images, 4, 4]
            self.intrinsics_all_inv = torch.inverse(self.intrinsics_all)
            
            self.focal = self.intrinsics_all[0][0, 0]
            
            self.object_bbox_min = np.array([-1.01, -1.01, -1.01])
            self.object_bbox_max = np.array([ 1.01,  1.01,  1.01])
            
            self.scale_mats_np = np.eye(4) # self.scale_mat
        
        elif self.type == 'super':
            
            camera_dict = np.load(os.path.join(self.data_dir, self.render_cameras_name))
            self.camera_dict = camera_dict
            self.normal_lis = sorted(glob(os.path.join(self.data_dir, normal_dir, '*.exr')))
            self.n_images = len(self.normal_lis)
            self.train_images = set(range(self.n_images)) - set(self.exclude_view_list)
            self.img_idx_list = [int(os.path.basename(x).split('.')[0]) for x in self.normal_lis]

            print("loading normal maps...")
            self.normal_np = np.stack([pyexr.read(im_name)[..., :3] for im_name in self.normal_lis])
            self.normals = torch.from_numpy(self.normal_np.astype(np.float32)).to(self.device)  # [n_images, H, W, 3]
            self.images = self.normals
            print("loading normal maps done.")

            self.masks_lis = sorted(glob(os.path.join(self.data_dir, 'mask/*.png')))
            self.masks_np = np.stack([cv.imread(im_name) for im_name in self.masks_lis]) / 255.0        # (20, 512, 612, 3)

            self.masks_np = self.masks_np[..., 0]       # (20, 512, 612)
            # self.total_pixel = np.sum(self.masks_np)

            # set background of normal map to 0
            self.normal_np[self.masks_np == 0] = 0

            # world_mat is a projection matrix from world to image
            self.world_mats_np = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in self.img_idx_list]
            self.scale_mats_np = []

            # scale_mat: used for coordinate normalization, we assume the scene to render is inside a unit sphere at origin.
            self.scale_mats_np = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in self.img_idx_list]

            self.intrinsics_all = []
            self.pose_all = []
            self.V_inverse_all = []

            self.H, self.W = self.normal_np.shape[1], self.normal_np.shape[2]
            for scale_mat, world_mat, normal_map, mask in zip(self.scale_mats_np, self.world_mats_np, self.normals, self.masks_np):
                P = world_mat @ scale_mat
                P = P[:3, :4]
                intrinsics, pose = load_K_Rt_from_P(None, P)
                
                self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
                self.pose_all.append(torch.from_numpy(pose).float())

                intrinsics_inverse = torch.inverse(torch.from_numpy(intrinsics).float())
                pose = torch.from_numpy(pose).float()
                # compute the V_inverse
                tx = torch.linspace(0, self.W - 1, int(self.W))
                ty = torch.linspace(0, self.H - 1, int(self.H))
                pixels_x, pixels_y = torch.meshgrid(tx, ty)
                p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).to(intrinsics_inverse.device)  # W, H, 3
                p = torch.matmul(intrinsics_inverse[None, None, :3, :3],
                                p[:, :, :, None]).squeeze()  # W, H, 3
                rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
                rays_v = torch.matmul(pose[None, None, :3, :3],
                                    rays_v[:, :, :, None]).squeeze()  # W, H, 3
                rays_v = rays_v.transpose(0, 1) # H, W, 3

                # the axis direction of the camera coordinate system in the world coordinate system
                rays_right = pose[None, :3, 0].expand(rays_v.shape)  # H, W, 3
                rays_down = pose[None, :3, 1].expand(rays_v.shape)  # H, W, 3

                V_concat = torch.cat([rays_v[..., None, :],
                                    rays_right[..., None, :],
                                    rays_down[..., None, :]], dim=-2)  # (H, W, 3, 3)

                # computing the inverse may take a while if the resolution is high
                # For 512x612, it takes about 0.8ms
                V_inverse = torch.inverse(V_concat)  # (H, W, 3, 3)
                self.V_inverse_all.append(V_inverse)

            self.masks = torch.from_numpy(self.masks_np.astype(np.float32)) # [n_images, H, W, 3]
            self.intrinsics_all = torch.stack(self.intrinsics_all)   # [n_images, 4, 4]
            self.intrinsics_all_inv = torch.inverse(self.intrinsics_all)  # [n_images, 4, 4]
            self.focal_length = self.intrinsics_all[0][0, 0]
            self.pose_all = torch.stack(self.pose_all)  # [n_images, 4, 4]
            self.image_pixels = self.H * self.W
            self.V_inverse_all = torch.stack(self.V_inverse_all)  # [n_images, H, W, 3, 3]
            self.object_bbox_min = np.array([-1.01, -1.01, -1.01])
            self.object_bbox_max = np.array([ 1.01,  1.01,  1.01])
            self.scale_mats_np = self.scale_mats_np[0]
        elif self.type == 'only_for_get_c2w':
            camera_dict = np.load(os.path.join(self.data_dir, self.render_cameras_name))
            self.camera_dict = camera_dict

            # world_mat is a projection matrix from world to image
            self.masks_lis = sorted(glob(os.path.join(self.data_dir, 'mask/*.png')))
            self.img_idx_list = [int(os.path.basename(x).split('.')[0]) for x in self.masks_lis]
            self.world_mats_np = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in self.img_idx_list]
            self.scale_mats_np = []

            # scale_mat: used for coordinate normalization, we assume the scene to render is inside a unit sphere at origin.
            self.scale_mats_np = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in self.img_idx_list]

            self.intrinsics_all = []
            self.pose_all = []
            self.V_inverse_all = []

            for scale_mat, world_mat in zip(self.scale_mats_np, self.world_mats_np):
                P = world_mat @ scale_mat
                P = P[:3, :4]
                intrinsics, pose = load_K_Rt_from_P(None, P)
                self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
                self.pose_all.append(torch.from_numpy(pose).float())

                intrinsics_inverse = torch.inverse(torch.from_numpy(intrinsics).float())
                pose = torch.from_numpy(pose).float()

            self.pose_all = torch.stack(self.pose_all).to(self.device)  # [n_images, 4, 4]
            np.save(self.data_dir+'/pose.npy',self.pose_all.cpu().numpy())


        else:
            camera_dict = np.load(os.path.join(self.data_dir, self.render_cameras_name))
            self.camera_dict = camera_dict
            self.images_lis = sorted(glob(os.path.join(self.data_dir, 'image/*')))
            self.n_images = len(self.images_lis)
            self.images_np = np.stack([cv.imread(im_name) for im_name in self.images_lis]) / 256.0
            self.masks_lis = sorted(glob(os.path.join(self.data_dir, 'mask/*')))
            self.masks_np = np.stack([cv.imread(im_name) for im_name in self.masks_lis]) / 256.0

            self.world_mats_np = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

            self.scale_mats_np = []

            self.scale_mats_np = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

            self.intrinsics_all = []
            self.pose_all = []

            for scale_mat, world_mat in zip(self.scale_mats_np, self.world_mats_np):
                P = world_mat @ scale_mat
                P = P[:3, :4]
                intrinsics, pose = load_K_Rt_from_P(None, P)
                self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
                self.pose_all.append(torch.from_numpy(pose).float())

            self.images = torch.from_numpy(self.images_np.astype(np.float32)).to(self.device) # .cpu() # CPU  # [n_images, H, W, 3]
            self.masks  = torch.from_numpy(self.masks_np.astype(np.float32)).to(self.device) # .cpu() # CPU  # [n_images, H, W, 3]
            self.intrinsics_all = torch.stack(self.intrinsics_all).to(self.device)   # [n_images, 4, 4]
            self.intrinsics_all_inv = torch.inverse(self.intrinsics_all)  # [n_images, 4, 4]
            self.focal = self.intrinsics_all[0][0, 0]
            self.pose_all = torch.stack(self.pose_all).to(self.device)  # [n_images, 4, 4]
            
            

            object_bbox_min = np.array([-1.01, -1.01, -1.01, 1.0])
            object_bbox_max = np.array([ 1.01,  1.01,  1.01, 1.0])
            object_scale_mat = np.load(os.path.join(self.data_dir, self.object_cameras_name))['scale_mat_0']
            object_bbox_min = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_min[:, None]
            object_bbox_max = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_max[:, None]
            self.object_bbox_min = object_bbox_min[:3, 0]
            self.object_bbox_max = object_bbox_max[:3, 0]

        self.idx = [idx for idx in range(self.n_images)]
        np.save(self.data_dir+'/pose.npy',self.pose_all.cpu().numpy())
        print('Load data: End')

    def gen_patches_at(self, img_idx, resolution_level=1, patch_H=3, patch_W=3):
        tx = torch.linspace(0, self.W - 1, int(self.W // resolution_level))
        ty = torch.linspace(0, self.H - 1, int(self.H // resolution_level))
        pixels_y, pixels_x = torch.meshgrid(ty, tx)

        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1) # H, W, 3
        p = torch.matmul(self.intrinsics_all_inv[img_idx, :3, :3], p[..., None]).squeeze()  # H, W, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        rays_v = torch.matmul(self.pose_all[img_idx, :3, :3], rays_v[:, :, :, None]).squeeze()  # H, W, 3

        # split rays_v into non-overlapping patches
        height, width, _ = rays_v.shape
        horizontal_num_patch = width // patch_W
        vertical_num_patch = height // patch_H
        rays_v_patches_all = []
        rays_V_inverse_patches_all = []
        rays_ez_patches_all = []
        mask_value = []
        for i in range(0, height-patch_H//2-1, patch_H):
            for j in range(0, width-patch_W//2-1, patch_W):
                rays_v_patch = rays_v[i:i + patch_H, j:j + patch_W]
                rays_v_patches_all.append(rays_v_patch)

                rays_V_inverse_patch = self.V_inverse_all[img_idx][i:i + patch_H, j:j + patch_W]
                rays_V_inverse_patches_all.append(rays_V_inverse_patch)

                rays_ez_patch = self.normals[img_idx][i + patch_H//2, j + patch_W//2]
                rays_ez_patches_all.append(rays_ez_patch)

                mask_value.append(self.masks_np[img_idx][i + patch_H//2, j + patch_W//2].astype(bool))
        rays_v_patches_all = torch.stack(rays_v_patches_all, dim=0)  # (num_patch, patch_H, patch_W, 3)
        rays_V_inverse_patches_all = torch.stack(rays_V_inverse_patches_all, dim=0)  # (num_patch, patch_H, patch_W, 3, 3)
        rays_o_patches_all = self.pose_all[img_idx, :3, 3].expand(rays_v_patches_all.shape)  # (num_patch, patch_H, patch_W, 3)

        rays_o_patch_center = rays_o_patches_all[:, patch_H//2, patch_W//2]  # (num_patch, 3)
        rays_d_patch_center = rays_v_patches_all[:, patch_H//2, patch_W//2]  # (num_patch, 3)

        marching_plane_normal_patches_all = self.pose_all[img_idx, :3, 2].expand(rays_d_patch_center.shape)  # (num_patch, 3)

        return rays_o_patch_center, \
                rays_d_patch_center, \
            rays_o_patches_all, \
            rays_v_patches_all, \
            marching_plane_normal_patches_all, \
            rays_V_inverse_patches_all, horizontal_num_patch, vertical_num_patch
    
    def gen_rays_at(self, img_idx, resolution_level=1):
        """
        Generate rays at world space from one camera.
        """
        H,W,_ = self.images[img_idx].shape
        l = resolution_level
        tx = torch.linspace(0, W - 1, W // l)
        ty = torch.linspace(0, H - 1, H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1) # W, H, 3
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, None, :3, :3].to(self.device), p[:, :, :, None].to(self.device)).squeeze()  # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        rays_v = torch.matmul(self.pose_all[img_idx, None, None, :3, :3].to(self.device), rays_v[:, :, :, None].to(self.device)).squeeze()  # W, H, 3
        rays_o = self.pose_all[img_idx, None, None, :3, 3].expand(rays_v.shape)  # W, H, 3

        # 现在只有1种相机参数，所以只用调用1次，radiis和ray_cos
        origins, directions, radiis, ray_cos = self.pinhole(img_idx,H,W)
        index_x = pixels_x.to(torch.long)
        index_y = pixels_y.to(torch.long)
        radiis_select = radiis[index_y, index_x, :]
        ray_cos_select = ray_cos[index_y, index_x, :]
        directions_select = directions[index_y, index_x, :]
        directions_select = (self.pose_all[img_idx, None, None, :3, :3].to(self.device) @ directions_select[..., None]).squeeze()
        # marching_plane_normal = self.pose_all[img_idx, :3, 2].expand(directions_select.shape)  # (num_patch, 3)

        return rays_o.transpose(0, 1), directions_select.transpose(0, 1), radiis_select.transpose(0, 1), ray_cos_select.transpose(0, 1)

    def gen_random_rays_at(self, batch_size):
        """
        Generate random rays at world space from one camera.
        """
        # self.train_images = [0,1,2,3,4]
        img_idx = np.random.choice(list(self.train_images))
        H,W,_ = self.images[img_idx].shape
        pixels_x = torch.randint(low=0, high=W, size=[batch_size])
        pixels_y = torch.randint(low=0, high=H, size=[batch_size])
        color = self.images[img_idx][(pixels_y.to('cpu'), pixels_x.to('cpu'))]    # batch_size, 3
        mask = self.masks[img_idx][(pixels_y.to('cpu'), pixels_x.to('cpu'))]      # batch_size, 3

        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).float()  # batch_size, 3
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, :3, :3].to(self.device), p[:, :, None].to(self.device)).squeeze() # batch_size, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)    # batch_size, 3
        rays_v = torch.matmul(self.pose_all[img_idx, None, :3, :3].to(self.device), rays_v[:, :, None]).squeeze()  # batch_size, 3
        rays_o = self.pose_all[img_idx, None, :3, 3].to(self.device).expand(rays_v.shape) # batch_size, 3   和trimiprf一致
        # return torch.cat([rays_o.cpu(), rays_v.cpu(), color, mask[:, :1], radiis_select.cpu(), ray_cos_select.cpu()], dim=-1).cuda()    # batch_size, 10
    

        # 现在只有1种相机参数，所以只用调用1次，radiis和ray_cos
        origins, directions, radiis, ray_cos = self.pinhole(img_idx,H,W)       # H*W*3 这与原始的NeuS W*H*3不符
        radiis_select = radiis[pixels_y, pixels_x, :]
        ray_cos_select = ray_cos[pixels_y, pixels_x , :]
        
        directions_select = directions[pixels_y,pixels_x, :]
        directions_select = (self.pose_all[img_idx, None, :3, :3].to(self.device) @ directions_select[..., None]).squeeze()
        # 这里的directions_select相比于rays_v有0.5像素的偏移，方向向量指向像素中心

        # marching_plane_normal = self.pose_all[img_idx, :3, 2].expand((batch_size, 3))  # (num_patch, 3)
        pixel_idx = (pixels_x, pixels_y)
        if self.real_data == True:
            return torch.cat([rays_o.cpu(), directions_select.cpu(), color.cpu(), mask.unsqueeze(-1).cpu(), radiis_select.cpu(), ray_cos_select.cpu()], dim=-1).cuda(), img_idx, pixel_idx    # batch_size, 10
        else:
            return torch.cat([rays_o.cpu(), directions_select.cpu(), color.cpu(), mask[:, :1].cpu(), radiis_select.cpu(), ray_cos_select.cpu()], dim=-1).cuda(), img_idx, pixel_idx    # batch_size, 10
    
    def pinhole(self,img_idx,H,W,scale=1):
        # 给定相机内参，计算r, cos等等，所以不同的相机对应不同的半径
        fx = self.intrinsics_all[img_idx][0][0]      # /scale
        fy = self.intrinsics_all[img_idx][1][1]      # /scale
        cx = self.intrinsics_all[img_idx][0][2]      # W/(2*scale)
        cy = self.intrinsics_all[img_idx][1][2]      # H/(2*scale)

        cameras = PinholeCamera(
                    fx=fx,
                    fy=fy,
                    cx=cx,
                    cy=cy,
                    width=W,
                    height=H,
                )
        origins, directions, radiis, ray_cos = cameras.build(self.device,H,W,scale)
        radiis[:] = radiis.mean()
        return origins, directions, radiis, ray_cos     # 这里的origins, directions是相机坐标系 可以不管，但是radiis, ray_cos没有变化

    def gen_rays_between(self, idx_0, idx_1, ratio, resolution_level=1):
        """
        Interpolate pose between two cameras.
        """
        self.W, self.H = 800,800
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1)  # W, H, 3
        p = torch.matmul(self.intrinsics_all_inv[0, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        trans = self.pose_all[idx_0, :3, 3] * (1.0 - ratio) + self.pose_all[idx_1, :3, 3] * ratio
        pose_0 = self.pose_all[idx_0].detach().cpu().numpy()
        pose_1 = self.pose_all[idx_1].detach().cpu().numpy()
        pose_0 = np.linalg.inv(pose_0)
        pose_1 = np.linalg.inv(pose_1)
        rot_0 = pose_0[:3, :3]
        rot_1 = pose_1[:3, :3]
        rots = Rot.from_matrix(np.stack([rot_0, rot_1]))
        key_times = [0, 1]
        slerp = Slerp(key_times, rots)
        rot = slerp(ratio)
        pose = np.diag([1.0, 1.0, 1.0, 1.0])
        pose = pose.astype(np.float32)
        pose[:3, :3] = rot.as_matrix()
        pose[:3, 3] = ((1.0 - ratio) * pose_0 + ratio * pose_1)[:3, 3]
        pose = np.linalg.inv(pose)
        rot = torch.from_numpy(pose[:3, :3]).cuda()
        trans = torch.from_numpy(pose[:3, 3]).cuda()
        rays_v = torch.matmul(rot[None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = trans[None, None, :3].expand(rays_v.shape)  # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

    def near_far_from_sphere(self, rays_o, rays_d):
        a = torch.sum(rays_d**2, dim=-1, keepdim=True).to(self.device)
        b = 2.0 * torch.sum(rays_o.to(self.device) * rays_d.to(self.device), dim=-1, keepdim=True)
        mid = 0.5 * (-b) / a
        near = mid - 1.0
        far = mid + 1.0
        return near, far

    def image_at(self, idx, resolution_level):
        img = cv.imread(self.images_lis[idx])
        H,W,_ = self.images[idx].shape
        return (cv.resize(img, (W // resolution_level, H // resolution_level))).clip(0, 255)

