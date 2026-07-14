import os
import sys
import torch
import numpy as np
import trimesh
from numbers import Number
from typing import *
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = "1"
import trimesh

import math
import json
import cv2
from scipy import ndimage
from scipy.spatial.transform import Rotation as R
import nvdiffrast.torch as dr
import time
import skimage as ski

import warnings
class no_warnings:
    def __init__(self, action: str = 'ignore', **kwargs):
        self.action = action
        self.filter_kwargs = kwargs
    
    def __call__(self, fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            with warnings.catch_warnings():
                warnings.simplefilter(self.action, **self.filter_kwargs)
                return fn(*args, **kwargs)
        return wrapper  
    
    def __enter__(self):
        self.warnings_manager = warnings.catch_warnings()
        self.warnings_manager.__enter__()
        warnings.simplefilter(self.action, **self.filter_kwargs)
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.warnings_manager.__exit__(exc_type, exc_val, exc_tb)

def directions_to_spherical_uv_torch(directions):
    directions = directions / torch.linalg.norm(directions, dim=-1, keepdims=True)
    u = 1 - torch.arctan2(directions[..., 1], directions[..., 0]) / (2 * torch.pi) % 1.0
    v = torch.arccos(directions[..., 2]) / torch.pi
    return torch.stack([u, v], dim=-1)


def image_uv(
    height: int,
    width: int,
    left: int = None,
    top: int = None,
    right: int = None,
    bottom: int = None,
    dtype: np.dtype = np.float32
) -> np.ndarray:

    if left is None: left = 0
    if top is None: top = 0
    if right is None: right = width
    if bottom is None: bottom = height
    u = np.linspace((left + 0.5) / width, (right - 0.5) / width, right - left, dtype=dtype)
    v = np.linspace((top + 0.5) / height, (bottom - 0.5) / height, bottom - top, dtype=dtype)
    u, v = np.meshgrid(u, v, indexing='xy')
    return np.stack([u, v], axis=2)
def spherical_uv_to_directions(uv: np.ndarray):
    theta, phi = (1 - uv[..., 0]) * (2 * np.pi), uv[..., 1] * np.pi
    directions = np.stack([np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)], axis=-1)
    return directions
def uv_to_pixel_torch(
    uv,
    width,
    height,
):
    """
    Args:
        pixel (np.ndarray): [..., 2] pixel coordinrates defined in image space,  x range is (0, W - 1), y range is (0, H - 1)
        width (int | np.ndarray): [...] image width(s)
        height (int | np.ndarray): [...] image height(s)

    Returns:
        (np.ndarray): [..., 2] pixel coordinrates defined in uv space, the range is (0, 1)
    """
    if not isinstance(width, int):
        pixel = uv * torch.stack([width, height], dim=-1).to(uv.dtype).to(uv.device) - 0.5
    else:
        Ns = uv.ndim
        tmp = torch.tensor([width, height]).to(uv.dtype).to(uv.device)
        for i in range(Ns - 1):
            tmp = tmp.unsqueeze(0)
        pixel = uv * tmp - 0.5
    return pixel
def depth_edge_torch(depth: np.ndarray, atol: float = None, rtol: float = None, kernel_size: int = 3, mask = None) -> np.ndarray:
    """
    Compute the edge mask from depth map. The edge is defined as the pixels whose neighbors have large difference in depth.
    
    Args:
        depth (np.ndarray): shape (..., height, width), linear depth map
        atol (float): absolute tolerance
        rtol (float): relative tolerance

    Returns:
        edge (np.ndarray): shape (..., height, width) of dtype torch.bool
    """
    func = torch.nn.MaxPool2d(kernel_size, stride=1, padding=kernel_size//2)
    if mask is None:
        # diff = (max_pool_2d(depth, kernel_size, stride=1, padding=kernel_size // 2) + max_pool_2d(-depth, kernel_size, stride=1, padding=kernel_size // 2))
        diff = (func(depth[None,None]) + func(-depth[None,None]))[0,0]
    else:
        diff = (func(torch.where(mask, depth, -torch.inf)[None,None]) + func(torch.where(mask, -depth, -torch.inf)[None,None]))[0,0]

    edge = torch.zeros_like(depth, dtype=torch.bool, device=depth.device)
    if atol is not None:
        edge |= diff > atol
    
    if rtol is not None:
        edge |= diff / depth > rtol
    return edge


def correct_pano_depth_batch(pano_depth):
    # ok...
    # pano_depth: [N, H, W] numpy array
    # pano_rgb: [H, W, 3] numpy array
    # spherical_uv_to_directions(uv: np.ndarray):
    N, H, W = pano_depth.shape
    uv = image_uv(width=W, height=H)
    directions = spherical_uv_to_directions(uv)

    pivot_directions = np.stack([directions[0, W//2],directions[H//2, W//2],directions[H-1, W//2],directions[H//2, 0],directions[H//2, W//4],directions[H//2, 3*W//4]])

    cos = (np.sum(pivot_directions[:,None,None,None,:] * directions[None], axis=-1)).max(axis=0)

    pano_depth_correct = pano_depth / cos
    
    return pano_depth_correct

def unreal_to_opencv_w2c(pitch, yaw, roll, x, y, z):
    # Step 1: 从欧拉角构造旋转矩阵（Unreal顺序）
    r_unreal = R.from_euler('YXZ', [yaw, pitch, roll], degrees=True)
    R_unreal = r_unreal.as_matrix()  # 3x3旋转矩阵

    # Step 3: 世界到相机坐标（变换旋转和平移）
    R_cv = R_unreal
    # t_cv = np.array([x, y, -z])
    t_cv = np.array([y, -z, x])

    # Step 4: 构造4x4的 W2C 矩阵
    T_c2w = np.eye(4)
    T_c2w[:3, :3] = R_cv
    T_c2w[:3, 3] = t_cv
    #print(f"tcv: {t_cv}")
    T_w2c = np.linalg.inv(T_c2w)

    return T_w2c


def image_uv_torch(height: int, width: int, left: int = None, top: int = None, right: int = None, bottom: int = None, device: torch.device = None, dtype: torch.dtype = None) -> torch.Tensor:

    if left is None: left = 0
    if top is None: top = 0
    if right is None: right = width
    if bottom is None: bottom = height
    u = torch.linspace((left + 0.5) / width, (right - 0.5) / width, right - left, device=device, dtype=dtype)
    v = torch.linspace((top + 0.5) / height, (bottom - 0.5) / height, bottom - top, device=device, dtype=dtype)
    u, v = torch.meshgrid(u, v, indexing='xy')
    uv = torch.stack([u, v], dim=-1)
    return uv

def spherical_uv_to_directions_torch(uv):
    # device = uv.device
    theta, phi = (1 - uv[..., 0]) * (2 * torch.pi), uv[..., 1] * torch.pi
    directions = torch.stack([torch.sin(phi) * torch.cos(theta), torch.sin(phi) * torch.sin(theta), torch.cos(phi)], axis=-1)
    return directions

def get_pano_pcs_torch(pano_depth):
    device = pano_depth.device
    H, W = pano_depth.shape
    uv = image_uv_torch(width=W, height=H, device=device)
    directions = spherical_uv_to_directions_torch(uv)

    vertices_cam = pano_depth[:,:,None] * directions
    return vertices_cam.reshape((-1,3))

def get_world_pcs_pano_torch(csv_cam, depth):
    device = depth.device
    # 提取深度在相机系下的点云；
    rot_matrix = torch.tensor([
        [0,1,0],[0,0,-1],[-1,0,0.]
    ], dtype=depth.dtype,device=device)

    cam_pcs = get_pano_pcs_torch(depth) @ rot_matrix.T
    X_,Y_,Z_,p,y,r = csv_cam
    w2c = unreal_to_opencv_w2c(p,y,r,X_,Y_,Z_)
    c2w = torch.from_numpy(np.linalg.inv(w2c)).to(device).to(depth.dtype)
    # print(cam_pcs.dtype, c2w.dtype)
    world_pcs = cam_pcs @ c2w[:3,:3].T + c2w[:3,3][None]
    return world_pcs

def get_world_pcs_pano_torch_Rt(Rt, depth):
    device = depth.device
    # 提取深度在相机系下的点云；
    rot_matrix = torch.tensor([
        [0,1,0],[0,0,-1],[-1,0,0.]
    ], dtype=depth.dtype,device=device)

    cam_pcs = get_pano_pcs_torch(depth) @ rot_matrix.T

    w2c = Rt
    c2w = torch.from_numpy(np.linalg.inv(w2c)).to(device).to(depth.dtype)
    # print(cam_pcs.dtype, c2w.dtype)
    world_pcs = cam_pcs @ c2w[:3,:3].T + c2w[:3,3][None]
    return world_pcs


def get_mesh_from_pano(pano_depth, pano_mask, csv_cam,dtype=np.float32):
    # ok...
    # pano_depth: [H, W] numpy array
    # pano_rgb: [H, W, 3] numpy array
    # spherical_uv_to_directions(uv: np.ndarray):
    H, W = pano_depth.shape
    uv = image_uv(width=W, height=H)
    pano_depth_ = torch.from_numpy(pano_depth).float()
    wrd_pcs = get_world_pcs_pano_torch(csv_cam, pano_depth_)
    vertices = wrd_pcs.reshape(H,W,3).numpy()
    #directions = spherical_uv_to_directions(uv)
    #vertices = pano_depth[:,:,None] * directions + origin_position
    
    pts_topology = np.zeros((H - 1, W, 2, 3), dtype=np.int32)
    index_ = np.arange(H*W).astype(np.int32).reshape(H,W)
    pts_topology[:,:-1,0,0] = index_[:-1,:-1]
    pts_topology[:,:-1,0,2] = index_[:-1,1:]
    pts_topology[:,:-1,0,1] = index_[1:,:-1]
    pts_topology[:,-1,0,0] = index_[:-1,-1]
    pts_topology[:,-1,0,2] = index_[:-1,0]
    pts_topology[:,-1,0,1] = index_[1:,-1]


    pts_topology[:,:,1,0] = index_[:-1,1:]
    pts_topology[:,:,1,1] = index_[1:,:-1]
    pts_topology[:,:,1,2] = index_[1:,1:]
    pts_topology[:,-1,1,0] = index_[:-1,0]
    pts_topology[:,-1,1,1] = index_[1:,-1]
    pts_topology[:,-1,1,2] = index_[1:,0]

    pano_mask_ext = np.zeros((H,W+1),dtype=np.bool_)
    pano_mask_ext[:,:-1] = pano_mask[:,:]
    pano_mask_ext[:,-1] = pano_mask[:,0]

    topo_mask_0 = pano_mask_ext[:-1,:-1] + pano_mask_ext[:-1,1:] + pano_mask_ext[1:,:-1]
    topo_mask_1 = pano_mask_ext[:-1,1:] + pano_mask_ext[1:,:-1] + pano_mask_ext[1:,1:]


    topo0 = pts_topology[:,:,0][topo_mask_0]
    topo1 = pts_topology[:,:,1][topo_mask_1]

    topo = np.concatenate([topo0, topo1], axis=0)

    vertices = vertices.reshape(-1,3)
    mesh_trimesh = trimesh.Trimesh(vertices=vertices, faces=topo)

    print(f"mesh construction complete; mesh info: {mesh_trimesh.vertices.shape} {mesh_trimesh.faces.shape}")

    return mesh_trimesh

def get_mesh_from_pano_Rt(pano_depth, pano_mask, Rt, dtype=np.float32, pano_rgb=None, alpha_mask=None):
    # ok...
    # pano_depth: [H, W] numpy array
    # pano_rgb: [H, W, 3] numpy array
    # spherical_uv_to_directions(uv: np.ndarray):
    H, W = pano_depth.shape
    uv = image_uv(width=W, height=H)
    pano_depth_ = torch.from_numpy(pano_depth).float()
    wrd_pcs = get_world_pcs_pano_torch_Rt(Rt, pano_depth_)
    vertices = wrd_pcs.reshape(H,W,3).numpy()
    #directions = spherical_uv_to_directions(uv)
    #vertices = pano_depth[:,:,None] * directions + origin_position
    
    pts_topology = np.zeros((H - 1, W, 2, 3), dtype=np.int32)
    index_ = np.arange(H*W).astype(np.int32).reshape(H,W)
    pts_topology[:,:-1,0,0] = index_[:-1,:-1]
    pts_topology[:,:-1,0,2] = index_[:-1,1:]
    pts_topology[:,:-1,0,1] = index_[1:,:-1]
    pts_topology[:,-1,0,0] = index_[:-1,-1]
    pts_topology[:,-1,0,2] = index_[:-1,0]
    pts_topology[:,-1,0,1] = index_[1:,-1]


    pts_topology[:,:-1,1,0] = index_[:-1,1:]
    pts_topology[:,:-1,1,1] = index_[1:,:-1]
    pts_topology[:,:-1,1,2] = index_[1:,1:]
    pts_topology[:,-1,1,0] = index_[:-1,0]
    pts_topology[:,-1,1,1] = index_[1:,-1]
    pts_topology[:,-1,1,2] = index_[1:,0]

    pano_mask_ext = np.zeros((H,W+1),dtype=np.bool_)
    pano_mask_ext[:,:-1] = pano_mask[:,:]
    pano_mask_ext[:,-1] = pano_mask[:,0]

    topo_mask_0 = pano_mask_ext[:-1,:-1] + pano_mask_ext[:-1,1:] + pano_mask_ext[1:,:-1]
    topo_mask_1 = pano_mask_ext[:-1,1:] + pano_mask_ext[1:,:-1] + pano_mask_ext[1:,1:]

    topo0 = pts_topology[:,:,0][topo_mask_0]
    topo1 = pts_topology[:,:,1][topo_mask_1]

    topo = np.concatenate([topo0, topo1], axis=0)

    vertices = vertices.reshape(-1,3)
    if pano_rgb is None and alpha_mask is None:
        mesh_trimesh = trimesh.Trimesh(vertices=vertices, faces=topo)
    elif alpha_mask is None:
        pano_rgb = pano_rgb.reshape(-1,3)
        mesh_trimesh = trimesh.Trimesh(vertices=vertices, faces=topo, vertex_colors = pano_rgb)
    elif alpha_mask is not None:
        if pano_rgb is not None:
            pano_rgb = pano_rgb.reshape(-1,3)
        else:
            pano_rgb = np.ones((H*W,3),dtype=np.float32)
        pano_rgb_ = np.ones((H*W,4),dtype=np.float32)
        pano_rgb_[:,:3] = pano_rgb
        alpha_mask = alpha_mask.reshape(-1)
        pano_rgb_[~alpha_mask] = 0. 
        mesh_trimesh = trimesh.Trimesh(vertices=vertices, faces=topo, vertex_colors = pano_rgb_)
    print(f"mesh construction complete; mesh info: {mesh_trimesh.vertices.shape} {mesh_trimesh.faces.shape}")

    return mesh_trimesh

def get_diffrast_camera_parameter_from_cv(K, H, W, near, far, device):
    K_ = torch.zeros((4,4), dtype=torch.float32, device=device)
    K_[0,0] = K[0,0] * 2. / W
    K_[1,1] = K[1,1] * 2. / H
    K_[2,2] = (far + near) / (far - near) 
    K_[2,3] = - 2. * (near * far) / (far - near) 
    K_[3,2] = 1.
    return K_

def get_mesh_render_depth(mesh, Ks, Rts, H, W, near, far, device):
    #print("begin",Ks.shape, Rts.shape)
    with torch.no_grad():
        if isinstance(Ks, np.ndarray):
            Ks = torch.from_numpy(Ks).float().to(device)
            Rts = torch.from_numpy(Rts).float().to(device)
        #print("sdafa0")
        glctx = dr.RasterizeCudaContext(device=device)
        #glctx = dr.RasterizeGLContext(device=device)
        #print("sdafa_u")
        pos_tensor = torch.from_numpy(np.array(mesh.vertices)).float().to(device)
        tri_tensor = torch.from_numpy(np.array(mesh.faces)).int().to(device)

        pos_qc = torch.ones((Rts.shape[0],pos_tensor.shape[0],4),dtype=torch.float32, device=device)

        N = Rts.shape[0]
        all_images = []
        #print("sdafa")

        print(pos_tensor.shape, Rts.shape)
        pos_tensor_cam = pos_tensor[None] @ Rts[:,:3,:3].permute(0,2,1) + Rts[:,:3,3][:,None,:]

        K_ = get_diffrast_camera_parameter_from_cv(Ks[0], H, W, near, far, device)
        pos_qc[:,:,:3] = pos_tensor_cam
        dist = pos_tensor_cam.norm(dim=-1)[:,:,None].repeat(1,1,2)
        dist[:,:,1] = 1.
        pos_rast = (pos_qc @ (K_.T)[None])

        rast, _ = dr.rasterize(glctx, pos_rast, tri_tensor, resolution=[H, W],grad_db=False)
        out, _ = dr.interpolate(dist, rast, tri_tensor)

        img = out.cpu().numpy()[:, :, :, :] # Flip vertically.
        all_images=(img[:,:,:,:2])

        all_canvas = all_images[:,:,:,0]
        all_canvas_mask = all_images[:,:,:,1] > 1. - 1e-4
        #all_canvas,all_canvas_mask = fill_image_nvdiffrast(all_canvas,all_canvas_mask)
        return all_canvas,all_canvas_mask

def get_mesh_render_color(mesh, Ks, Rts, H, W, near, far, device):
    with torch.no_grad():
        if isinstance(Ks, np.ndarray):
            Ks = torch.from_numpy(Ks).float().to(device)
            Rts = torch.from_numpy(Rts).float().to(device)
        # import pdb
        # pdb.set_trace()
        glctx = dr.RasterizeCudaContext(device=device)
        pos_tensor = torch.from_numpy(np.array(mesh.vertices)).float().to(device)
        col_tensor = torch.from_numpy(np.array(mesh.visual.vertex_colors)).float().to(device)
        print(col_tensor.shape,col_tensor.max())
        tri_tensor = torch.from_numpy(np.array(mesh.faces)).int().to(device)
        if col_tensor.max() > 5:
            col_tensor /= 255.
        pos_qc = torch.ones((pos_tensor.shape[0],4),dtype=torch.float32, device=device)

        N = Rts.shape[0]
        all_images = []

        for i in range(N):
            pos_tensor_cam = pos_tensor @ Rts[i,:3,:3].T + Rts[i:i+1,:3,3]
            K_ = get_diffrast_camera_parameter_from_cv(Ks[i], H, W, near, far, device)
            pos_qc[:,:3] = pos_tensor_cam
            pos_rast = (pos_qc @ K_.T)[None]
            rast, _ = dr.rasterize(glctx, pos_rast, tri_tensor, resolution=[H, W])
            out, _ = dr.interpolate(col_tensor[None], rast, tri_tensor)
            # out = dr.antialias(out, rast, pos_rast, tri_tensor, topology_hash=None, pos_gradient_boost=1.0)
            #out = dr.antialias(out, rast, pos_rast, tri_tensor, topology_hash=None, pos_gradient_boost=1.0)
            img = out.cpu().numpy()[0, :, :, :] # Flip vertically.
            #img = np.clip(np.rint(img * 255), 0, 255).astype(np.uint8) # Quantize to np.uint8
            all_images.append(img)
        all_images = np.stack(all_images, axis=0)
        all_canvas = all_images[:,:,:,:3]
        all_canvas_mask = all_images[:,:,:,3] > 0.999
        #all_canvas,all_canvas_mask = fill_image_nvdiffrast(all_canvas,all_canvas_mask)
        return all_canvas,all_canvas_mask
    



def csv_cam_to_opencv(csv_cams):
    frame_size = len(csv_cams)
    rot_matrix = np.array([
        [0,1,0],[0,0,-1],[-1,0,0.]
    ], dtype=np.float32)
    all_w2cs = []
    for i in range(frame_size):
        X_,Y_,Z_,p,y,r = csv_cams[i]
        cur_w2c = (unreal_to_opencv_w2c(p,y,r,X_,Y_,Z_))
        all_w2cs.append(cur_w2c)
    return np.stack(all_w2cs)

def read_cam_csv(pose_file):
    cam_params = []
    with open(pose_file, 'r') as f:
        next(f)  # Skip header
        for line in f:
            frame_data = line.strip().split(',')
            if len(frame_data) != 3:
                continue
            
            # Parse position
            pos = frame_data[1].split()
            X = float(pos[0].split('=')[1])
            Y = float(pos[1].split('=')[1])
            Z = float(pos[2].split('=')[1])
            
            
            # Parse rotation
            rot = frame_data[2].split()
            P = float(rot[0].split('=')[1])
            Yaw = float(rot[1].split('=')[1])
            R = float(rot[2].split('=')[1])
            
            cam_params.append((X, Y, Z, P, Yaw, R))
    return cam_params

def project_cv(
        points: np.ndarray,
        extrinsics: np.ndarray = None,
        intrinsics: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project 3D points to 2D following the OpenCV convention

    Args:
        points (np.ndarray): [..., N, 3] or [..., N, 4] 3D points to project, if the last
            dimension is 4, the points are assumed to be in homogeneous coordinates
        extrinsics (np.ndarray): [..., 4, 4] extrinsics matrix
        intrinsics (np.ndarray): [..., 3, 3] intrinsics matrix

    Returns:
        uv_coord (np.ndarray): [..., N, 2] uv coordinates, value ranging in [0, 1].
            The origin (0., 0.) is corresponding to the left & top
        linear_depth (np.ndarray): [..., N] linear depth
    """
    assert intrinsics is not None, "intrinsics matrix is required"
    if points.shape[-1] == 3:
        points = np.concatenate([points, np.ones_like(points[..., :1])], axis=-1)
    if extrinsics is not None:
        points = points @ extrinsics.swapaxes(-1, -2)
    points = points[..., :3] @ intrinsics.swapaxes(-1, -2)
    with no_warnings():
        uv_coord = points[..., :2] / points[..., 2:]
    linear_depth = points[..., 2]
    return uv_coord, linear_depth

def uv_to_pixel(
    uv: np.ndarray,
    width: Union[int, np.ndarray],
    height: Union[int, np.ndarray]
) -> np.ndarray:
    """
    Args:
        pixel (np.ndarray): [..., 2] pixel coordinrates defined in image space,  x range is (0, W - 1), y range is (0, H - 1)
        width (int | np.ndarray): [...] image width(s)
        height (int | np.ndarray): [...] image height(s)

    Returns:
        (np.ndarray): [..., 2] pixel coordinrates defined in uv space, the range is (0, 1)
    """
    pixel = uv * np.stack([width, height], axis=-1).astype(uv.dtype) - 0.5
    return pixel

def merge_panorama_image(width: int, height: int, distance_maps, pred_masks, extrinsics, intrinsics_, init_frame=None, init_mask=None, mode="overwrite"):
    # preprocess:
    N = len(intrinsics_)
    intrinsics = []
    H, W = distance_maps[0].shape[:2]
    for i in range(N):
        ss = intrinsics_[i].copy()
        ss[0,...] /= W
        ss[1,...] /= H
        intrinsics.append(ss)
    uv = image_uv(width=width, height=height)
    spherical_directions = spherical_uv_to_directions(uv)

    # Warp each view to the panorama
    panorama_log_distance_grad_maps, panorama_grad_masks = [], []
    panorama_log_distance_laplacian_maps, panorama_laplacian_masks = [], []
    panorama_pred_masks = []
    
    panorama_log_distance_map = init_frame
    panorama_pred_mask = init_mask
    for i in range(len(distance_maps)):
        projected_uv, projected_depth = project_cv(spherical_directions, extrinsics=extrinsics[i], intrinsics=intrinsics[i])
        # print(projected_uv.shape, projected_depth.shape)
        projection_valid_mask = (projected_depth >= 0) & (projected_uv >= 0).all(axis=-1) & (projected_uv <= 1).all(axis=-1)        
        projected_pixels = uv_to_pixel(np.clip(projected_uv, 0, 1), width=distance_maps[i].shape[1], height=distance_maps[i].shape[0]).astype(np.float32)

        panorama_log_distance_map_r = np.where(projection_valid_mask, cv2.remap(distance_maps[i][...,0], projected_pixels[..., 0], projected_pixels[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE), 0)
        panorama_log_distance_map_g = np.where(projection_valid_mask, cv2.remap(distance_maps[i][...,1], projected_pixels[..., 0], projected_pixels[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE), 0)
        panorama_log_distance_map_b = np.where(projection_valid_mask, cv2.remap(distance_maps[i][...,2], projected_pixels[..., 0], projected_pixels[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE), 0)
        
        panorama_log_distance_map_ = np.stack([panorama_log_distance_map_r, panorama_log_distance_map_g, panorama_log_distance_map_b], axis=-1)
        panorama_pred_mask_ = projection_valid_mask & (cv2.remap(pred_masks[i].astype(np.uint8), projected_pixels[..., 0], projected_pixels[..., 1], cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE) > 0)
        # print(panorama_log_distance_map.shape, panorama_pred_mask.shape)
        # print(panorama_pred_mask_.dtype, panorama_pred_mask_.shape)
        # cv2.imwrite("./sdasd.png", panorama_pred_mask_.astype(np.uint8)*255)
        if panorama_log_distance_map is None:
            panorama_log_distance_map = panorama_log_distance_map_
            panorama_pred_mask = panorama_pred_mask_
        else:
            if mode=="overwrite":
                panorama_log_distance_map[panorama_pred_mask_] = panorama_log_distance_map_[panorama_pred_mask_]
            else:
                uu = panorama_pred_mask_ * panorama_pred_mask
                uu1 = (~panorama_pred_mask_) * panorama_pred_mask
                uu2 = (panorama_pred_mask_) * (~panorama_pred_mask)
                panorama_log_distance_map[uu] = (panorama_log_distance_map_[uu] + panorama_log_distance_map[uu])/2.
                panorama_log_distance_map[uu2] = (panorama_log_distance_map_[uu2])
            panorama_pred_mask += panorama_pred_mask_
        
    return panorama_log_distance_map, panorama_pred_mask

def merge_panorama_depth(width: int, height: int, distance_maps, pred_masks, extrinsics, intrinsics_, init_frame=None, init_mask=None, mode="overwrite"):
    # preprocess:
    N = len(intrinsics_)
    intrinsics = []
    H, W = distance_maps[0].shape[:2]
    for i in range(N):
        ss = intrinsics_[i].copy()
        ss[0,...] /= W
        ss[1,...] /= H
        intrinsics.append(ss)
    uv = image_uv(width=width, height=height)
    spherical_directions = spherical_uv_to_directions(uv)

    # Warp each view to the panorama
    panorama_log_distance_grad_maps, panorama_grad_masks = [], []
    panorama_log_distance_laplacian_maps, panorama_laplacian_masks = [], []
    panorama_pred_masks = []
    
    panorama_log_distance_map = init_frame
    panorama_pred_mask = init_mask
    for i in range(len(distance_maps)):
        projected_uv, projected_depth = project_cv(spherical_directions, extrinsics=extrinsics[i], intrinsics=intrinsics[i])
        # print(projected_uv.shape, projected_depth.shape)
        projection_valid_mask = (projected_depth >= 0) & (projected_uv >= 0).all(axis=-1) & (projected_uv <= 1).all(axis=-1)        
        projected_pixels = uv_to_pixel(np.clip(projected_uv, 0, 1), width=distance_maps[i].shape[1], height=distance_maps[i].shape[0]).astype(np.float32)
        #torch.where(condition, input, other, *, out=None)
        panorama_log_distance_map_ = np.where(projection_valid_mask, cv2.remap(distance_maps[i], projected_pixels[..., 0], projected_pixels[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE), 0)

        
        panorama_pred_mask_ = projection_valid_mask & (cv2.remap(pred_masks[i].astype(np.uint8), projected_pixels[..., 0], projected_pixels[..., 1], cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE) > 0)
        # print(panorama_log_distance_map.shape, panorama_pred_mask.shape)
        # print(panorama_pred_mask_.dtype, panorama_pred_mask_.shape)
        # cv2.imwrite("./sdasd.png", panorama_pred_mask_.astype(np.uint8)*255)
        if panorama_log_distance_map is None:
            panorama_log_distance_map = panorama_log_distance_map_
            panorama_pred_mask = panorama_pred_mask_
        else:

            panorama_log_distance_map[panorama_pred_mask_] = panorama_log_distance_map_[panorama_pred_mask_]
            panorama_pred_mask[panorama_pred_mask_] = True
        
    return panorama_log_distance_map, panorama_pred_mask


def project_cv_torch(points, extrinsics, intrinsics):
    #points: [H,W,3]
    #extrinsics: [4,4]
    #intrinsics: [3,3]
    eps = 1e-5
    points_cam = points @ (extrinsics[:3,:3].T)[None] + extrinsics[:3,3][None,None]
    points_pix = points_cam @ (intrinsics.T)[None]
    tmp = points_pix[:,:,2:].clone()
    tmp_mask = torch.abs(points_pix[:,:,2:]) < eps
    tmp[tmp_mask] = eps
    points_pix = points_pix / tmp
    return points_pix, tmp[...,0]
def merge_panorama_depth_torch(width: int, height: int, distance_maps, pred_masks, extrinsics, intrinsics_, device):
    # preprocess:
    N = len(intrinsics_)
    intrinsics = intrinsics_.clone()
    H, W = distance_maps[0].shape[:2]
    intrinsics[:,0,:] /= W
    intrinsics[:,1,:] /= H

    uv = image_uv_torch(width=width, height=height,device=device,dtype=torch.float32)
    spherical_directions = spherical_uv_to_directions_torch(uv)

    # Warp each view to the panorama
    panorama_log_distance_grad_maps, panorama_grad_masks = [], []
    panorama_log_distance_laplacian_maps, panorama_laplacian_masks = [], []
    panorama_pred_masks = []
    
    panorama_log_distance_map = None

    for i in range(len(distance_maps)):
        projected_uv, projected_depth = project_cv_torch(spherical_directions, extrinsics=extrinsics[i], intrinsics=intrinsics[i])
        # print(projected_uv.shape, projected_depth.shape)
        projection_valid_mask = (projected_depth >= 1e-5) & (projected_uv >= 0).all(dim=-1) & (projected_uv <= 1).all(dim=-1)        
        projected_pixels = (torch.clip((projected_uv * 2. - 1.),-1.,1.))
        #torch.where(condition, input, other, *, out=None)
        panorama_log_distance_map_ = torch.where(projection_valid_mask, torch.nn.functional.grid_sample(distance_maps[i][None,None], projected_pixels[None,:,:,:2], align_corners=True), 0).squeeze()


        if panorama_log_distance_map is None:
            panorama_log_distance_map = panorama_log_distance_map_
        else:

            panorama_log_distance_map[projection_valid_mask] = panorama_log_distance_map_[projection_valid_mask]
        
    return panorama_log_distance_map

def merge_panorama_color_torch(width: int, height: int, distance_maps, pred_masks, extrinsics, intrinsics_, device):
    # preprocess:
    N = len(intrinsics_)
    intrinsics = intrinsics_.clone()
    H, W = distance_maps[0].shape[:2]
    intrinsics[:,0,:] /= W
    intrinsics[:,1,:] /= H

    uv = image_uv_torch(width=width, height=height,device=device,dtype=torch.float32)
    spherical_directions = spherical_uv_to_directions_torch(uv)

    # Warp each view to the panorama
    panorama_log_distance_grad_maps, panorama_grad_masks = [], []
    panorama_log_distance_laplacian_maps, panorama_laplacian_masks = [], []
    panorama_pred_masks = []
    
    panorama_log_distance_map = None

    for i in range(len(distance_maps)):
        projected_uv, projected_depth = project_cv_torch(spherical_directions, extrinsics=extrinsics[i], intrinsics=intrinsics[i])
        # print(projected_uv.shape, projected_depth.shape)
        projection_valid_mask = (projected_depth >= 1e-5) & (projected_uv >= 0).all(dim=-1) & (projected_uv <= 1).all(dim=-1)        
        projected_pixels = (torch.clip((projected_uv * 2. - 1.),-1.,1.))
        #torch.where(condition, input, other, *, out=None)
        panorama_log_distance_map_ = torch.where(projection_valid_mask, torch.nn.functional.grid_sample(distance_maps[i][None].permute(0,3,1,2), projected_pixels[None,:,:,:2], align_corners=True), 0).squeeze().permute(1,2,0)


        if panorama_log_distance_map is None:
            panorama_log_distance_map = panorama_log_distance_map_
        else:

            panorama_log_distance_map[projection_valid_mask] = panorama_log_distance_map_[projection_valid_mask]
        
    return panorama_log_distance_map
def mesh_pano_render(mesh, Rts, height, width, near, far, device):
    # the rotational matrix for turning render views;
    #print(f"dsadsad:{Rts.shape}")
    mat = np.array(
        [
            [
                [1,0,0],
                [0,1,0],
                [0,0,1]
            ],
            [
                [0,0,-1],
                [0,1,0],
                [1,0,0]
            ],
            [
                [-1,0,0],
                [0,1,0],
                [0,0,-1]
            ],
            [
                [0,0,1],
                [0,1,0],
                [-1,0,0]
            ],
            [
                [1,0,0],
                [0,0,1],
                [0,-1,0]
            ],
            [
                [1,0,0],
                [0,0,-1],
                [0,1,0]
            ],
        ],
        dtype=np.float32
    )
    #the extrinsic matrix for gluing images to panorama;
    mat_glue = np.array(
        [
            [
                [0,0,-1],
                [0,-1,0],
                [-1,0,0]
            ],
            [
                [0,0,-1],
                [-1,0,0],
                [0,1,0]
            ],
            [
                [0,0,-1],
                [0,1,0],
                [1,0,0]
            ],
            [
                [0,0,-1],
                [1,0,0],
                [0,-1,0]
            ],
            [
                [-1,0,0],
                [0,-1,0],
                [0,0,1]
            ],
            [
                [1,0,0],
                [0,-1,0],
                [0,0,-1]
            ],
        ],
        dtype=np.float32
    )
    mat_glue = np.array([
        [
            [0,-1,0],
            [1,0,0],
            [0,0,1],
        ]
    ]) @ mat_glue
    tmp = np.eye(4).astype(np.float32)[None].repeat(6,0)
    tmp[:,:3,:3] = mat_glue

    K = np.array([
        [256.,0,256.],
        [0.,256,256.],
        [0.,0,1.],
    ]).astype(np.float32)

    all_pano_depth_map = []
    all_pano_mask_map = []
    N = Rts.shape[0]

    render_batch_size = 5
    for i in range(0,N,render_batch_size):

        cur_process_size = min(render_batch_size,N-i)
        cur_Rt = Rts[i:i+cur_process_size][:,None,:,:]
        panorama_render_Rts = cur_Rt.repeat(6,1)
        panorama_render_Rts[:,:,:3,:] = mat[None] @ panorama_render_Rts[:,:,:3,:]
        render_depth,render_mask = get_mesh_render_depth(mesh, (K[None].repeat(6*cur_process_size,0)), (panorama_render_Rts.reshape(cur_process_size*6,4,4)), 512,512,near,far, device)
        #print(render_depth.max(), render_depth.min())
        render_depth[~render_mask] = torch.inf

        for j in range(cur_process_size):
            panorama_log_distance_map = merge_panorama_depth_torch(width,height,torch.from_numpy(render_depth[6*j:6*(j+1)]).to(device),None,torch.from_numpy(tmp).to(device), torch.from_numpy(K[None].repeat(6*render_batch_size,0)).to(device), device)
            # print(f"panorama check: {panorama_log_distance_map.max()} {panorama_log_distance_map.min()}")
            all_pano_depth_map.append(panorama_log_distance_map.cpu().numpy())
            all_pano_mask_map.append((torch.ones_like(panorama_log_distance_map).bool()).cpu().numpy())
        
    return all_pano_depth_map, all_pano_mask_map


def mesh_pano_render_color(mesh, Rts, height, width, near, far, device):
    # the rotational matrix for turning render views;
    #print(f"dsadsad:{Rts.shape}")
    mat = np.array(
        [
            [
                [1,0,0],
                [0,1,0],
                [0,0,1]
            ],
            [
                [0,0,-1],
                [0,1,0],
                [1,0,0]
            ],
            [
                [-1,0,0],
                [0,1,0],
                [0,0,-1]
            ],
            [
                [0,0,1],
                [0,1,0],
                [-1,0,0]
            ],
            [
                [1,0,0],
                [0,0,1],
                [0,-1,0]
            ],
            [
                [1,0,0],
                [0,0,-1],
                [0,1,0]
            ],
        ],
        dtype=np.float32
    )
    #the extrinsic matrix for gluing images to panorama;
    mat_glue = np.array(
        [
            [
                [0,0,-1],
                [0,-1,0],
                [-1,0,0]
            ],
            [
                [0,0,-1],
                [-1,0,0],
                [0,1,0]
            ],
            [
                [0,0,-1],
                [0,1,0],
                [1,0,0]
            ],
            [
                [0,0,-1],
                [1,0,0],
                [0,-1,0]
            ],
            [
                [-1,0,0],
                [0,-1,0],
                [0,0,1]
            ],
            [
                [1,0,0],
                [0,-1,0],
                [0,0,-1]
            ],
        ],
        dtype=np.float32
    )
    mat_glue = np.array([
        [
            [0,-1,0],
            [1,0,0],
            [0,0,1],
        ]
    ]) @ mat_glue
    tmp = np.eye(4).astype(np.float32)[None].repeat(6,0)
    tmp[:,:3,:3] = mat_glue

    K = np.array([
        [256.,0,256.],
        [0.,256,256.],
        [0.,0,1.],
    ]).astype(np.float32)

    all_pano_depth_map = []
    all_pano_mask_map = []
    N = Rts.shape[0]

    # Per-frame progress: drives the node's ComfyUI progress bar AND a console
    # line per batch. Best-effort -- this vendored renderer must still import and
    # run when ComfyUI's ProgressBar isn't available (e.g. standalone use).
    _pbar = None
    try:
        from comfy.utils import ProgressBar
        _pbar = ProgressBar(N)
    except Exception:
        _pbar = None

    # Loop-invariant uploads, hoisted: the glue extrinsics and intrinsics were
    # re-uploaded host->device twice per frame inside the merge loop below.
    # merge_panorama_color_torch only ever reads entries 0..5 of the intrinsics.
    glue_Rts_t = torch.from_numpy(tmp).to(device)
    K_t = torch.from_numpy(K[None].repeat(6,0)).to(device)

    render_batch_size = 5
    for i in range(0,N,render_batch_size):

        cur_process_size = min(render_batch_size,N-i)
        cur_Rt = Rts[i:i+cur_process_size][:,None,:,:]
        panorama_render_Rts = cur_Rt.repeat(6,1)
        panorama_render_Rts[:,:,:3,:] = mat[None] @ panorama_render_Rts[:,:,:3,:]
        render_color,render_mask = get_mesh_render_color(mesh, (K[None].repeat(6*cur_process_size,0)), (panorama_render_Rts.reshape(cur_process_size*6,4,4)), 512,512,near,far, device)
        if isinstance(render_color, np.ndarray):
            # The vendored renderer returns host arrays; the injected fast one
            # (matrix3d_pipeline._install_fast_color_render) returns device
            # tensors directly, skipping a GPU->CPU->GPU round-trip of the
            # whole batch. Merge consumes device tensors either way.
            render_color = torch.from_numpy(render_color).to(device)
            render_mask = torch.from_numpy(render_mask).to(device)
        render_color[~render_mask] = torch.inf

        for j in range(cur_process_size):
            panorama_log_distance_map = merge_panorama_color_torch(width,height,render_color[6*j:6*(j+1)],None,glue_Rts_t, K_t, device)
            msk = panorama_log_distance_map<torch.inf
            panorama_log_distance_map[panorama_log_distance_map==torch.inf] = 0.
            all_pano_depth_map.append(panorama_log_distance_map.cpu().numpy())
            all_pano_mask_map.append(msk[:,:,0].cpu().numpy())

        done = min(i + cur_process_size, N)
        print(f"[CameraPlot] rendered {done}/{N} frames", flush=True)
        if _pbar is not None:
            _pbar.update_absolute(done, N)

    return all_pano_depth_map, all_pano_mask_map

def generate_mask_video_projection_torch_simulate(csv_cams, depths,first_frame_rgb, skybox_masks = None, apply_boundary_occulusion=True):
    device = depths.device
    H, W = depths.shape[1:3]
    frame_size = len(csv_cams)
    # 1. generate first frame deoth edge mask;
    edge_mask_first_frame = ~(depth_edge_torch(depths[0],rtol=0.05))

    rot_matrix = torch.tensor([
        [0,1,0],[0,0,-1],[-1,0,0.]
    ], dtype=depths.dtype,device=device)

    # print(mask.sum())
    all_w2cs = []
    for i in range(frame_size):
        X_,Y_,Z_,p,y,r = csv_cams[i]
        # print(f"union check: {X_} {Y_} {Z_} {frame_size}")
        cur_w2c = unreal_to_opencv_w2c(p,y,r,X_,Y_,Z_)
        all_w2cs.append(cur_w2c)

    # raise ValueError("ssad")
    all_w2cs = torch.from_numpy(np.stack(all_w2cs)).to(device).to(depths.dtype)
    all_c2ws = torch.linalg.inv(all_w2cs)
    # print(f"check: {all_w2cs[:10]}")
    
    all_canvas_mask = torch.zeros((frame_size, H, W), dtype=torch.bool, device=device)
    all_canvas_rgb = torch.zeros((frame_size, H, W, 3), dtype=torch.float32, device=device)
    if apply_boundary_occulusion:
        mesh_occulusion = get_mesh_from_pano(depths[0].cpu().numpy(),(~edge_mask_first_frame).cpu().numpy(), csv_cams[0])
        near = 1e-3
        far = depths[i].max() * 1.1
        rendered_depth_, rendered_mask_ = mesh_pano_render(mesh_occulusion, all_w2cs[1:].cpu().numpy(), H, W, near, far, device)
        # debug
        

    for i in range(1,frame_size):
        if apply_boundary_occulusion:
            rendered_depth = rendered_depth_[i-1].reshape(-1)
            rendered_depth = torch.from_numpy(rendered_depth).float().to(device)

        #rendered_mask = torch.from_numpy(rendered_mask).bool().to(device)

        wrd_pcs = get_world_pcs_pano_torch(csv_cams[i], depths[i])

        cur_pcs_cam = wrd_pcs @ all_w2cs[0,:3,:3].T + all_w2cs[0,:3,3][None]
        pcs_cur_c2w_sph = cur_pcs_cam @ rot_matrix

        pcs_cur_c2w_norm = torch.linalg.norm(pcs_cur_c2w_sph, dim=-1)
        pcs_cur_c2w_direction = pcs_cur_c2w_sph / (pcs_cur_c2w_norm[...,None] + 1e-8)

        pcs_cur_c2w_uv = directions_to_spherical_uv_torch(pcs_cur_c2w_direction)
        pcs_cur_c2w_pix = uv_to_pixel_torch(pcs_cur_c2w_uv, width=W, height=H).to(torch.float32)
        
        pcs_cur_c2w_pix_normalized = pcs_cur_c2w_pix.clone()
        pcs_cur_c2w_pix_normalized[...,1] = pcs_cur_c2w_pix[...,1]/H * 2. - 1.
        pcs_cur_c2w_pix_normalized[...,0] = pcs_cur_c2w_pix[...,0]/W * 2. - 1.
        #print(f"range check: ", pcs_cur_c2w_pix_normalized.max(), pcs_cur_c2w_pix_normalized.min())
        sampled_depth_value = torch.nn.functional.grid_sample(depths[0][None,None], pcs_cur_c2w_pix_normalized[None,None], mode='bilinear', padding_mode='zeros', align_corners=True).reshape(-1)
        sampled_rgb_value = torch.nn.functional.grid_sample(first_frame_rgb[None].permute(0,3,1,2), pcs_cur_c2w_pix_normalized[None,None], mode='bilinear', padding_mode='zeros', align_corners=True).squeeze().permute(1,0)
        sampled_mask_value = torch.nn.functional.grid_sample(edge_mask_first_frame[None,None].float(), pcs_cur_c2w_pix_normalized[None,None], mode='nearest', padding_mode='zeros', align_corners=True).reshape(-1)
        
        mask_pixvalid = (pcs_cur_c2w_pix_normalized[...,0] >=-1.) * (pcs_cur_c2w_pix_normalized[...,1] >=-1.) * (pcs_cur_c2w_pix_normalized[...,0] <=1.) * (pcs_cur_c2w_pix_normalized[...,1] <=1.)        
        
        mask_depth_value = (sampled_depth_value * 1.02 >  pcs_cur_c2w_norm)
        mask_mask_value = sampled_mask_value == 1.


        
        mask_valid = mask_pixvalid * mask_depth_value * mask_mask_value
        if skybox_masks is not None:
            #print("sad")
            sampled_skybox_value = torch.nn.functional.grid_sample(skybox_masks[0][None,None].float(), pcs_cur_c2w_pix_normalized[None,None], mode='nearest', padding_mode='zeros', align_corners=True).reshape(-1) == 1.
            mask_skybox_value = (sampled_skybox_value * skybox_masks[i].reshape(-1)) * mask_pixvalid * mask_mask_value
            mask_valid[mask_skybox_value] = True
            if apply_boundary_occulusion:
                mask_valid = mask_valid * (depths[i].reshape(-1) < 1.1 * rendered_depth)
        
        mask_valid_reshape = mask_valid.reshape(H,W)
        sampled_rgb_valid = sampled_rgb_value[mask_valid]

        all_canvas_rgb[i][mask_valid_reshape] = sampled_rgb_valid
        all_canvas_mask[i][mask_valid_reshape] = True

    all_canvas_rgb[0] = first_frame_rgb
    all_canvas_mask[0,...] = True 
    return all_canvas_rgb, all_canvas_mask

def find_mask_cc(mask):
    canvas = np.zeros_like(mask).astype(np.int32)
    canvas[mask] = 1
    labels, numbers = ski.measure.label(mask, background=None, return_num=True, connectivity=2)
    return labels, numbers
def depth_repair(depth, small_area_threshold=1600):
    max_depth = depth.max()
    mask_valid = depth < 0.99 * max_depth

    labels, numbers = find_mask_cc(mask_valid)
    for i in range(numbers):
        cur_mask = labels == i+1
        if cur_mask.sum() < small_area_threshold:
            depth[cur_mask] = max_depth
    return depth
def generate_pc_render(Rts,first_frame_rgb, first_frame_depth, first_frame_Rt=None):
    device = first_frame_depth.device
    H, W = first_frame_depth.shape[:2]
    frame_size = len(Rts)
    # 1. generate first frame deoth edge mask;
    edge_mask_first_frame = ~(depth_edge_torch(first_frame_depth,rtol=0.05))

    rot_matrix = torch.tensor([
        [0,1,0],[0,0,-1],[-1,0,0.]
    ], dtype=first_frame_depth.dtype,device=device)



    # raise ValueError("ssad")
    all_w2cs = Rts
    all_c2ws = torch.linalg.inv(all_w2cs)
    # print(f"check: {all_w2cs[:10]}")
    
    # mesh_pano_render_color already returns the merged frames as host (numpy)
    # arrays, and the only consumer (render_control) immediately copies the result
    # back to the host. Assembling the canvas on the GPU therefore did a pointless
    # round-trip -- 81 host->device uploads of the full video plus a multi-GB device
    # buffer -- only for that buffer to be downloaded again. Build on the host
    # instead. Set P2S_CANVAS_GPU=1 to restore the original device buffer.
    _canvas_dev = device if os.environ.get("P2S_CANVAS_GPU", "0") == "1" else "cpu"
    all_canvas_mask = torch.zeros((frame_size, H, W), dtype=torch.bool, device=_canvas_dev)
    all_canvas_rgb = torch.zeros((frame_size, H, W, 3), dtype=torch.float32, device=_canvas_dev)
    #if apply_boundary_occulusion:
    #mesh_occulusion = get_mesh_from_pano(first_frame_depth.cpu().numpy(),(~edge_mask_first_frame).cpu().numpy(), csv_cams[0])
    #get_mesh_from_pano_Rt(pano_depth, pano_mask, Rt, dtype=np.float32, pano_rgb=None, alpha_mask=None):
    '''mesh_occulusion = get_mesh_from_pano_Rt(first_frame_depth.cpu().numpy(),(~edge_mask_first_frame).cpu().numpy(), Rts[0])
    near = 1e-3
    far = first_frame_depth.max() * 1.1
    rendered_depth, rendered_occulusion_mask = mesh_pano_render(mesh_occulusion, all_w2cs.cpu().numpy(), H, W, near, far, device)'''
    if first_frame_Rt is None:
        first_frame_Rt = Rts[0]
    mesh_full = get_mesh_from_pano_Rt(first_frame_depth.cpu().numpy(),np.ones((H,W),dtype=np.bool_), first_frame_Rt.cpu().numpy(), pano_rgb=first_frame_rgb.cpu().numpy(), alpha_mask=(edge_mask_first_frame).cpu().numpy())
    #mesh_full = get_mesh_from_pano_Rt(first_frame_depth.cpu().numpy(),(edge_mask_first_frame).cpu().numpy(), Rts[0].cpu().numpy(), pano_rgb=first_frame_rgb.cpu().numpy())
    near = 1e-3
    far = first_frame_depth.max() * 1.1

    print("mesh_pano_render_color")
    torch.cuda.empty_cache()
    print(torch.cuda.memory_allocated() / 1024**3, "GB used")  
    print(torch.cuda.memory_reserved() / 1024**3, "GB reserved")  
    
    # import pdb
    # pdb.set_trace()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.memory_summary()
    rendered_cmap, rendered_cmap_mask = mesh_pano_render_color(mesh_full, all_w2cs.cpu().numpy(), H, W, near, far, device)
    # debug
        

    for i in range(frame_size):
        cmap_mask = rendered_cmap_mask[i]
        cmap = rendered_cmap[i]
        cmap[~cmap_mask] = 0.

        all_canvas_rgb[i] = torch.from_numpy(cmap).float().to(_canvas_dev)
        all_canvas_mask[i] = torch.from_numpy(cmap_mask).float().to(_canvas_dev)

    all_canvas_rgb[0] = first_frame_rgb.to(_canvas_dev)
    all_canvas_mask[0,...] = True
    return all_canvas_rgb, all_canvas_mask

def perform_camera_movement(firstframe_rgb, firstframe_depth, angle=0., movement_ratio = 0.5, frame_size=81):
    device = firstframe_rgb.device
    depth_np = firstframe_depth.cpu().numpy()
    depth_np = depth_repair(depth_np)
    firstframe_depth = torch.from_numpy(depth_np).to(device)
    
    H, W = firstframe_rgb.shape[:2]
    pvt = int(angle/360. * W + W//2) % W
    firstframe_rgb = torch.cat([firstframe_rgb[:,pvt:],firstframe_rgb[:,:pvt]], dim=1)
    firstframe_rgb = torch.cat([firstframe_rgb[:,W//2:],firstframe_rgb[:,:W//2]], dim=1)
    firstframe_depth = torch.cat([firstframe_depth[:,pvt:],firstframe_depth[:,:pvt]], dim=1)
    firstframe_depth = torch.cat([firstframe_depth[:,W//2:],firstframe_depth[:,:W//2]], dim=1)

    depth_pvt = firstframe_depth[H//2,int(W//2*1.05)]
    Rt0 = torch.tensor(
        [
            [1.,0,0,0],
            [0,1.,0,0],
            [0,0,1,0],
            [0,0,0,1]
        ],
        dtype=torch.float32,
        device=device
    )

    Rts = Rt0[None].repeat(frame_size,1,1)
    for i in range(frame_size):
        Rts[i,2,3] = -(depth_pvt * i *movement_ratio)/(frame_size-1)
    rgbs, masks = generate_pc_render(Rts, firstframe_rgb, firstframe_depth)
    return rgbs, masks, Rts, firstframe_rgb, firstframe_depth
def generate_rail(depth, angle, movement_ratio, frame_size, mode="straight"):
    print("generating rail...")
    device=depth.device
    H, W = depth.shape[:2]
    q = int(angle/360. * W + W//2)%W
    depth_pvt = depth[int(H//2*1.05),q]
    z_axis_ = torch.tensor([0,0,1.]).float().to(device)
    x_axis_ = torch.tensor([1,0,0.]).float().to(device)
    y_axis = torch.tensor([0,1,0]).float().to(device)

    z_axis = x_axis_ * math.sin(math.radians(angle)) + z_axis_ * math.cos(math.radians(angle))
    x_axis = z_axis_ * -math.sin(math.radians(angle)) + x_axis_ * math.cos(math.radians(angle))
    print(z_axis, x_axis)
    if mode == "straight":
        Rt0_ = torch.stack([x_axis,y_axis,z_axis],dim=0)
        Rt0 = torch.eye(4).float().to(device)
        Rt0[:3,:3] = Rt0_
        Rts = Rt0[None].repeat(frame_size,1,1)
        for i in range(frame_size):
            Rts[i,2,3] = -(depth_pvt * i *movement_ratio)/(frame_size-1)
    if mode == "s_curve":
        l = depth_pvt * movement_ratio
        sin_ratio = 0.15
        all_Rts = []
        for i in range(frame_size):
            cur_t = (depth_pvt * i *movement_ratio)/(frame_size-1)
            pos = z_axis * (depth_pvt * i *movement_ratio)/(frame_size-1) + x_axis * math.sin(cur_t / l * math.pi * 2) * sin_ratio
            direction_ = z_axis + sin_ratio * (2 * math.pi / l) * math.cos(math.pi * 2 / l * cur_t) * x_axis
            direction = direction_/direction_.norm()
            cur_z_axis = direction
            cur_y_axis = y_axis
            cur_x_axis = torch.cross(cur_y_axis, cur_z_axis)
            cur_c2w = torch.eye(4).float().to(device)
            cur_c2w[:3,:3] = torch.stack([cur_x_axis,y_axis,cur_z_axis],dim=1)
            cur_c2w[:3,3] = pos
            all_Rts.append(torch.linalg.inv(cur_c2w))
        Rts = torch.stack(all_Rts,axis=0)
    if mode == "r_curve":
        print("r curve")
        curve_deg = 30.
        l = depth_pvt * movement_ratio
        r = float(l/math.sin(math.radians(curve_deg)))
        c = torch.tensor([r,0,0],dtype=torch.float32,device=device)

        #sin_ratio = 0.15
        all_Rts = []
        for i in range(frame_size):
            theta = curve_deg * float(i)/(frame_size-1)
            v = r * (-x_axis_ * math.cos(math.radians(theta)) + z_axis_ * math.sin(math.radians(theta)))
            pos = c + v
            z_axis = (x_axis_ * math.sin(math.radians(theta)) + z_axis_ * math.cos(math.radians(theta)))
            x_axis = -(-x_axis_ * math.cos(math.radians(theta)) + z_axis_ * math.sin(math.radians(theta)))
            cur_c2w = torch.eye(4).float().to(device)
            cur_c2w[:3,:3] = torch.stack([x_axis,y_axis,z_axis],dim=1)
            cur_c2w[:3,3] = pos
            all_Rts.append(torch.linalg.inv(cur_c2w))
        Rts = torch.stack(all_Rts,axis=0)
    if mode == "l_curve":
        print("l curve")
        curve_deg = 30.
        l = depth_pvt * movement_ratio
        r = float(l/math.sin(math.radians(curve_deg)))
        c = torch.tensor([-r,0,0],dtype=torch.float32,device=device)

        #sin_ratio = 0.15
        all_Rts = []
        for i in range(frame_size):
            theta = curve_deg * float(i)/(frame_size-1)
            v = r * (x_axis_ * math.cos(math.radians(theta)) + z_axis_ * math.sin(math.radians(theta)))
            pos = c + v
            z_axis = (-x_axis_ * math.sin(math.radians(theta)) + z_axis_ * math.cos(math.radians(theta)))
            x_axis = (x_axis_ * math.cos(math.radians(theta)) + z_axis_ * math.sin(math.radians(theta)))

            z_axis[0] = -z_axis[0]
            x_axis[0] = -x_axis[0]
            pos[0] = -pos[0]
            cur_c2w = torch.eye(4).float().to(device)
            cur_c2w[:3,:3] = torch.stack([x_axis,y_axis,z_axis],dim=1)
            cur_c2w[:3,3] = pos
            all_Rts.append(torch.linalg.inv(cur_c2w))
        Rts = torch.stack(all_Rts,axis=0)
    if mode == "coverage":
        # Pano2World: forward progress + a wide lateral sweep with the camera held facing
        # forward. The lateral translation is parallax that sees AROUND objects (fills
        # disocclusions, fewer black holes) and widens the reconstructed bubble into a 2D
        # footprint -- all as ONE coherent forward path (no cross-video hallucination
        # mismatch). amp ~ a quarter of the forward reach; 1.5 periods = left/right/left.
        l = depth_pvt * movement_ratio
        amp = l * 0.25
        periods = 1.5
        all_Rts = []
        for i in range(frame_size):
            t = i / (frame_size - 1)
            pos = z_axis * (l * t) + x_axis * (amp * math.sin(periods * 2 * math.pi * t))
            cur_c2w = torch.eye(4).float().to(device)
            cur_c2w[:3, :3] = torch.stack([x_axis, y_axis, z_axis], dim=1)
            cur_c2w[:3, 3] = pos
            all_Rts.append(torch.linalg.inv(cur_c2w))
        Rts = torch.stack(all_Rts, axis=0)
    if mode == "coverage_back":
        # Pano2World: like 'coverage' but travelling backward (toward the rear of the scene)
        # with the same lateral sweep + fixed forward-facing orientation. SAME angle-0 world
        # frame as 'coverage', so the two fuse directly with no re-alignment.
        l = depth_pvt * movement_ratio
        amp = l * 0.25
        periods = 1.5
        all_Rts = []
        for i in range(frame_size):
            t = i / (frame_size - 1)
            pos = z_axis * (-l * t) + x_axis * (amp * math.sin(periods * 2 * math.pi * t))
            cur_c2w = torch.eye(4).float().to(device)
            cur_c2w[:3, :3] = torch.stack([x_axis, y_axis, z_axis], dim=1)
            cur_c2w[:3, 3] = pos
            all_Rts.append(torch.linalg.inv(cur_c2w))
        Rts = torch.stack(all_Rts, axis=0)
    # --- multi-trajectory FUSION s-curves (Pano2World 14b-fusion idea) -------------
    # Two "back-and-forth S-curve" modes meant to be RENDERED + WAN-generated
    # SEPARATELY and then FUSED into one dataset (see make_equirect_dataset_fused):
    #   bf_forward -- s-curve back-and-forth along the FORWARD (view) axis
    #   bf_lateral -- the same, but turned 90 degrees onto the OTHER (lateral) axis
    # Both share the SAME angle-0 world frame and start at the identity pose
    # (frame 0 == origin), so their depths + cameras concatenate with NO re-alignment.
    #
    # The motion is a genuine 3D S-CURVE, not a straight dolly:
    #   * MAIN axis  : back AND forth  -- a full sine, 0 -> +reach -> 0 -> -reach -> 0,
    #                  so both sides of the start point are covered (reach = l).
    #   * WEAVE axis : the perpendicular horizontal axis swings at 2x the main rate
    #                  (sin(4*pi*t)) -> the path SNAKES (this is the "S"), widening the
    #                  parallax baseline into 2D.
    #   * UP/DOWN    : the camera also bobs up and down ONCE over the trajectory
    #                  (sin(2*pi*t) on Y), amplitude scaled by movement_ratio -- so the
    #                  s-curve sweep simultaneously rises and dips (a 3D snake), instead
    #                  of needing a separate vertical pass.
    # Camera orientation is held FIXED (faces +Z): an EQUIRECTANGULAR frame already
    # sees all 360 degrees, so coverage comes from TRANSLATION (parallax that opens
    # disocclusions), not from rotating the camera -- and a shared fixed heading is what
    # lets the two trajectories fuse without ghosting (turning would only roll the pano).
    if mode in ("bf_forward", "bf_lateral"):
        l = depth_pvt * movement_ratio
        weave = l * 0.3                       # lateral S amplitude (the snake)
        bob = l * 0.3                         # vertical up/down amplitude (scales w/ range)
        main_ax = z_axis if mode == "bf_forward" else x_axis   # back-and-forth axis
        weave_ax = x_axis if mode == "bf_forward" else z_axis  # perpendicular snake axis
        all_Rts = []
        for i in range(frame_size):
            t = i / (frame_size - 1)
            pos = (main_ax * (l * math.sin(2 * math.pi * t))         # back and forth
                   + weave_ax * (weave * math.sin(2 * 2 * math.pi * t))  # S-snake
                   + y_axis * (bob * math.sin(2 * math.pi * t)))     # up/down once
            cur_c2w = torch.eye(4).float().to(device)
            cur_c2w[:3, :3] = torch.stack([x_axis, y_axis, z_axis], dim=1)
            cur_c2w[:3, 3] = pos
            all_Rts.append(torch.linalg.inv(cur_c2w))
        Rts = torch.stack(all_Rts, axis=0)
    # --- BRAVER "travel further & around" modes -----------------------------------
    # These push the camera much further / arc it around the scene so SfM gets a far
    # bigger parallax baseline (a bigger walkable bubble, fewer holes). All keep the
    # camera heading FIXED at +Z (an equirect frame already sees all 360 deg, so
    # coverage must come from TRANSLATION, not rotation) and start at the identity
    # pose (frame 0 == origin) in the angle-0 world frame -- so they remain
    # fusion-compatible with each other and with the bf_* sweeps.
    denom = max(frame_size - 1, 1)
    if mode == "orbit":
        # Single pass that ARCS partway around the scene: the camera rides a
        # horizontal circle of radius ~l centred on the pivot point in front, sweeping
        # ~90 deg of azimuth while still facing +Z. Huge lateral travel -> SfM sees the
        # scene from very different angles. Fusion-friendly (fixed heading, origin start).
        l = depth_pvt * movement_ratio
        arc_deg = 90.0
        center = z_axis * l                 # orbit centre: the pivot, l ahead of origin
        all_Rts = []
        for i in range(frame_size):
            t = i / denom
            th = math.radians(arc_deg * t)
            # radius vector from centre to camera (frame 0 == -z_axis*l => origin),
            # rotated by th about the up axis in the (x_axis, z_axis) plane.
            pos = center + x_axis * (-l * math.sin(th)) + z_axis * (-l * math.cos(th))
            cur_c2w = torch.eye(4).float().to(device)
            cur_c2w[:3, :3] = torch.stack([x_axis, y_axis, z_axis], dim=1)
            cur_c2w[:3, 3] = pos
            all_Rts.append(torch.linalg.inv(cur_c2w))
        Rts = torch.stack(all_Rts, axis=0)
    if mode == "push_in":
        # Dramatic "fly into the scene": a clean, strong forward dolly that travels the
        # FULL reach l with no lateral weave -- further/straighter than s_curve. Best as
        # a single hero pass; fixed +Z heading, origin start (so still fusable if wanted).
        l = depth_pvt * movement_ratio
        all_Rts = []
        for i in range(frame_size):
            t = i / denom
            pos = z_axis * (l * t)
            cur_c2w = torch.eye(4).float().to(device)
            cur_c2w[:3, :3] = torch.stack([x_axis, y_axis, z_axis], dim=1)
            cur_c2w[:3, 3] = pos
            all_Rts.append(torch.linalg.inv(cur_c2w))
        Rts = torch.stack(all_Rts, axis=0)
    if mode == "spiral":
        # Corkscrew: forward progress combined with a WIDENING helix on the lateral+up
        # axes (radius grows 0 -> ~0.35*l over ~1.5 turns) for maximum 3D parallax in one
        # pass. Fixed +Z heading, origin start.
        l = depth_pvt * movement_ratio
        amp = l * 0.35
        turns = 1.5
        all_Rts = []
        for i in range(frame_size):
            t = i / denom
            ang = turns * 2 * math.pi * t
            r = amp * t                      # widen the helix as we travel forward
            pos = (z_axis * (l * t)
                   + x_axis * (r * math.cos(ang))
                   + y_axis * (r * math.sin(ang)))
            cur_c2w = torch.eye(4).float().to(device)
            cur_c2w[:3, :3] = torch.stack([x_axis, y_axis, z_axis], dim=1)
            cur_c2w[:3, 3] = pos
            all_Rts.append(torch.linalg.inv(cur_c2w))
        Rts = torch.stack(all_Rts, axis=0)
    if mode == "bf_orbit":
        # FUSION-INTENDED 3rd/4th branch: a big back-and-forth that ARCS around the
        # scene. The azimuth swings a full sine th = arc*sin(2*pi*t) (origin -> left
        # arc -> origin -> right arc -> origin) on the orbit circle of radius l about the
        # pivot, plus a single up/down bob. Same fixed +Z heading / angle-0 frame /
        # origin start as bf_forward & bf_lateral, so it fuses with them directly.
        l = depth_pvt * movement_ratio
        arc_deg = 75.0
        bob = l * 0.3
        center = z_axis * l
        all_Rts = []
        for i in range(frame_size):
            t = i / denom
            th = math.radians(arc_deg) * math.sin(2 * math.pi * t)   # swing both ways
            pos = (center
                   + x_axis * (-l * math.sin(th))
                   + z_axis * (-l * math.cos(th))
                   + y_axis * (bob * math.sin(2 * math.pi * t)))      # up/down once
            cur_c2w = torch.eye(4).float().to(device)
            cur_c2w[:3, :3] = torch.stack([x_axis, y_axis, z_axis], dim=1)
            cur_c2w[:3, 3] = pos
            all_Rts.append(torch.linalg.inv(cur_c2w))
        Rts = torch.stack(all_Rts, axis=0)
    return Rts

        


def perform_camera_movement_new(firstframe_rgb, firstframe_depth, angle=0., movement_ratio = 0.5, frame_size=81, preset_rail=None,mode="s_curve"):
    '''device = firstframe_rgb.device
    depth_np = firstframe_depth.cpu().numpy()
    depth_np = depth_repair(depth_np)
    firstframe_depth = torch.from_numpy(depth_np).to(device)
    
    H, W = firstframe_rgb.shape[:2]
    pvt = int(angle/360. * W + W//2) % W
    firstframe_rgb = torch.cat([firstframe_rgb[:,pvt:],firstframe_rgb[:,:pvt]], dim=1)
    firstframe_rgb = torch.cat([firstframe_rgb[:,W//2:],firstframe_rgb[:,:W//2]], dim=1)
    firstframe_depth = torch.cat([firstframe_depth[:,pvt:],firstframe_depth[:,:pvt]], dim=1)
    firstframe_depth = torch.cat([firstframe_depth[:,W//2:],firstframe_depth[:,:W//2]], dim=1)

    depth_pvt = firstframe_depth[H//2,int(W//2*1.05)]
    Rt0 = torch.tensor(
        [
            [1.,0,0,0],
            [0,1.,0,0],
            [0,0,1,0],
            [0,0,0,1]
        ],
        dtype=torch.float32,
        device=device
    )

    Rts = Rt0[None].repeat(frame_size,1,1)'''
    device = firstframe_depth.device
    depth_np = firstframe_depth.cpu().numpy()
    depth_np = depth_repair(depth_np)
    W = firstframe_rgb.shape[1]
    firstframe_depth = torch.from_numpy(depth_np).to(device)
    q = int(angle/360. * W + W//2)%W
    firstframe_rgb = torch.cat([firstframe_rgb[:,q:],firstframe_rgb[:,:q]],dim=1)
    firstframe_depth = torch.cat([firstframe_depth[:,q:],firstframe_depth[:,:q]],dim=1)
    firstframe_rgb = torch.cat([firstframe_rgb[:,W//2:],firstframe_rgb[:,:W//2]],dim=1)
    firstframe_depth = torch.cat([firstframe_depth[:,W//2:],firstframe_depth[:,:W//2]],dim=1)

    if preset_rail is None:
        Rts = generate_rail(firstframe_depth, 0., movement_ratio, frame_size, mode=mode)
    else:
        Rts = preset_rail
    
    #depth_pvt = depth[int(H//2*1.05),q]
    
    '''for i in range(frame_size):
        Rts[i,2,3] = -(depth_pvt * i *movement_ratio)/(frame_size-1)'''

    print(Rts.device, firstframe_rgb.device, firstframe_depth.device)
    print(Rts.shape, firstframe_rgb.shape, firstframe_depth.shape)  
    print(Rts.dtype, firstframe_rgb.dtype, firstframe_depth.dtype) 
    print(torch.cuda.memory_allocated() / 1024**3, "GB used")
    print(torch.cuda.memory_reserved() / 1024**3, "GB reserved") 
    torch.cuda.empty_cache()
    print(torch.cuda.memory_allocated() / 1024**3, "GB used")  
    print(torch.cuda.memory_reserved() / 1024**3, "GB reserved")  
    # import pdb
    # pdb.set_trace()
        
    rgbs, masks = generate_pc_render(Rts, firstframe_rgb, firstframe_depth)
    return rgbs, masks, Rts, firstframe_rgb, firstframe_depth
def intersection_check(firstframe_depth, Rts, max_ratio = 0.5):
    # 假设首帧为单位阵，也即首帧的坐标系为世界坐标系。
    c2ws = torch.linalg.inv(Rts)
    pos = c2ws[:,:3,3]
    pos_norm = pos.norm(dim=-1)
    proj_pos = pos/pos.norm(dim=-1,keepdim=True)

    cos_value = proj_pos[:,2]
    sin_value = proj_pos[:,0]
    ac = torch.arccos(cos_value)
    mask_2pi_minus = sin_value<0
    angle_rad = torch.arccos(cos_value)
    q = 2.*math.pi-angle_rad[mask_2pi_minus]
    angle_rad[mask_2pi_minus] = q
    angle_azimuth = torch.rad2deg(angle_rad)

    sin_value_elevation = proj_pos[:,1]
    angle_rad_elevation = torch.arcsin(sin_value_elevation)
    angle_elevation = torch.rad2deg(angle_rad_elevation)
    
    W = firstframe_depth.shape[1]
    H = firstframe_depth.shape[0]
    pixel_value = torch.zeros((angle_azimuth.shape[0],2),dtype=torch.int32,device=Rts.device)
    #pixel_value[:,0] = H//2
    pixel_value[:,0] = ((angle_elevation + 90.) / 180. * (H - 1)).long()
    pixel_value[:,1] = (((angle_azimuth / 360.) * W).long() + W//2)%W
    depth_fetch = firstframe_depth[pixel_value[:,0],pixel_value[:,1]]
    # print(depth_fetch.shape,pos_norm.shape, depth_fetch.shape,pixel_value,angle_azimuth)
    ratio = pos_norm/depth_fetch
    rmax= ratio.max()
    print(f"intersection ratio max: {rmax}")
    c2ws[:,:3,3] = c2ws[:,:3,3]/rmax * max_ratio
    return torch.linalg.inv(c2ws)
def absolute_scale(firstframe_depth, Rts, scale):
    # Absolute alternative to intersection_check: instead of dividing the path by its
    # global worst frame (which makes anchor magnitude meaningless and couples every
    # anchor to every other), translate by a FIXED gain tied only to the scene's
    # depth scale. So anchor coordinates become absolute -- dragging one point farther
    # moves the camera farther there, leaving the rest of the path unchanged, and
    # `scale` is a one-time per-scene gain rather than something to re-tune per edit.
    #
    # Unit: 1 anchor unit == `scale` x (median scene depth). The median over valid
    # depth is a stable "typical scene distance", so the same anchors behave
    # consistently across scenes regardless of MoGe's depth magnitude. There is NO
    # collision guard here -- the camera CAN enter geometry (that's the point).
    c2ws = torch.linalg.inv(Rts)
    finite = torch.isfinite(firstframe_depth)
    valid = firstframe_depth[finite & (firstframe_depth > 0)]
    d_ref = float(valid.median()) if valid.numel() > 0 else 1.0
    c2ws[:,:3,3] = c2ws[:,:3,3] * (scale * d_ref)
    print(f"absolute scale: gain={scale} x d_ref(median depth)={d_ref:.4f}")
    return torch.linalg.inv(c2ws)
def literal_scale(Rts, scale):
    # WYSIWYG absolute: anchor coordinates are taken LITERALLY in scene/depth units,
    # translated by `scale` only (no d_ref normalization, no collision guard). With
    # scale=1 an anchor sits exactly at that 3D point in the scene -- which is what the
    # Camera Plot + Geometry editor draws, so the camera goes where you placed it
    # relative to the point cloud.
    c2ws = torch.linalg.inv(Rts)
    c2ws[:,:3,3] = c2ws[:,:3,3] * scale
    print(f"literal scale: gain={scale}")
    return torch.linalg.inv(c2ws)
def load_rail(json_path):
    with open(json_path,"r") as F_:
        d = json.load(F_)
    arr = np.array(d)
    return arr
def perform_camera_movement_with_cam_input(firstframe_rgb, firstframe_depth, angle=0., movement_ratio = 0.5, frame_size=81, preset_rail=None,mode="s_curve", scale_mode="auto"):

    device = firstframe_depth.device
    depth_np = firstframe_depth.cpu().numpy()
    depth_np = depth_repair(depth_np)
    if preset_rail is not None:
        # 把相机转成和首帧一致
        '''
        z_axis = x_axis_ * math.sin(math.radians(angle)) + z_axis_ * math.cos(math.radians(angle))
        x_axis = z_axis_ * -math.sin(math.radians(angle)) + x_axis_ * math.cos(math.radians(angle))
        '''
        if isinstance(preset_rail, np.ndarray):
            preset_rail = torch.from_numpy(preset_rail).float().to(device)
        # print("sd")
        firstframe_Rt = preset_rail[0]
        firstframe_z = firstframe_Rt[2,:3]
        firstframe_x = firstframe_Rt[0,:3]
        original_z = torch.tensor([0,0,1.],dtype=torch.float32).to(device)
        original_x = torch.tensor([1.,0,0],dtype=torch.float32).to(device)

        cos_value = float((firstframe_z * original_z).sum())
        sin_value = float((firstframe_z * original_x).sum())
        angle_rad = math.acos(cos_value) if sin_value >=0 else 2 * math.pi - math.acos(cos_value)
        angle = math.degrees(angle_rad)
        
        preset_rail_ = preset_rail[...] @ torch.linalg.inv(preset_rail[0])[None]
        


    W = firstframe_rgb.shape[1]
    firstframe_depth = torch.from_numpy(depth_np).to(device)
    q = int(angle/360. * W + W//2)%W
    firstframe_rgb = torch.cat([firstframe_rgb[:,q:],firstframe_rgb[:,:q]],dim=1)
    firstframe_depth = torch.cat([firstframe_depth[:,q:],firstframe_depth[:,:q]],dim=1)
    firstframe_rgb = torch.cat([firstframe_rgb[:,W//2:],firstframe_rgb[:,:W//2]],dim=1)
    firstframe_depth = torch.cat([firstframe_depth[:,W//2:],firstframe_depth[:,:W//2]],dim=1)

    if preset_rail is None:
        Rts = generate_rail(firstframe_depth, 0., movement_ratio, frame_size, mode=mode)
    elif scale_mode == "absolute":
        # Anchor coordinates are absolute (in units of scene depth); movement_ratio is
        # a one-time gain. No global renormalization -> editing one anchor is local.
        Rts = absolute_scale(firstframe_depth, preset_rail_, movement_ratio)
    elif scale_mode == "absolute_literal":
        # WYSIWYG: anchors literally in scene/depth units (matches the geometry editor).
        Rts = literal_scale(preset_rail_, movement_ratio)
    else:
        # "auto": honor the caller's movement_ratio as the collision-guard target
        # fraction (was hardcoded 0.5). The whole path is renormalized so its most-
        # extreme frame sits at that fraction of the scene depth.
        Rts = intersection_check(firstframe_depth,preset_rail_,movement_ratio)
    
    #depth_pvt = depth[int(H//2*1.05),q]
    
    '''for i in range(frame_size):
        Rts[i,2,3] = -(depth_pvt * i *movement_ratio)/(frame_size-1)'''
    rgbs, masks = generate_pc_render(Rts, firstframe_rgb, firstframe_depth)
    return rgbs, masks, Rts, firstframe_rgb, firstframe_depth, angle




if __name__ == "__main__":
    print("nv test")
    base_dir = "/ai-video-sh/common/common/Datasets/Panorama_Selected0425/CamTracks/POLYGON_CityPack/POLYGON_CityPackData/Day/Rail_PCG_Road2_50_42__Take_183"
    debug_dir = "/ai-video-sh/zhongqi.yang/code/zhongqi.yang/Trajcrafter_Training/dataprocess_minimal/debug"

    frame_size = 49
    filelist = os.listdir(base_dir)
    rgbs = []
    depths = []
    csv_name = None
    for i in filelist:
        if i.endswith("col.exr"):
            rgbs.append(i)
        elif i.endswith("depth.exr"):
            depths.append(i)
        elif i.endswith("csv"):
            csv_name = i
    rgbs = sorted(rgbs)
    depths = sorted(depths)
    if csv_name is None:
        raise ValueError("csv not found!")
    # load csv as ypr.
    csv_file = os.path.join(base_dir, csv_name)
    csv_parameters = read_cam_csv(csv_file)
    csv_parameters = csv_parameters[:frame_size]
    all_rgbs = []
    all_depths = []
    for i in range(frame_size):
        depth = cv2.imread(os.path.join(base_dir, depths[i]), cv2.IMREAD_ANYCOLOR|cv2.IMREAD_ANYDEPTH)[...,0].astype(np.float32)
        rgb = np.clip(cv2.imread(os.path.join(base_dir, rgbs[i]), cv2.IMREAD_ANYCOLOR|cv2.IMREAD_ANYDEPTH)[...].astype(np.float32), 0, 1.)
        all_rgbs.append(rgb)
        all_depths.append(depth)
    all_rgbs = np.stack(all_rgbs, axis=0)
    all_depths = np.stack(all_depths, axis=0)
    # (images, depths, K, Rts, resolution, frame_interval = 8, horizontal_cut_num = 4, vertical_cut_num = 1):
    K = np.array([
        [256.,0,256.],
        [0.,256,256.],
        [0.,0,1.],
    ])
    Rts = csv_cam_to_opencv(csv_parameters)
    all_depths = correct_pano_depth_batch(all_depths)
    print(all_depths[0].max(),all_depths[0].min())
    #glctx = dr.RasterizeCudaContext(device="cuda:0")
    mask = all_depths > 1000.
    all_depths[mask] = 1000.
    mesh = get_mesh_from_pano(all_depths[0], np.ones_like(all_depths[0]).astype(np.bool_), csv_parameters[0] ,dtype=np.float32)
    
    depth,render_mask = get_mesh_render_depth(mesh, (K[None]), (Rts[:1]), 512,512,1e-3,2000, "cuda:0")
    print(depth.max(), depth.min())
    mask_ = depth < 1e-5
    depth[mask_ ] = 0.1
    depth_inverse = 1./depth[0]
    depth_inverse_np = (255. * (depth_inverse - depth_inverse.min()) / (depth_inverse.max() - depth_inverse.min())).astype(np.uint8)
    cv2.imwrite("./test.png", depth_inverse_np[:,:,None].repeat(3,2))
    cv2.imwrite("./rgb_pano.png", (all_rgbs[0] * 255.).astype(np.uint8))

