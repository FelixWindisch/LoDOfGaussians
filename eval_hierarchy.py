#
# Copyright (C) 2023 - 2024, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import argparse
from utils.general_utils import get_expon_lr_func
import os
import torch
from torch import nn
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render_gsplat
import sys
from scene import Scene, GaussianModel, OurAdam
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from torch.utils.data import DataLoader
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.general_utils import strip_symmetric, build_scaling_rotation
import math
import torchvision
from fused_ssim import fused_ssim
import random
from torch.utils.tensorboard import SummaryWriter
import time
from torchviz import make_dot
from scipy.spatial import KDTree
import numpy as np
from gaussian_hierarchy._C import  get_spt_cut_cuda
from gaussian_renderer import occlusion_cull
import json
import pickle
from utils.image_utils import psnr
from lpipsPyTorch import lpips

clock_start = True
clock_time = time.time()
def clock():
    global clock_start
    global clock_time
    if clock_start:
        clock_start = False
        clock_time = time.time()
    else:
        clock_start = True
        return time.time()-clock_time


def direct_collate(x):
    return x




SH_properties = [0, 3, 8, 15]
number_SH_properties = [0, 3, 8, 15]
SH_properties_single = None
SH_properties = None
xyz1 = 0
xyz2 = 3
scales1 = 3
scales2 = 6
rotation1 = 6
rotation2 = 10
features1 = 10
features2 = 13
opacity1 = 13
opacity2 = 14
features_rest1 = 14
features_rest2 = None
number_properties = features_rest2

range1 = [xyz1, scales1, rotation1, features1, opacity1, features_rest1]
range2 = [xyz2, scales2, rotation2, features2, opacity2, features_rest2]

non_blocking=False
def render(dataset, opt:OptimizationParams, pipe,   replay=False, cam_path_id=0, test_set=False):
    global SH_properties, Max_SH_Degree, features_rest2, SH_properties, SH_properties_single

    first_iter = 0
    prepare_output_and_logger(dataset)
    gaussians = GaussianModel(None) # SH degree is determined by hierarchy file
    gaussians.scaffold_points = None
    with torch.no_grad():
        gaussians._features_dc = gaussians._features_dc.abs() 
    dataset.eval = False
    scene = Scene(dataset, gaussians, resolution_scales = [1], create_from_hier=True, llff_hold = opt.llff_hold)
    features_rest2 = 14 + number_SH_properties[gaussians.max_sh_degree] * 3 
    range2[-1] = 14 + number_SH_properties[gaussians.max_sh_degree] * 3
    SH_properties_single = number_SH_properties[gaussians.max_sh_degree] 
    SH_properties = number_SH_properties[gaussians.max_sh_degree] * 3
    gaussians.skybox_points = 100000
    base_focal_length = scene.getTrainCameras()[0].focal_length
    SPT_Target_Granularity = (1.0/base_focal_length) * opt.target_granularity_pixels
        
        
    gaussians.compact_gaussians(opt.storage_device, None,  False, False)

    gaussians.build_hierarchical_SPT(opt.SPT_root_volume, SPT_Target_Granularity, use_bounding_spheres=opt.use_bounding_spheres)
    print(f"Built {len(gaussians.SPT_starts)} SPTs")

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    first_iter += 1


    iteration = first_iter
    #means3D, opacity, scales, rotations, features_dc, features_rest, gaussian_indices = torch.empty((0, 3), device='cuda', dtype=torch.float32), torch.empty((0, 1), device='cuda', dtype=torch.float32), torch.empty((0, 3), device='cuda', dtype=torch.float32), torch.empty((0,4), device='cuda', dtype=torch.float32), torch.empty((0, 1, 3), device='cuda', dtype=torch.float32), torch.empty((0,15, 3), device='cuda', dtype=torch.float32), torch.empty(0, device='cuda', dtype=torch.int32)
    means3D = gaussians.properties[:gaussians.skybox_points, xyz1:xyz2].cuda().contiguous()
    scales = gaussians.properties[:gaussians.skybox_points, scales1:scales2].cuda().contiguous()
    rotations = gaussians.properties[:gaussians.skybox_points, rotation1:rotation2].cuda().contiguous()
    features_dc = gaussians.properties[:gaussians.skybox_points, features1:features2].cuda().unsqueeze(1).contiguous()
    opacity = gaussians.properties[:gaussians.skybox_points, opacity1].cuda().unsqueeze(1).contiguous()
    features_rest = gaussians.properties[:gaussians.skybox_points, features_rest1: features_rest2].cuda().reshape(gaussians.skybox_points, SH_properties_single, 3).contiguous()

    
    gaussians.xyz_scheduler_args = get_expon_lr_func(lr_init=opt.position_lr_init*gaussians.spatial_lr_scale,
                                                    lr_final=opt.position_lr_final*gaussians.spatial_lr_scale,
                                                    lr_delay_mult=opt.position_lr_delay_mult,
                                                    max_steps=opt.position_lr_max_steps)
    if len(scene.getTestCameras()) == 0:
        test_set = True
    training_generator = DataLoader(scene.getTestCameras() if test_set else scene.getTrainCameras(), num_workers = 8, prefetch_factor = 1, persistent_workers = True, collate_fn=direct_collate, shuffle=False)
    psnrs = 0.0
    ssims = 0.0
    lpipss = 0.0
    
    
    psnr_aerial = 0.0
    ssims_aerial = 0.0
    lpipss_aerial = 0.0
    
    
    psnr_street = 0.0
    ssims_street = 0.0
    lpipss_street = 0.0
    
    street_images = 0
    aerial_images = 0
    distance_multiplier = 1

    
    i = 0
    for viewpoint_batch in tqdm(training_generator):
        for viewpoint_cam in viewpoint_batch:
            i+=1
            viewpoint_cam.world_view_transform = viewpoint_cam.world_view_transform.cuda()
            #viewpoint_cam.projection_matrix = viewpoint_cam.projection_matrix.cuda()
            viewpoint_cam.full_proj_transform = viewpoint_cam.full_proj_transform.cuda()
            viewpoint_cam.camera_center = viewpoint_cam.camera_center.cuda()
            ############# SPT Cache
            if opt.use_bounding_spheres:
                bounds = gaussians.bounding_sphere_radii
            else: 
                bounds = (gaussians.scaling_activation(torch.max(gaussians.upper_tree_scaling, dim=-1)[0]) * 3.0)
            planes = gaussians.extract_frustum_planes(viewpoint_cam.full_proj_transform.cuda())
            if opt.use_frustum_culling:
                frustum_cull = lambda indices : gaussians.frustum_cull_spheres(gaussians.upper_tree_xyz[indices], bounds[indices], planes)
            else:
                frustum_cull = lambda indices : torch.ones(len(indices), dtype = torch.bool)
            camera_position = viewpoint_cam.camera_center.cuda()
            LOD_detail_cut = lambda indices : gaussians.min_distance_squared[indices] > (camera_position - gaussians.upper_tree_xyz[indices]).square().sum(dim=-1) * distance_multiplier
            # The coarse cut contains intermediate nodes from the upper tree and leaf nodes, with some leaf nodes containing SPTs
            coarse_cut = gaussians.cut_hierarchy_on_condition(gaussians.upper_tree_nodes, LOD_detail_cut, return_upper_tree=False, root_node=0, leave_out_of_cut_condition=frustum_cull)
            
            cut_leaf_nodes = coarse_cut[gaussians.upper_tree_nodes[coarse_cut, 2] == 0]
            # separate the cut into leafs that contain an SPT and those that don't
            # The SPT indices are the child indices of those nodes that have a 0 child count
            SPT_indices = gaussians.upper_tree_nodes[cut_leaf_nodes][gaussians.upper_tree_nodes[cut_leaf_nodes, 3] >= 0, 3]
            upper_tree_nodes_to_render = gaussians.upper_tree_nodes[coarse_cut][gaussians.upper_tree_nodes[coarse_cut, 3] <= 0, 5]
            SPT_upper_tree_indices = cut_leaf_nodes[gaussians.upper_tree_nodes[cut_leaf_nodes, 3] >= 0]
            SPT_distances = (gaussians.upper_tree_xyz[SPT_upper_tree_indices] - camera_position).pow(2).sum(1).sqrt() * distance_multiplier
            
            
            # Keep all the gaussians that are contained in an STP that is reused and close enough
            # Cumulative Sum Trick
            
            # Keep Skybox
            #keep_gaussians_mask[:gaussians.skybox_points] = True
            #mask = torch.isin(SPT_indices, keep_SPT_indices)
              
            load_SPT_indices = SPT_indices
            load_SPT_distances = SPT_distances
            if len(load_SPT_indices) > 0:
                #LOAD SPT CUT
                #load_SPT_distances = torch.full((len(load_SPT_indices),), 10000.0).cuda()
                load_SPT_gaussian_indices, load_SPT_starts = get_spt_cut_cuda(len(load_SPT_indices), gaussians.SPT_gaussian_indices, gaussians.SPT_starts, gaussians.SPT_max, gaussians.SPT_min, load_SPT_indices, load_SPT_distances)
            else:
                print("No SPTs loaded")
                load_SPT_gaussian_indices, load_SPT_starts = torch.empty(0, dtype=torch.int32, device='cuda'), torch.empty(0, dtype=torch.int32, device='cuda')
            if len(load_SPT_starts) > 0:    
                if len(load_SPT_gaussian_indices) == load_SPT_starts[-1]:
                    print("Last SPT empty")
                    load_SPT_starts = load_SPT_starts[:-1]
                    load_SPT_distances = load_SPT_distances[:-1]
                    load_SPT_indices = load_SPT_indices[:-1]
            #    
            ### BAND AID FIX
            
            load_from_disk_indices = torch.cat((upper_tree_nodes_to_render, load_SPT_gaussian_indices))
            load_from_disk_indices = load_from_disk_indices.to(opt.storage_device)
            
            
            
            load_tensor = gaussians.properties[load_from_disk_indices, :].cuda(non_blocking=non_blocking)
            means3D = nn.Parameter(torch.cat((means3D[:gaussians.skybox_points], load_tensor[:, xyz1:xyz2].cuda(non_blocking=non_blocking))).contiguous())
            opacity = nn.Parameter(torch.cat((opacity[:gaussians.skybox_points], load_tensor[:, opacity1:opacity2].cuda(non_blocking=non_blocking))).contiguous())
            scales = nn.Parameter(torch.cat((scales[:gaussians.skybox_points], load_tensor[:, scales1:scales2].cuda(non_blocking=non_blocking))).contiguous())
            rotations = nn.Parameter(torch.cat((rotations[:gaussians.skybox_points], load_tensor[:, rotation1:rotation2].cuda(non_blocking=non_blocking))).contiguous())
            # TODO: ABS?
            features_dc = nn.Parameter(torch.cat((features_dc[:gaussians.skybox_points], load_tensor[:, features1:features2].cuda(non_blocking=non_blocking).unsqueeze(1))).contiguous())
            features_rest = nn.Parameter(torch.cat((features_rest[:gaussians.skybox_points], load_tensor[:, features_rest1:features_rest2].cuda(non_blocking=non_blocking).reshape(len(load_tensor), SH_properties_single, 3))).contiguous())
            torch.cuda.empty_cache()
                
                
            
            
            
            render_pkg = render_gsplat(
                viewpoint_cam, 
                means3D,
                gaussians.opacity_activation(opacity),
                gaussians.scaling_activation(scales), 
                gaussians.rotation_activation(rotations),
                features_dc,
                features_rest,
                pipe, 
                background,
                override_color = None, 
                sh_degree = gaussians.active_sh_degree,
                )
            torch.cuda.empty_cache()
            
            image = render_pkg["render"]
            gt_image = viewpoint_cam.original_image.cuda()
            
            if viewpoint_cam.alpha_mask is not None:
                alpha_mask = viewpoint_cam.alpha_mask.cuda()
                image *= alpha_mask
                gt_image *= alpha_mask
            
            iteration += 1
            psnr_current = psnr(image.detach(), gt_image).mean().double()
            ssim_current = ssim(image.detach(), gt_image).mean().double()
            lpips_current = lpips(image.detach(), gt_image, net_type='vgg').mean().double()
            torchvision.utils.save_image(image,  "output/" + os.path.basename(viewpoint_cam.image_name) + ".png")
            #torchvision.utils.save_image(gt_image,  "output/" + viewpoint_cam.image_name + "_gt.png")
            psnrs += psnr_current
            ssims += ssim_current
            lpipss += lpips_current
            if image.shape[2]  < 1100:
                street_images += 1
                psnr_street += psnr_current
                ssims_street += ssim_current
                lpipss_street += lpips_current
            else:
                aerial_images += 1
                psnr_aerial += psnr_current
                ssims_aerial += ssim_current
                lpipss_aerial += lpips_current
                ####### RENDER
                

    psnrs /= len(training_generator)
    ssims /= len(training_generator)
    lpipss /= len(training_generator)

    print(f"PSNR: {psnrs:.5f} SSIM: {ssims:.5f} LPIPS: {lpipss:.5f}")
    print(f"Renders saved to output/")
    exit()

def prepare_output_and_logger(args):    
    
    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--replay', type=bool, default=False)
    parser.add_argument('--ID', type=int, default=0)
    
    # Whether to use test set or training set.
    parser.add_argument('--test_set', type=bool, default=True)
    
    parser.add_argument('--config', default="")
    
    args = parser.parse_args(sys.argv[1:])
    
    
    
    with open(f"configs/{args.config}", "r") as f:
        data = json.load(f)
    config = argparse.Namespace(**data)
    optimization_params = op.extract(config)
    
    render(lp.extract(args), optimization_params, pp.extract(args), args.replay, args.ID, args.test_set)
        
    print("\nEval complete.")
