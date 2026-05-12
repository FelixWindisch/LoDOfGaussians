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
import traceback

from utils.general_utils import get_expon_lr_func
import os
import torch
from torch import nn
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import  render_gsplat, network_gui
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
from gaussian_hierarchy._C import expand_to_size, get_interpolation_weights, expand_to_size_dynamic, get_interpolation_weights_dynamic
import time
from torchviz import make_dot
from scipy.spatial import KDTree
import numpy as np
from gaussian_hierarchy._C import  get_spt_cut_cuda
from gaussian_renderer import occlusion_cull
import json
import pickle
import colorsys
import argparse
import warnings
warnings.filterwarnings("ignore", module="gsplat")
def generate_colors(n, saturation=0.6, lightness=0.5):
    colors = torch.zeros((n, 3), dtype=torch.float32, device='cuda')
    for i in range(n):
        hue = i / n  # evenly spaced hues
        r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
        #rgb = tuple(int(x * 255) for x in (r, g, b))
        colors[i, 0] = r
        colors[i, 1] = g
        colors[i, 2] = b
    return colors


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


sub_clock_start = True
sub_clock_time = time.time()
def sub_clock():
    global sub_clock_start
    global sub_clock_time
    if sub_clock_start:
        sub_clock_start = False
        sub_clock_time = time.time()
    else:
        sub_clock_start = True
        return time.time()-sub_clock_time

def direct_collate(x):
    return x


replay_stats = {"frame_time": [], "cut_time": [], "VRAM" : [], "number_rendered" : []}

WriteTensorBoard = False


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
def render(dataset, opt:OptimizationParams, pipe, checkpoint,   replay=False, cam_path_id=0):
    global SH_properties, features_rest2, SH_properties, SH_properties_single

    camera_path_id = random.randint(0, 100000) if not replay else cam_path_id
    global Reuse_SPT_Tolerarance
    #torch.cuda.memory._record_memory_history()
    #torch.autograd.set_detect_anomaly(True)
    
    first_iter = 0
    prepare_output_and_logger(dataset)
    gaussians = GaussianModel(None)
    gaussians.scaffold_points = None
    with torch.no_grad():
        gaussians._features_dc = gaussians._features_dc.abs() 
    dataset.eval = True
    scene = Scene(dataset, gaussians, resolution_scales = [1], create_from_hier=True, llff_hold= opt.llff_hold)
    features_rest2 = 14 + number_SH_properties[gaussians.max_sh_degree] * 3
    range2[-1] = 14 + number_SH_properties[gaussians.max_sh_degree] * 3 
    SH_properties_single = number_SH_properties[gaussians.max_sh_degree] 
    SH_properties = number_SH_properties[gaussians.max_sh_degree] * 3
    
    gaussians.skybox_points = 100000
    base_focal_length = 1000 #TODO ????
    SPT_Target_Granularity = (1.0/base_focal_length) * opt.target_granularity_pixels
    #with torch.no_grad():
    #    gaussians._opacity.clamp_(0, 0.99999)
    #    gaussians._opacity = gaussians.inverse_opacity_activation(gaussians._opacity)
        
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
        
    gaussian_indices = torch.zeros(gaussians._xyz.size(0)).int().cuda()
    
    #scene.dump_gaussians("Dump", only_leaves=True)

    gaussians.compact_gaussians(opt.storage_device, max_number_of_gaussians=None, densification=False, training=False)

    gaussians.build_hierarchical_SPT(opt.SPT_root_volume, SPT_Target_Granularity, opt.use_bounding_spheres)
    print("Built SPTs")
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    ema_loss_for_log = 0.0
    #progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    indices = None

    iteration = first_iter
    # Dataloader loads data from disk
    # DONT SHUFFLE IF USING CORRESPONDENCE GRAPH

    
    #means3D, opacity, scales, rotations, features_dc, features_rest, gaussian_indices = torch.empty((0, 3), device='cuda', dtype=torch.float32), torch.empty((0, 1), device='cuda', dtype=torch.float32), torch.empty((0, 3), device='cuda', dtype=torch.float32), torch.empty((0,4), device='cuda', dtype=torch.float32), torch.empty((0, 1, 3), device='cuda', dtype=torch.float32), torch.empty((0,15, 3), device='cuda', dtype=torch.float32), torch.empty(0, device='cuda', dtype=torch.int32)

    gaussian_indices = torch.arange(0, gaussians.skybox_points, device='cuda')
    means3D = gaussians.properties[:gaussians.skybox_points, xyz1:xyz2].cuda().contiguous()
    scales = gaussians.properties[:gaussians.skybox_points, scales1:scales2].cuda().contiguous()
    rotations = gaussians.properties[:gaussians.skybox_points, rotation1:rotation2].cuda().contiguous()
    features_dc = gaussians.properties[:gaussians.skybox_points, features1:features2].cuda().unsqueeze(1).contiguous()
    opacity = gaussians.properties[:gaussians.skybox_points, opacity1].cuda().unsqueeze(1).contiguous()
    features_rest = gaussians.properties[:gaussians.skybox_points, features_rest1: features_rest2].cuda().reshape(gaussians.skybox_points, SH_properties_single, 3).contiguous()
    
    
    # for each SPT, store the distance, index and start in the rendering set from the previous iteration
    prev_SPT_distances = torch.empty(0, dtype = torch.float32, device='cuda')
    prev_SPT_indices = torch.empty(0, dtype = torch.int32, device='cuda')
    prev_SPT_starts = torch.empty(0, dtype = torch.int32, device='cuda')    

    if not replay:
        with open(f"CameraPaths/camera_path_{camera_path_id}.txt", "w") as f:
            f.write("")
    else: 
        cam_path_file = open(f"CameraPaths/camera_path_{camera_path_id}.txt", "rb")
    ######## VIEWER
    if not replay:
        network_gui.init("127.0.0.1", 6009)
    torch.cuda.reset_max_memory_allocated()
    
    viewer_options = {
        "distance_multiplier" : 1.0, 
        "render_SPTs" : False, 
        "freeze_view" : False, 
        "color_distance" : False, 
        "color_size" : False,
        "reuse_SPT_tolerance" : 0.0,
        "separate_SPTs" : False,
        "highlight_leaves" : False,
        "show_occlusion" : False,
        "use_occlusion_culling" : False,
        "record_traj" : False
        }
    
    # Render Loop
    while True:
        if network_gui.conn == None and not replay:
            print("Try Connect")
            network_gui.try_connect()
        while network_gui.conn != None or replay:
            try:
                net_image_bytes = None
                if not replay:
                    custom_cam, do_training, keep_alive_, scaling_modifer, slider = network_gui.receive()
                    for key, value in slider.items():
                        if key in viewer_options:
                            if type(viewer_options[key]) is bool:
                                viewer_options[key] = value > 0
                            if type(viewer_options[key]) is float:
                                viewer_options[key] = value
                else:
                    custom_cam =  pickle.load(cam_path_file)
                if custom_cam != None:
                    ####### RENDER
                    viewpoint_cam = custom_cam
                    
                    if viewer_options["record_traj"]:
                        with open(f"CameraPaths/camera_path_{camera_path_id}.txt", "ab") as f:
                            pickle.dump(custom_cam, f)
                    clock()
                    viewpoint_cam.world_view_transform = viewpoint_cam.world_view_transform.cuda()
                    viewpoint_cam.full_proj_transform = viewpoint_cam.full_proj_transform.cuda()
                    viewpoint_cam.camera_center = viewpoint_cam.camera_center.cuda()
                    
                    if replay:
                        viewpoint_cam.image_width = 1100
                        viewpoint_cam.image_height = 900
                        viewpoint_cam.FoVx =viewpoint_cam.FoVx * 1.1
                        viewpoint_cam.FoVy =viewpoint_cam.FoVy * 0.9
                    if not viewer_options["freeze_view"]:

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
                        sub_clock()
                        LOD_detail_cut = lambda indices : gaussians.min_distance_squared[indices] > (camera_position - gaussians.upper_tree_xyz[indices]).square().sum(dim=-1) * viewer_options["distance_multiplier"]
                        # The coarse cut contains intermediate nodes from the upper tree and leaf nodes, with some leaf nodes containing SPTs
                        coarse_cut = gaussians.cut_hierarchy_on_condition(gaussians.upper_tree_nodes, LOD_detail_cut, return_upper_tree=False, root_node=0, leave_out_of_cut_condition=frustum_cull)

                        if viewer_options["use_occlusion_culling"]: #opt.use_occlusion_culling:
                            bg_color = [0, 0, 0]
                            background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
                            temp = len(coarse_cut)
                            occlusion_indices = gaussians.upper_tree_nodes[coarse_cut, 5]
                            occlusion_mask, occlusion_image = occlusion_cull(occlusion_indices.to(opt.storage_device), gaussians, viewpoint_cam, pipe, background)
                            occlusion_mask = occlusion_mask.cuda()
                            coarse_cut = coarse_cut[occlusion_mask]
                            print(f"Occlusion Cull {temp - len(coarse_cut)} out of {temp} upper tree gaussians")

                        # leaf nodes have 0 children
                        cut_leaf_nodes = coarse_cut[gaussians.upper_tree_nodes[coarse_cut, 2] == 0]
                        # separate the cut into leafs that contain an SPT and those that don't
                        # The SPT indices are the child indices of those nodes that have a 0 child count
                        SPT_indices = gaussians.upper_tree_nodes[cut_leaf_nodes][gaussians.upper_tree_nodes[cut_leaf_nodes, 3] >= 0, 3]

                        upper_tree_nodes_to_render = gaussians.upper_tree_nodes[coarse_cut][gaussians.upper_tree_nodes[coarse_cut, 3] <= 0, 5]

                        SPT_upper_tree_indices = cut_leaf_nodes[gaussians.upper_tree_nodes[cut_leaf_nodes, 3] >= 0]

                        SPT_distances = (gaussians.upper_tree_xyz[SPT_upper_tree_indices] - camera_position).pow(2).sum(1).sqrt() * viewer_options["distance_multiplier"]

                        

                        prev_to_new_SPT_order = torch.searchsorted(SPT_indices, prev_SPT_indices)

                        equal_SPT_cache_mask = (prev_to_new_SPT_order < len(SPT_indices)) & (SPT_indices[prev_to_new_SPT_order.clamp_max(len(SPT_indices)-1)] == prev_SPT_indices)
                        prev_equal_SPT_cache_indices = torch.nonzero(equal_SPT_cache_mask, as_tuple=True)[0]
                        equal_SPT_cache_indices = prev_to_new_SPT_order[equal_SPT_cache_mask]

                        prev_distances_compare = prev_SPT_distances[prev_equal_SPT_cache_indices]
                        distances_compare = SPT_distances[equal_SPT_cache_indices]
                        #close_enough = torch.isclose(distances_compare, prev_distances_compare, rtol=Reuse_SPT_Tolerarance, atol=0.05)
                        close_enough = (prev_distances_compare/distances_compare) > 0.3
                        close_enough &= (prev_distances_compare/distances_compare) < 1.5
                        #close_enough &= (prev_distances_compare/distances_compare) < -1.5
                        reuse_SPT_indices = SPT_indices[equal_SPT_cache_indices[close_enough]]


                        prev_keep_SPT_cache_indices = prev_equal_SPT_cache_indices[close_enough]

                        # Keep all the gaussians that are contained in an STP that is reused and close enough
                        # Cumulative Sum Trick
                        reuse_gaussians_mask = torch.zeros(len(gaussian_indices)+1, dtype=torch.int32, device='cuda')
                        debug_x = prev_SPT_starts[prev_keep_SPT_cache_indices]
                        if len(debug_x) > 0:
                            if torch.max(debug_x, 0)[0].item() > len(reuse_gaussians_mask):
                                print("Debug X is bigger than reuse_gaussians_mask")
                            if torch.min(debug_x, 0)[0].item() < 0:
                                print("Debug X is less than 0")
                        reuse_gaussians_mask[debug_x] += 1
                        debug_y = prev_SPT_starts[prev_keep_SPT_cache_indices+1]
                        reuse_gaussians_mask[debug_y] -= 1
                        reuse_gaussians_mask = reuse_gaussians_mask.cumsum(0)[:-1].bool()

                        load_SPT_mask = torch.zeros(len(SPT_indices), device='cuda', dtype=torch.bool)
                        load_SPT_mask.scatter_(0, equal_SPT_cache_indices[close_enough].to(torch.int64), True)      
                        load_SPT_mask = ~load_SPT_mask                  
                        load_SPT_indices = SPT_indices[load_SPT_mask]
                        load_SPT_distances = SPT_distances[load_SPT_mask]

                        if len(load_SPT_indices) > 0:
                            #LOAD SPT CUT
                            load_SPT_gaussian_indices, load_SPT_starts = get_spt_cut_cuda(len(load_SPT_indices), gaussians.SPT_gaussian_indices, gaussians.SPT_starts, gaussians.SPT_max, gaussians.SPT_min, load_SPT_indices, load_SPT_distances)
                        else:
                            print("No SPTs loaded")
                            load_SPT_gaussian_indices, load_SPT_starts = torch.empty(0, dtype=torch.int32, device='cuda'), torch.empty(0, dtype=torch.int32, device='cuda')
                        #SPT_counts += gaussians.skybox_points
                        cut_time = sub_clock()

                        if len(load_SPT_starts) > 0:    
                            if len(load_SPT_gaussian_indices) == load_SPT_starts[-1]:
                                print("Last SPT empty")
                                load_SPT_starts = load_SPT_starts[:-1]
                                load_SPT_distances = load_SPT_distances[:-1]
                                load_SPT_indices = load_SPT_indices[:-1]
                        #    
                        ### BAND AID FIX

                        #cache_SPT_cache_indices = torch.where(~equal_SPT_cache_mask)[0]    
                        #cache_SPT_indices = prev_SPT_indices[cache_SPT_cache_indices]
                        SPT_indices = torch.cat((load_SPT_indices, reuse_SPT_indices))


                        SPT_starts_new = torch.zeros(len(load_SPT_indices) + len(reuse_SPT_indices) + 1,dtype=torch.int32, device='cuda')
                        # compact the prefix sum of SPT_counts

                        SPT_starts_new[:len(load_SPT_starts)] = load_SPT_starts + gaussians.skybox_points
                        SPT_starts_new[len(load_SPT_starts)] = len(load_SPT_gaussian_indices) + gaussians.skybox_points
                        #prefix = len(cut_SPTs) + gaussians.skybox_points
                        #for index, i in enumerate(SPT_keep_counts_indices):
                        #    SPT_starts_new[index + len(SPT_counts)] = prefix
                        #    prefix += (prev_SPT_counts[i+1] - prev_SPT_counts[i]).item()

                        sizes = prev_SPT_starts[prev_keep_SPT_cache_indices + 1] - prev_SPT_starts[prev_keep_SPT_cache_indices]
                        SPT_starts_new[len(load_SPT_starts) + 1:len(load_SPT_starts) + 1 + len(sizes)] = torch.cumsum(sizes, dim=0) +  len(load_SPT_gaussian_indices) + gaussians.skybox_points

                        number_of_gaussians_to_render = SPT_starts_new[len(load_SPT_starts) + len(sizes)]


                        SPT_distances = torch.cat((load_SPT_distances, prev_SPT_distances[prev_keep_SPT_cache_indices]))

                        load_from_disk_indices = torch.cat((upper_tree_nodes_to_render, load_SPT_gaussian_indices))



                        gaussian_indices = torch.cat((gaussian_indices[:gaussians.skybox_points], load_from_disk_indices, gaussian_indices[reuse_gaussians_mask]))
                        print(f"Load Percent: {len(load_from_disk_indices) * 100/ number_of_gaussians_to_render}")
                        load_from_disk_indices = load_from_disk_indices.to(opt.storage_device)
                        SPT_starts_new += len(upper_tree_nodes_to_render)

                        assert(SPT_starts_new[-1] == len(gaussian_indices))
                        number_to_render = len(gaussian_indices)
                        distance_multiplier = viewer_options["distance_multiplier"]
                        
                        load_tensor = gaussians.properties[load_from_disk_indices, :].cuda(non_blocking=non_blocking)

                        means3D = nn.Parameter(torch.cat((means3D[:gaussians.skybox_points], load_tensor[:, xyz1:xyz2].cuda(non_blocking=non_blocking), means3D[reuse_gaussians_mask])).contiguous())
                        opacity = nn.Parameter(torch.cat((opacity[:gaussians.skybox_points], load_tensor[:, opacity1:opacity2].cuda(non_blocking=non_blocking), opacity[reuse_gaussians_mask])).contiguous())
                        scales = nn.Parameter(torch.cat((scales[:gaussians.skybox_points], load_tensor[:, scales1:scales2].cuda(non_blocking=non_blocking), scales[reuse_gaussians_mask])).contiguous())
                        rotations = nn.Parameter(torch.cat((rotations[:gaussians.skybox_points], load_tensor[:, rotation1:rotation2].cuda(non_blocking=non_blocking), rotations[reuse_gaussians_mask])).contiguous())
                        # TODO: ABS?
                        features_dc = nn.Parameter(torch.cat((features_dc[:gaussians.skybox_points], load_tensor[:, features1:features2].cuda(non_blocking=non_blocking).unsqueeze(1), features_dc[reuse_gaussians_mask])).contiguous())
                        features_rest = nn.Parameter(torch.cat((features_rest[:gaussians.skybox_points], load_tensor[:, features_rest1:features_rest2].cuda(non_blocking=non_blocking).reshape(len(load_tensor), SH_properties_single, 3), features_rest[reuse_gaussians_mask])).contiguous())


                        prev_SPT_indices = SPT_indices
                        prev_SPT_distances = SPT_distances
                        prev_SPT_starts = SPT_starts_new
                        torch.cuda.empty_cache()
                        
                        
                    
                    if viewer_options["color_distance"]:
                        colors_precomp = torch.zeros_like(scales, device='cuda')
                        SPT_distances_remapped = (SPT_distances - SPT_distances.min()) / (SPT_distances.max() - SPT_distances.min())
                        #SPT_distances_remapped = (SPT_distances) / 500.0

                        # Skybox is blue
                        colors_precomp[:gaussians.skybox_points, 0] = 0
                        colors_precomp[:gaussians.skybox_points, 1] = 0
                        colors_precomp[:gaussians.skybox_points, 2] = 1
                        for i in range(len(SPT_indices)):
                            min_range =  SPT_starts_new[i]
                            max_range = len(gaussian_indices) if i+1 == len(SPT_indices) else  SPT_starts_new[i+1]
                            colors_precomp[min_range:max_range, 0] = SPT_distances_remapped[i]
                            colors_precomp[min_range:max_range, 1] = SPT_distances_remapped[i]
                            colors_precomp[min_range:max_range, 2] = SPT_distances_remapped[i]
                    elif viewer_options["color_size"]:
                        colors_precomp = torch.zeros_like(scales, device='cuda')
                        SPT_sizes = torch.abs(SPT_starts_new[1:] - SPT_starts_new[:-1]) / (gaussians.SPT_starts[SPT_indices+1] - gaussians.SPT_starts[SPT_indices])
                        SPT_sizes_remapped = (SPT_sizes - SPT_sizes.min()) / (SPT_sizes.max() - SPT_sizes.min())
                        # Skybox is blue
                        colors_precomp[:gaussians.skybox_points, 0] = 0
                        colors_precomp[:gaussians.skybox_points, 1] = 0
                        colors_precomp[:gaussians.skybox_points, 2] = 1
                        for i in range(len(SPT_indices)):
                            min_range = SPT_starts_new[i]
                            max_range = len(gaussian_indices) if i+1 == len(SPT_indices) else SPT_starts_new[i+1]
                            colors_precomp[min_range:max_range, 0] = SPT_sizes_remapped[i]
                            colors_precomp[min_range:max_range, 1] = SPT_sizes_remapped[i]
                            colors_precomp[min_range:max_range, 2] = SPT_sizes_remapped[i]
                    elif viewer_options["separate_SPTs"]:
                        colors_precomp = torch.zeros_like(scales, device='cuda')
                        # Skybox is blue
                        colors_precomp[:gaussians.skybox_points, 0] = 0
                        colors_precomp[:gaussians.skybox_points, 1] = 0
                        colors_precomp[:gaussians.skybox_points, 2] = 1
                        random_colors = generate_colors(len(SPT_indices))
                        for i in range(len(SPT_indices)):
                            min_range =  SPT_starts_new[i]
                            max_range = len(gaussian_indices) if i+1 == len(SPT_indices) else  SPT_starts_new[i+1]
                            colors_precomp[min_range:max_range, :] = random_colors[i]
                        colors_precomp[-len(upper_tree_nodes_to_render):, :] = 1
                    elif viewer_options["highlight_leaves"]:
                        colors_precomp = torch.zeros_like(scales, device='cuda')
                        # Skybox is blue
                        colors_precomp[:, 0] = 1
                        leaf_mask = gaussians.nodes[gaussian_indices.cpu(), 2] == 0
                        colors_precomp[leaf_mask.cuda(), :] = 1
                    else:
                        colors_precomp = None
                        
                    try:
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
                            #splat_args=splat_settings,
                            override_color = colors_precomp, 
                            sh_degree = gaussians.active_sh_degree,
                            )
                    except Exception:
                        print("\n" + "!"*30 + " ERROR FOUND " + "!"*30)
                        traceback.print_exc()
                        sys.exit(1)
                    
                    image = render_pkg["render"]
                    frame_time = clock()
                    if replay:
                        replay_stats["VRAM"].append(torch.cuda.max_memory_allocated())
                        replay_stats["frame_time"].append(frame_time)
                        replay_stats["cut_time"].append(cut_time)
                        replay_stats["number_rendered"].append(len(gaussian_indices))
                    if replay:
                        print("STATS")
                        #print(replay_stats["VRAM"][-1])
                        #print(torch.tensor(replay_stats["frame_time"][5:]).mean())
                        #print(torch.tensor(replay_stats["frame_time"]).max())
                        #print(torch.tensor(replay_stats["cut_time"]).mean())
                        #print(torch.tensor(replay_stats["cut_time"]).max())
                    ####### RENDER
                    if viewer_options["show_occlusion"]:
                        image=occlusion_image
                    
                    if replay:
                        torchvision.utils.save_image(image, f"CameraPaths/{camera_path_id}/frame_{iteration}.png")
                        iteration += 1
                    else:
                        net_image = image.cpu()
                        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().to('cpu').numpy())
                if not replay:
                    train_params = {"Num_Rendered" : len(gaussian_indices), "Number_of_SPTs" : len(SPT_indices), "Percentage_Rendered" : len(gaussian_indices)/gaussians.size, "Percentage_SPTs" : len(SPT_indices)/len(gaussians.SPT_starts)}
                    network_gui.send(net_image_bytes, json.dumps({"iteration" : 99, "num_gaussians" : gaussians.size, "loss" : 0, "sh_degree":1, "error" : 0, "paused" : False, "train_params" : train_params})) #dataset.source_path)
                    if do_training and ((iteration < int(opt.iterations)) or not keep_alive_):
                        break
            except EOFError as e:
                if replay:
                        print(replay_stats["VRAM"][-1])
                        print(torch.tensor(replay_stats["frame_time"]).mean())
                        print(torch.tensor(replay_stats["cut_time"]).mean())
                        with open("render_timings.pkl", "wb") as file:
                            pickle.dump(replay_stats["frame_time"], file)
                print(e)
                raise e
                network_gui.conn = None
######## VIEWER

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
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    #parser.add_argument("--hierarchy_path", type=str, default = None)
    # Re-render a recorded camera path
    parser.add_argument('--replay', type=bool, default=False)
    # If replay is true, specify the camera path id to replay
    parser.add_argument('--ID', type=int, default=0)
    
    parser.add_argument('--config', default="")

    
    args = parser.parse_args(sys.argv[1:])
    
    args.source_path = "utils/dummy_data"
        
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    with open(f"configs/{args.config}", "r") as f:
        data = json.load(f)
    config = argparse.Namespace(**data)
    optimization_params = op.extract(config)
    # Start GUI server, configure and run training
    render(lp.extract(args), optimization_params, pp.extract(args), args.replay, args.ID)

    print("\nTraining complete.")
