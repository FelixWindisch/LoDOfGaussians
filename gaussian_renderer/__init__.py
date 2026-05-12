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

import torch
from torch import nn
import math

from utils.sh_utils import eval_sh
import numpy as np
import torchvision
from gsplat import rasterization
from globals import *

def occlusion_cull(indices, gaussians, camera, pipe, background, opacity_multiplier = 1, scale_multiplier = 1):
    features_rest2 = 14 + number_SH_properties[gaussians.max_sh_degree] * 3
    SH_properties_single = number_SH_properties[gaussians.max_sh_degree]
    means3D = gaussians.properties[indices, xyz1:xyz2].cuda().contiguous()
    opacity = torch.clamp(gaussians.opacity_activation(gaussians.properties[indices, opacity1].cuda().contiguous()) * opacity_multiplier, 0, 1)
    scales = gaussians.scaling_activation(gaussians.properties[indices, scales1:scales2].cuda().contiguous()) * scale_multiplier
    rotations = gaussians.rotation_activation(gaussians.properties[indices, rotation1:rotation2].cuda().contiguous())
    features_dc = gaussians.properties[indices, features1:features2].cuda().contiguous()
    features_rest = gaussians.properties[indices, features_rest1:features_rest2].cuda().reshape(len(indices), SH_properties_single, 3).contiguous()
    #shs = torch.cat((features_dc, features_rest), dim=1).contiguous()
    render_pkg = render_vanilla(camera, means3D, opacity, scales, rotations, features_dc, features_rest, pipe, background)
    #torchvision.utils.save_image(render_pkg["render"], "occlusion.png")
    return render_pkg["contribution"] > 0, render_pkg["render"]

def occlusion_cull_cached(gaussians, camera, pipe, background, opacity_multiplier = 1, scale_multiplier = 1):
    #shs = torch.cat((features_dc, features_rest), dim=1).contiguous()
    render_pkg = render_vanilla(camera, gaussians.SPT_means3D, gaussians.SPT_opacity, gaussians.SPT_scales, gaussians.SPT_rotations, gaussians.SPT_features_dc, gaussians.SPT_features_rest, pipe, background)
    #torchvision.utils.save_image(render_pkg["render"], "occlusion.png")
    return render_pkg["contribution"] > 0.00, render_pkg["render"]





def render_gsplat(viewpoint_camera, 
        means3D,
        opacity,
        scales, 
        rotations,
        dc, shs,   pipe, bg_color : torch.Tensor, sh_degree=3, scaling_modifier = 1.0, override_color = None, use_trained_exp=False, anti_aliasing=True,gaussians=None, use_depth=False):
    W = int(viewpoint_camera.image_width)
    H = int(viewpoint_camera.image_height)

    # Extract focal lengths from FoV
    fx = W / (2.0 * math.tan(viewpoint_camera.FoVx * 0.5))
    fy = H / (2.0 * math.tan(viewpoint_camera.FoVy * 0.5))
    cx, cy = W / 2.0, H / 2.0

    # Intrinsics Matrix (K) -> Shape: (1, 3, 3)
    K = torch.tensor([
        [fx, 0., cx],
        [0., fy, cy],
        [0., 0.,  1.]
    ], dtype=torch.float32, device="cuda").unsqueeze(0)

    # Extrinsics Matrix (viewmat) -> Shape: (1, 4, 4)
    # INRIA stores W2C transposed (so p_view = p_world @ W2C_inria). 
    # gsplat expects standard math formulation (p_view = W2C_gsplat @ p_world).
    viewmat = viewpoint_camera.world_view_transform.transpose(0, 1).unsqueeze(0)

    # Background color -> Shape: (1, 3)
    bg = bg_color.squeeze()

    # ---------------------------------------------------------------------------
    # 2. Format Gaussian Parameters
    # ---------------------------------------------------------------------------
    # Opacities: gsplat expects 1D (N,) tensor. INRIA usually gives (N, 1)
    opacities_gsp = opacity.squeeze(-1) 

    # Colors / SHs
    if override_color is not None:
        colors_gsp = override_color
        sh_degree_gsp = 0 
    else:
        # INRIA separates DC (N, 1, 3) and SH rest (N, 15, 3). 
        # gsplat wants them concatenated (N, 16, 3)
        colors_gsp = torch.cat([dc, shs], dim=1) 
        sh_degree_gsp = sh_degree

    # ---------------------------------------------------------------------------
    # 3. Render using gsplat
    # ---------------------------------------------------------------------------
    # "RGB+ED" returns a 4-channel image: [R, G, B, Expected Depth]
    rendermode = "RGB+D" if use_depth else "RGB"
    renders, alphas, meta = rasterization(
        means=means3D,
        quats=rotations,         # gsplat and INRIA both natively use [w, x, y, z]
        scales=scales,
        opacities=opacities_gsp,
        colors=colors_gsp,
        viewmats=viewmat,
        Ks=K,
        width=W,
        height=H,
        sh_degree=sh_degree_gsp,
        render_mode="RGB",    # Fetches depth alongside RGB for invdepth
        backgrounds=bg,
        packed=True             # Set to True if using gsplat's sparse packing for extra speed
    )

    # ---------------------------------------------------------------------------
    # 4. Unpack Outputs to match your original Alt-INRIA format
    # ---------------------------------------------------------------------------
    # renders shape: (1, H, W, 4). alphas shape: (1, H, W, 1)

    # Slice RGB and permute to (3, H, W)
    rendered_image = renders[0, ..., :3].permute(2, 0, 1)

    # Slice Expected Depth, calculate inverse depth, and permute to (1, H, W)
    #depth = renders[0, ..., 3:4].permute(2, 0, 1)
    #invdepth = 1.0 / (depth + 1e-7)
    packed_indices = meta["gaussian_ids"]

    radii = meta["radii"]
    means2D = meta["means2d"]
    
#
    try:
        means2D.retain_grad()
    except:
        pass
    out = {
        "render": rendered_image,
        "viewspace_points": means2D,
        "radii": radii,
        "packed_indices": packed_indices
        #,"render_buffer_overhead" : render_buffer_overhead
        }
    
    if use_depth:
        out["depth"] = meta["depth"][0]
    
    return out

def render(
        viewpoint_camera, pc, 
        pipe, 
        bg_color : torch.Tensor, 
        scaling_modifier = 1.0, 
        override_color = None, 
        indices = None, 
        use_trained_exp=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    render_indices = torch.empty(0).int().cuda()
    parent_indices = torch.empty(0).int().cuda()
    interpolation_weights = torch.empty(0).float().cuda()
    num_siblings = torch.empty(0).int().cuda()

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        # This is false for render_coarse
        do_depth=True,
        render_indices=render_indices,
        parent_indices=parent_indices,
        interpolation_weights=interpolation_weights,
        num_node_kids=num_siblings
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    
    if indices is not None:
        means3D = means3D[indices].contiguous()
        means2D = means2D[indices].contiguous()
        shs = shs[indices].contiguous()
        opacity = opacity[indices].contiguous()
        scales = scales[indices].contiguous()
        rotations = rotations[indices].contiguous() 

    rendered_image, radii, depth_image = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    
    
    # This is missing in render_coarse
    if use_trained_exp:
        exposure = pc.get_exposure_from_name(viewpoint_camera.image_name)
        rendered_image = torch.matmul(rendered_image.permute(1, 2, 0), exposure[:3, :3]).permute(2, 0, 1) + exposure[:3, 3,   None, None]
    rendered_image = rendered_image.clamp(0, 1)
    # This is missing in render_coarse


    subfilter = radii > 0
    if indices is not None:
        vis_filter = torch.zeros(pc._xyz.size(0), dtype=bool, device="cuda")
        w = vis_filter[indices]
        w[subfilter] = True
        vis_filter[indices] = w
    else:
        vis_filter = subfilter

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "depth" : depth_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : vis_filter.nonzero().flatten().long(),
            "radii": radii[subfilter]}

def render_on_disk(
    viewpoint_camera, 
        means3D,
        opacity,
        scales, 
        rotations,
        shs,
        pipe, 
        bg_color : torch.Tensor, 
        scaling_modifier = 1.0, 
        override_color = None,
        sh_degree = 3
        ):
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    
    screenspace_points = torch.zeros_like(means3D, dtype=torch.float32, requires_grad=False, device="cuda") + 0
    means2D = nn.Parameter(screenspace_points)

    render_indices = torch.empty(0).int().cuda()
    parent_indices = torch.empty(0).int().cuda()
    interpolation_weights = torch.empty(0).float().cuda()
    num_node_siblings = torch.empty(0).int().cuda()
    colors_precomp = None
    cov3D_precomp = None
    
    pipe.debug = True
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier= scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=int(sh_degree),
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        render_indices=render_indices,
        parent_indices=parent_indices,
        interpolation_weights=interpolation_weights,
        num_node_kids=num_node_siblings,
        do_depth=False
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    

    
    rendered_image, seen, depth = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    
    rendered_image = rendered_image.clamp(0, 1)
    #radii = radii[100000:]
    #vis_filter = radii > 0

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            #"visibility_filter" : vis_filter,
            #"radii": radii[vis_filter]
            "seen" : seen}



