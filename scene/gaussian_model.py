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

import json
import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from gaussian_hierarchy._C import load_hierarchy, write_hierarchy, load_dynamic_hierarchy, write_dynamic_hierarchy, get_morton_indices, get_spt_cut_cuda
from scene.OurAdam import Adam
from utils.reloc_utils import compute_relocation_cuda
import random
import math
import time
from gaussian_renderer import occlusion_cull
hierarchy_node_depth = 0
hierarchy_node_parent = 1
hierarchy_node_child_count = 2
hierarchy_node_first_child = 3
hierarchy_node_next_sibling = 4
hierarchy_node_max_side_length = 5



Max_SH_Degree = 1

number_SH_properties = [0, 3, 8, 15]
SH_properties_single = number_SH_properties[Max_SH_Degree] 
SH_properties = number_SH_properties[Max_SH_Degree] * 3

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
features_rest2 = 14 + SH_properties
number_properties = features_rest2


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


class GaussianModel:
    def extract_frustum_planes(self, view_proj_matrix):
        """
        Extract six frustum planes from the view-projection matrix.

        Args:
            view_proj_matrix (torch.Tensor): (4,4) matrix

        Returns:
            torch.Tensor: (6, 4) tensor where each row is a plane (nx, ny, nz, d)
        """
        m = view_proj_matrix.T
        planes = torch.stack([
            m[3] + m[0],  # Left
            m[3] - m[0],  # Right
            m[3] + m[1],  # Bottom
            m[3] - m[1],  # Top
            #m[3] + m[2],  # Near
            #m[3] - m[2]   # Far
        ])

        # Normalize planes (avoid issues with unnormalized normals)
        planes /= torch.norm(planes[:, :3], dim=1, keepdim=True)

        return planes  # (6, 4)

    def frustum_cull_spheres(self, points, radii, planes):
        """
            centers (torch.Tensor): (N, 3) tensor of sphere centers.
            radii (torch.Tensor): (N,) tensor of sphere radii.
            view_proj_matrix (torch.Tensor): (4,4) view-projection matrix.
        """
        
        normals = planes[:, :3]  # (6, 3)
        distances = planes[:, 3]  # (6,)
        visible = torch.ones(len(points), dtype = torch.bool)
        # Compute signed distance from sphere centers to each plane
        #TODO: Replace this for loop by pytorch
        for normal, distance in zip(normals, distances):
            signed_distances = torch.sum(points  * normal, dim=1) + distance
            visible[signed_distances + radii < 0] = False
        #signed_distances = torch.einsum("ij,kj->ik", normals, centers).transpose(0, 1) + distances  # (N, 6)

        # Check if any part of the sphere is inside or touching a plane
        #intersects = signed_distances >= 0 #-radii[:, None]  # (N, 6) boolean array

        # A sphere is visible if it intersects at least one frustum plane
        #visible = intersects.all(dim=1)  # (N,) boolean array

        return visible
    
    
    
    
    
    def get_SPT_cut(self, camera_position, full_proj_transform, target_granularity, camera, pipe, use_bouding_spheres=True, use_frustum_culling=True, use_occlusion_culling = False):
        # Scale factor decides strictness of frustum culling
        if use_bouding_spheres:
            bounds = self.bounding_sphere_radii
        else: 
            bounds = (self.scaling_activation(torch.max(self.upper_tree_scaling, dim=-1)[0]) * 3.0)
        #max_scales = self.scaling_activation(torch.max(self.upper_tree_scaling, dim=-1)[0]) * 3.0
        planes = self.extract_frustum_planes(full_proj_transform)
        #visible = gaussians.frustum_cull_spheres(gaussians._xyz[render_indices].cuda(), max_scales, planes)
        if use_frustum_culling:
            cull = lambda indices : self.frustum_cull_spheres(self.upper_tree_xyz[indices], bounds[indices], planes)
        else:
            cull = lambda indices : torch.ones(len(indices), dtype = torch.bool)
        
        # Detail level is sufficient (return 0 for cut) to cut if the closest distance at which we can view the Gaussian is less than the distance to camera
        LOD_detail_cut = lambda indices : self.min_distance_squared[indices] > (camera_position - self.upper_tree_xyz[indices]).square().sum()
        LOD_detail_cut = lambda indices : torch.ones_like(indices, dtype=torch.bool)
        coarse_cut = self.cut_hierarchy_on_condition(self.upper_tree_nodes, LOD_detail_cut, return_upper_tree=False, root_node=0, leave_out_of_cut_condition=cull)

        # occlusion cull
        if use_occlusion_culling:
            bg_color = [0, 0, 0]
            background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
            temp = len(coarse_cut)
            occlusion_indices = self.upper_tree_nodes[coarse_cut, hierarchy_node_max_side_length]
            occlusion_mask = occlusion_cull(occlusion_indices.cpu(), self, camera, pipe, background).cuda()
            coarse_cut = coarse_cut[occlusion_mask]
            print(f"Occlusion Cull {temp - len(coarse_cut)} out of {temp} upper tree gaussians")
        
        
        leaf_mask = self.upper_tree_nodes[coarse_cut, hierarchy_node_child_count] == 0
        leaf_nodes = coarse_cut[leaf_mask]
        
        SPT_indices = self.upper_tree_nodes[leaf_nodes][self.upper_tree_nodes[leaf_nodes, hierarchy_node_first_child] >= 0, hierarchy_node_first_child]
        
        SPT_node_indices = leaf_nodes[self.upper_tree_nodes[leaf_nodes, hierarchy_node_first_child] >= 0]
        
        print(f"Cut {len(SPT_indices)} out of {len(self.SPT)} SPTs")
        #clock()
        cut_indices = [torch.arange(0, self.skybox_points, device='cuda', dtype=torch.int32)]
        #
        SPT_distances = (self.upper_tree_xyz[SPT_node_indices] - camera_position).pow(2).sum(1).sqrt()
        
        dabadu, SPT_counts = get_spt_cut_cuda(len(SPT_indices), self.SPT_gaussian_indices, self.SPT_starts, self.SPT_max, self.SPT_min, SPT_indices, SPT_distances)
            #torch.cuda.empty_cache()
        cut_indices.append(dabadu)
        cut_indices.append(self.upper_tree_nodes[coarse_cut[~leaf_mask], hierarchy_node_max_side_length])
        return torch.cat(cut_indices), self.upper_tree_nodes[SPT_node_indices, hierarchy_node_max_side_length], SPT_distances
        #print(f"CUDA TIME: {clock()}")
        #clock()
        binary_search_indices=[]
        
        # Skybox is always rendered, as are hierarchy leafs that are not SPTS
        cut_indices = [torch.arange(0, self.skybox_points, device='cuda', dtype=torch.int32), 
                       self.upper_tree_nodes[coarse_cut[~leaf_mask], hierarchy_node_max_side_length]]
        for spt_index, leaf_node in zip(SPT_indices, SPT_node_indices):
            spt = self.SPT[spt_index]
            spt_center_distance = (self.upper_tree_xyz[leaf_node] - camera_position).pow(2).sum(0).sqrt()
            # Binary search for the first element in SPT the max distance is greater than center_distance
            min_index = 0
            max_index = len(spt)
            pivot = math.floor(max_index/2)
            while max_index-min_index > 1:
                if spt[pivot, 2] > spt_center_distance:
                    min_index = pivot
                else:
                    max_index = pivot
                pivot = math.floor((max_index+min_index)/2) 
            binary_search_indices.append(max_index)
            cut = torch.where(spt[:max_index, 1] < spt_center_distance)[0]
            cut_indices.append(spt[cut, 0].to(torch.int32))
        result = torch.cat(cut_indices)
        return result, None, None
        

    def build_hierarchical_SPT(self, SPT_Root_Volume, target_granularity, min_SPT_Size = 100, use_bounding_spheres=True, revive_gaussians = False):
        SPT_scale = SPT_Root_Volume
        device = self.properties.device
        
        condition = lambda indices : torch.prod(self.scaling_activation(self.properties[indices, scales1: scales2]), dim=-1) > SPT_scale
        upper_tree_indices, cut_indices = self.cut_hierarchy_on_condition(self.nodes, condition)
        
        
        
        bounding_sphere_radii = [] #torch.zeros_like(upper_tree_indices)
        self.SPT_starts = torch.zeros(1, device='cuda', dtype=torch.int32)
        self.SPT_min = torch.empty(0, device='cuda', dtype=torch.float32)
        self.SPT_max = torch.empty(0, device='cuda', dtype=torch.float32)
        self.SPT_gaussian_indices = torch.empty(0, device='cuda', dtype=torch.int32)

        #self.SPT = []
        SPT_indices = []
        leaf_spt_children = []
        # indices of SPT roots in global hierarchy
        SPT_root_hierarchy_indices = []
        for index, cut_node in enumerate(cut_indices):
            # do not build SPTs with only one element
            if self.nodes[cut_node, hierarchy_node_child_count] == 0:
                continue
            #create SPT
            SPT_center = self.properties[cut_node, xyz1:xyz2]
            SPT = torch.zeros(1, 3, device=device)
            SPT.requires_grad_(False)
            SPT[0, 0] = cut_node
            SPT[0, 1] = self.get_min_distance(cut_node, target_granularity)
            SPT[0, 2] = 1000000000000
            stack = torch.zeros(1, dtype=torch.int32, device = device)
            stack[0] = cut_node
            max_distances = torch.zeros(1, device=device)
            max_distances[0] = SPT[0,1]
            bounding_sphere_radius = torch.max(self.scaling_activation(self.properties[cut_node, scales1: scales2]), dim=-1)[0].item() * 3.0
            temp_additional_indices = torch.zeros(1, dtype=torch.int32, device=device)
            temp_additional_indices[0] = cut_node
            while len(stack) > 0:
                
                first_children = self.nodes[stack.to(device), hierarchy_node_first_child]
                second_children = self.nodes[first_children, hierarchy_node_next_sibling]

                stack = torch.cat((first_children, second_children)) 
                stack = stack[stack > 0]
                
                if len(stack) == 0:
                    break
                center_distances = torch.sqrt(torch.sum((self.properties[stack, xyz1:xyz2] - SPT_center) ** 2, dim=1))
                max_center_distances = torch.max(center_distances + torch.max(self.scaling_activation(self.properties[stack, scales1:scales2]), dim=-1)[0] * 3).item()
                bounding_sphere_radius = max(bounding_sphere_radius, max_center_distances)
                max_distances = max_distances[first_children > 0]
                stack_SPT = torch.zeros(len(stack), 3, device=device)
                stack_SPT[:, 0] = stack
                min_distances = self.get_min_distance(stack, target_granularity) + center_distances
                max_distances = torch.cat((max_distances, max_distances))
                
                
                ### Revive
                if(revive_gaussians):
                    revive_indices = torch.where(min_distances > max_distances)[0]
                    if len(revive_indices) > 0:
                        revive_current_SPT_distances = self.get_min_distance(stack[revive_indices], target_granularity)
                        revive_children1 = self.nodes[stack[revive_indices], hierarchy_node_first_child]
                        revive_children2 = self.nodes[revive_children1, hierarchy_node_next_sibling]
                        revive_children1_center_distances = torch.sqrt(torch.sum((self.properties[revive_children1, xyz1:xyz2] - SPT_center) ** 2, dim=1))
                        revive_children2_center_distances = torch.sqrt(torch.sum((self.properties[revive_children1, xyz1:xyz2] - SPT_center) ** 2, dim=1))
                        
                        max_child_min_distance = torch.maximum(self.get_min_distance(revive_children1, target_granularity, False) + revive_children1_center_distances, self.get_min_distance(revive_children2, target_granularity, False) + revive_children2_center_distances)
                        target_SPT_distances = (max_child_min_distance + max_distances[revive_indices]) / 2.0 - center_distances[revive_indices]
                        multiplier = target_SPT_distances / revive_current_SPT_distances
                        valid = torch.logical_and(self.get_surface_area(self.properties[stack[revive_indices], scales1:scales2]) > self.get_surface_area(self.properties[revive_children1, scales1:scales2]), \
                                                  self.get_surface_area(self.properties[stack[revive_indices], scales1:scales2]) > self.get_surface_area(self.properties[revive_children2, scales1:scales2])) 
                        
                        self.properties[stack[revive_indices][valid], scales1:scales2] = self.scaling_inverse_activation(self.scaling_activation(self.properties[stack[revive_indices][valid], scales1:scales2]) * multiplier[valid].unsqueeze(1).cpu())
                        min_distances[revive_indices] = target_SPT_distances
                        if self.properties[stack[revive_indices], scales1:scales2].isnan().any():
                            pass
                ### Revive
                
                stack_SPT[:, 1] = torch.where(min_distances < max_distances, min_distances, max_distances)
                stack_SPT[:, 2] = max_distances
                max_distances = stack_SPT[:, 1].clone()
                SPT = torch.cat((SPT, stack_SPT))
                temp_additional_indices = torch.cat((temp_additional_indices, stack))
            if len(SPT) > min_SPT_Size: 
                bounding_sphere_radii.append(bounding_sphere_radius)
                SPT = SPT.cuda()
                sort_indices = SPT[:,-1].argsort(dim=0, descending=True)
                SPT = SPT[sort_indices]
                leaf_spt_children.append(len(SPT_root_hierarchy_indices))
                #self.SPT.append(SPT)
                SPT_indices.append(cut_node)
                self.SPT_starts = torch.cat((self.SPT_starts, (self.SPT_starts[-1] + len(SPT)).unsqueeze(0)))
                self.SPT_min = torch.cat((self.SPT_min, SPT[:, 1]))
                self.SPT_max = torch.cat((self.SPT_max, SPT[:, 2]))
                self.SPT_gaussian_indices = torch.cat((self.SPT_gaussian_indices, temp_additional_indices.cuda()[sort_indices])) #torch.cat((self.SPT_gaussian_indices, SPT[:, 0].to(torch.int32)))
                SPT_root_hierarchy_indices.append(cut_node)
                #print(f"SPT {len(SPT_root_hierarchy_indices)} finished")
                #self.SPT_centers.append(torch.mean(self._xyz[SPT[:,0]].to(torch.int32)))
            else:
                upper_tree_indices = torch.cat((upper_tree_indices, temp_additional_indices[1:])) # SPT[:, 0].to(torch.int32)[1:]))
        
        upper_tree_indices = torch.sort(upper_tree_indices)[0]
        # make it contiguous for searchsorted()
        upper_tree_indices = upper_tree_indices.contiguous()
        self.upper_tree_nodes = self.nodes[upper_tree_indices].clone().cuda()

        cut_SPT_indices = torch.searchsorted(upper_tree_indices, torch.tensor(SPT_indices, device=device))
        
        # SPT Leaves store the SPT index in first_child
        self.upper_tree_nodes[cut_SPT_indices, hierarchy_node_child_count] = 0
        #self.upper_tree_nodes[cut_SPT_indices, hierarchy_node_first_child] = 0
        self.upper_tree_nodes[cut_SPT_indices, hierarchy_node_first_child] = torch.tensor(leaf_spt_children, dtype = torch.int32, device='cuda')
        
        self.upper_tree_xyz = self.properties[upper_tree_indices, xyz1:xyz2].cuda()
        # unactivated!
        self.upper_tree_scaling = self.properties[upper_tree_indices, scales1:scales2].cuda()
        
        self.upper_tree_nodes[:, hierarchy_node_max_side_length] = upper_tree_indices
        self.upper_tree_nodes[:, hierarchy_node_parent] = torch.searchsorted(upper_tree_indices.cuda(), self.upper_tree_nodes[:, hierarchy_node_parent])
        self.upper_tree_nodes[0, hierarchy_node_parent] = -1
        # Dont modify SPT leaves
        # TODO: Update to torch.range(0, len(self.upper_tree_nodes), device=device)
        non_leaf = ~torch.isin(torch.range(0, len(self.upper_tree_nodes)-1, device=device), cut_SPT_indices.clone().detach())
        # if it is a normal leaf node without SPT, set child to -1, otherwise translate the first_child from self.nodes to self.upper_tree_nodes index
        self.upper_tree_nodes[non_leaf, hierarchy_node_first_child] = torch.where(self.upper_tree_nodes[non_leaf, hierarchy_node_first_child] == 0, -1, torch.searchsorted(upper_tree_indices.cuda(), self.upper_tree_nodes[non_leaf, hierarchy_node_first_child]).to(torch.int32))
        
        
        first_sibling = self.upper_tree_nodes[:, hierarchy_node_next_sibling] > 0
        self.upper_tree_nodes[first_sibling, hierarchy_node_next_sibling] = torch.searchsorted(upper_tree_indices.cuda(), self.upper_tree_nodes[first_sibling, hierarchy_node_next_sibling]).to(torch.int32)
        
        # The squared min distance of the parent in the upper tree is used for granularity cutting the upper tree
        parents = self.upper_tree_nodes[self.upper_tree_nodes[:, hierarchy_node_parent], hierarchy_node_max_side_length]
        self.min_distance_squared = self.get_min_distance(parents.to(device), target_granularity).square().cuda()
        # The root can always be rendered
        self.min_distance_squared[0] = 1000000000000
        
        leaf_indices = torch.where(self.upper_tree_nodes[:, hierarchy_node_first_child] == -1)[0]
        if use_bounding_spheres:
            self.bounding_sphere_radii = torch.zeros_like(upper_tree_indices, device='cuda', dtype=torch.float32)
            self.bounding_sphere_radii[leaf_indices] = torch.max(self.scaling_activation(self.upper_tree_scaling[leaf_indices]), dim=-1)[0] * 3
            self.bounding_sphere_radii[cut_SPT_indices] = torch.tensor(bounding_sphere_radii, device ='cuda')
            # upward propagating bounding sphere radii
            level_indices = torch.where(self.upper_tree_nodes[:, hierarchy_node_child_count] == 0)[0]
            while level_indices.numel():
                parents = self.upper_tree_nodes[level_indices, hierarchy_node_parent]
                parents = parents[parents >= 0]
                first_children = self.upper_tree_nodes[parents, hierarchy_node_first_child]
                second_children = self.upper_tree_nodes[first_children, hierarchy_node_next_sibling]
                first_children_distance = (self.upper_tree_xyz[parents] - self.upper_tree_xyz[first_children]).square().sum(1).sqrt()
                second_children_distance = (self.upper_tree_xyz[parents] - self.upper_tree_xyz[second_children]).square().sum(1).sqrt()
                self.bounding_sphere_radii[parents] = torch.maximum(self.bounding_sphere_radii[first_children] + first_children_distance, self.bounding_sphere_radii[second_children] + second_children_distance)
                level_indices = parents
            
            
            self.bounding_sphere_radii = self.bounding_sphere_radii.cuda()
        #self.SPT = []
        return torch.tensor(SPT_root_hierarchy_indices)
        
        
        
        #self.upper_tree_nodes[upper_tree_cut_indices, hierarchy_node_first_child] = torch.range(0, len(upper_tree_cut_indices)-1, dtype=torch.int32, device='cuda')
        
    def get_surface_area(self, scales):
        return torch.sqrt(scales[:, 0] * scales[:, 1] + scales[:, 0] * scales[:, 2] + scales[:, 1] * scales[:, 2])
                
            
            
        
    def get_min_distance(self, nodes, target_granularity, leaves_are_negative_infinity = True):
        if nodes.numel() == 1:
            if self.nodes[nodes, hierarchy_node_child_count] == 0 and leaves_are_negative_infinity:
                return -1000000
            scales = self.scaling_activation(self.properties[nodes.item(), scales1:scales2])
            return torch.sqrt((scales[ 0] * scales[ 1] + scales[0] * scales[2] + scales[1] * scales[2]))/target_granularity
            
        leaves = self.nodes[nodes, hierarchy_node_child_count] == 0
        #return self.scaling_activation(torch.max(self._scaling[nodes], dim=-1)[0])/target_granularity
        scales = self.scaling_activation(self.properties[nodes, scales1:scales2])
        #ellipse surface
        
        min_distances = torch.sqrt(scales[:, 0] * scales[:, 1] + scales[:, 0] * scales[:, 2] + scales[:, 1] * scales[:, 2])/target_granularity
        if leaves_are_negative_infinity:
            min_distances[leaves] = -1000000000
        return min_distances
        
        
    def is_hierarchy_cut(self, hierarchy, cut_indices):
        _, cut = self.cut_hierarchy_on_condition( hierarchy, lambda indices : torch.isin(indices, cut_indices))
        return len(cut) == len(cut_indices)
    
    
    # Where condition
    def cut_hierarchy_on_condition(self, hierarchy, condition, return_upper_tree= True, root_node = 100000, leave_out_of_cut_condition = None):
        
        device = hierarchy.device
        
        stack = torch.zeros(1, dtype=torch.int32, device = device)
        stack[0] = root_node
        visited = 1
        if return_upper_tree:
            upper_tree = torch.empty(0, dtype=torch.int32, device = device)#stack.clone() 
        cut = torch.empty(0, dtype=torch.int32, device = device)
        while len(stack) > 0:
            if return_upper_tree:
                upper_tree = torch.cat((upper_tree, stack), dim=-1)
            
            
            if leave_out_of_cut_condition is not None:
                leave_out_mask = leave_out_of_cut_condition(stack)
                #print(f"Leave out {(leave_out_mask.sum())} out of {len(stack)}")
                stack = stack[leave_out_mask]
            #TODO Should this be 3 lines above?
            cut = torch.cat((cut, stack[hierarchy[stack, hierarchy_node_child_count] == 0]))
            stack = stack[hierarchy[stack, hierarchy_node_child_count] > 0]
            
            cut_mask = condition(stack)
            cut = torch.cat((cut, stack[~cut_mask]))
            stack = stack[cut_mask]
            
            
                
            #if include_in_cut_where_condition_is_false:
            #    cut = torch.cat((cut, stack[~cut_mask]))
            #stack = stack[cut_mask]
            
            first_children = hierarchy[stack, hierarchy_node_first_child]
            second_children = hierarchy[first_children, hierarchy_node_next_sibling]
            
            stack = torch.cat((first_children, second_children)) 
                        
            visited += len(stack)
        if return_upper_tree:
            return upper_tree, cut
        return cut
    
    # WAY TOO SLOW
    def revive_stuck_gaussians(self, target_granularity):
        stuck_indices = torch.where(self.SPT_min == self.SPT_max)[0]
        for index in stuck_indices:
            gaussian_index = self.SPT_gaussian_indices[index].cpu()
            child1 = self.nodes[gaussian_index, hierarchy_node_first_child].cuda()
            child2 = self.nodes[child1, hierarchy_node_next_sibling].cuda()
            max_SPT_child_distance = max(self.get_SPT_min_at_index(child1, target_granularity), self.get_SPT_min_at_index( child2, target_granularity)).cuda()
            
            if max_SPT_child_distance < self.SPT_min[index]:
                #resize Gaussian
                SPT_index = torch.searchsorted(self.SPT_starts, index) -1
                self_center_distance = (self._xyz[gaussian_index] - self._xyz[self.SPT_gaussian_indices[self.SPT_starts[SPT_index]]]).square().sum(-1).sqrt()
                current_min_distance = self.get_min_distance(torch.tensor(gaussian_index), target_granularity)
                target_min_distance = ((self.SPT_min[index] + max_SPT_child_distance)/2.0) - self_center_distance
                multiplier = target_min_distance / current_min_distance
                self._scaling[gaussian_index] = self.scaling_inverse_activation(self.scaling_activation(self._scaling[gaussian_index]) * multiplier.cpu())
                self.SPT_min[index] = ((self.SPT_min[index] + max_SPT_child_distance)/2.0)
                child1_index = torch.where(self.SPT_gaussian_indices == child1)[0]
                child2_index = torch.where(self.SPT_gaussian_indices == child2)[0]
                self.SPT_max[child1_index] = ((self.SPT_min[index] + max_SPT_child_distance)/2.0)
                self.SPT_max[child2_index] = ((self.SPT_min[index] + max_SPT_child_distance)/2.0)
                
                
                
    def get_SPT_min_at_index(self, gaussian_index, target_granularity):
        index = torch.where(self.SPT_gaussian_indices == gaussian_index)[0]
        SPT_index = torch.searchsorted(self.SPT_starts, index) - 1
        center_distance = (self._xyz[gaussian_index] - self._xyz[self.SPT_gaussian_indices[self.SPT_starts[SPT_index]].cpu()]).square().sum(-1).sqrt()
        return self.get_min_distance(torch.tensor(gaussian_index), target_granularity, False) + center_distance
    
    
    def get_leaf_cut(self):
        return torch.nonzero(self.nodes[:self.size, hierarchy_node_child_count] == 0, as_tuple=False).squeeze()
    
    def move_to_disk(self, max_number_of_gaussians):
        self.size = len(self._xyz)
        
        names = ["xyz", "f_dc", "scaling", "rotation", "opacity", "f_rest", "nodes"]
        new_tensors = []
        optimizer_state = {}
        tensors = [self._xyz, self._features_dc, self._scaling, self._rotation, self._opacity, self._features_rest, self.nodes]
        for name, tensor in zip(names, tensors):
            shape = list(tensor.size())
            shape[0] = max_number_of_gaussians
            shape = torch.Size(shape)
            file = torch.from_numpy(np.memmap(name + ".bin", dtype=tensor.cpu().detach().numpy().dtype, mode="w+", shape=shape))
            file[:len(tensor)] = tensor
            new_tensors.append(file)
            # Can we reduce the 
            exp_avgs = torch.from_numpy(np.memmap(name + "_exp_avgs.bin", dtype=np.float32, mode="w+", shape=shape))
            exp_avgs_sqs = torch.from_numpy(np.memmap(name + "_exp_avgs_sqs.bin", dtype=np.float32, mode="w+", shape=shape))
            optimizer_state[name] = {"exp_avgs" : exp_avgs, "exp_avgs_sqs" : exp_avgs_sqs}
        
        self._xyz = new_tensors[0]
        self._features_dc = new_tensors[1]
        self._scaling = new_tensors[2]
        self._rotation = new_tensors[3]
        self._opacity = new_tensors[4]
        self._features_rest = new_tensors[5]
        self.nodes = new_tensors[6]
        
        
        return optimizer_state
        

    
    
    
    def compact_gaussians(self, device, max_number_of_gaussians, densification, training=True, prune_unused_gaussians=True):
        if max_number_of_gaussians is None:
            max_number_of_gaussians = len(self._xyz)
        number_gaussian_properties = [14, 23, 38, 59]
        spherical_harmonics_properties = [0, 9, 24, 45][self.max_sh_degree]
        # Momentum parameters are required during training
        if training:
            number_gaussian_properties = number_gaussian_properties[self.max_sh_degree] * 3
        else:
            number_gaussian_properties = number_gaussian_properties[self.max_sh_degree]
        self.properties = torch.zeros((max_number_of_gaussians, number_gaussian_properties) , device=device)
        if densification == "classic":
            self._densification_criterium = torch.zeros(max_number_of_gaussians, device=device, dtype=torch.float)
        if prune_unused_gaussians:
            self._contributed = torch.zeros(max_number_of_gaussians, device=device, dtype=torch.bool)
        
        
        self.size = len(self._xyz)
        
        self._features_rest = self._features_rest[:, :spherical_harmonics_properties//3, :]
        tensors = [self._xyz,  self._scaling, self._rotation, self._features_dc.squeeze(), self._opacity, self._features_rest.reshape(self.size, spherical_harmonics_properties)]
        current_index = 0
        for index, tensor in enumerate(tensors):
            size = tensor[0].nelement()
            self.properties[:self.size, current_index:current_index+size] = tensor.cpu().detach()
            if index == 0:
                self._xyz = None
            if index == 1:
                self._scaling = None
            if index == 2:
                self._rotation = None
            if index == 3:
                self._features_dc = None
            if index == 4:
                self._opacity = None
            if index == 5:
                self._features_rest = None
            tensors[index] = None
            del tensor
            torch.cuda.empty_cache()
            current_index += size
            
            
        new_nodes = torch.zeros((max_number_of_gaussians, 6), device=device, dtype=torch.int32)    
        new_nodes[:len(self.nodes), :6] = self.nodes
        self.nodes = new_nodes
        
        
        del tensors
        self._xyz = None
        self._scaling= None
        self._rotation= None
        self._features_dc= None
        self._opacity= None
        self._features_rest= None
    
    
    
    # deprecated?
    def move_to_cpu(self, max_number_of_gaussians):
        self.size = len(self._xyz)
        
        names = ["xyz", "f_dc", "scaling", "rotation", "opacity", "f_rest", "nodes"]
        new_tensors = []
        optimizer_state = {}
        tensors = [self._xyz, self._features_dc, self._scaling, self._rotation, self._opacity, self._features_rest, self.nodes]
        for name, tensor in zip(names, tensors):
            shape = list(tensor.size())
            shape[0] = max_number_of_gaussians
            shape = torch.Size(shape)
            file = torch.zeros(shape, dtype=tensor.cpu().dtype )
            file[:len(tensor)] = tensor
            new_tensors.append(file)
            # Can we reduce the 
            exp_avgs = torch.zeros(shape, dtype=torch.float32)
            exp_avgs_sqs = torch.zeros(shape, dtype=torch.float32)
            optimizer_state[name] = {"exp_avgs" : exp_avgs, "exp_avgs_sqs" : exp_avgs_sqs}
        self._xyz = new_tensors[0]
        self._features_dc = new_tensors[1]
        self._scaling = new_tensors[2]
        self._rotation = new_tensors[3]
        self._opacity = new_tensors[4]
        self._features_rest = new_tensors[5]
        self.nodes = new_tensors[6]
        
        
        return optimizer_state
        
    
    # Start with a leaf cut and select a random subset of the leaf nodes. Then, it goes from the bottom level
    # upward, pushing the cut of all nodes in the random subset to the next level upward 
    # (but only if its sibling is also in the subset). 
    def get_random_cut(self, p):
        number_gaussians = len(self.nodes)-self.skybox_points
        cut = torch.zeros(len(self.nodes)).cuda()
        current_set = torch.where(self.nodes[:, hierarchy_node_child_count] == 0)[0]
        cut[current_set] = 1
        subset_size = int(math.floor(len(current_set)*p))
        subset = current_set[(torch.randperm(len(current_set))[:subset_size])]
        current_depth = torch.max(self.nodes[subset, hierarchy_node_depth])
        while(len(subset) > 0):
            depth_subset = subset[self.nodes[subset, hierarchy_node_depth] == current_depth]
            first_children = depth_subset[self.nodes[depth_subset, hierarchy_node_next_sibling] > 0]
            siblings = self.nodes[first_children, hierarchy_node_next_sibling]
            
            siblings_are_cut = cut[siblings] == 1
            first_children = first_children[siblings_are_cut]
            siblings = siblings[siblings_are_cut]
            
            parents = self.nodes[first_children, hierarchy_node_parent]
            cut[parents] = 1
            cut[first_children] = 0
            cut[siblings] = 0
            subset = torch.cat((parents, subset[self.nodes[subset, hierarchy_node_depth] < current_depth]))
            current_depth -= 1
        return torch.where(cut == 1)[0]
            
            
            
        
        
    def recompute_weights(self):
        ellipse_surface = self.get_scaling[self.skybox_points:, 0] * self.get_scaling[self.skybox_points:, 1] + self.get_scaling[self.skybox_points:, 0] * self.get_scaling[self.skybox_points:, 2] + self.get_scaling[self.skybox_points:, 1] * self.get_scaling[self.skybox_points:, 2]
        self.weights = ellipse_surface * self.get_opacity.squeeze()[self.skybox_points:]
        first_children = self.nodes[self.skybox_points:, hierarchy_node_first_child]
        first_children = first_children[first_children != 0]
        siblings = self.nodes[first_children, hierarchy_node_next_sibling]
        # TODO: Update for non-binary trees
        normalization = self.weights[siblings-self.skybox_points] + self.weights[first_children-self.skybox_points]
        self.weights[first_children-self.skybox_points] /= normalization
        self.weights[siblings-self.skybox_points] /= normalization
        self.weights[0] = 1
        self.weights = torch.cat((torch.full([self.skybox_points], -99).cuda(), self.weights))
        
    def sort_morton(self):
        #pass
        xyz = self._xyz[self.skybox_points:self.size].cuda()
        codes = torch.zeros_like(xyz[:, 0], dtype=torch.int64).cuda()
        get_morton_indices(xyz, torch.min(self._xyz[self.skybox_points:], dim=0)[0].cuda(), torch.max(self._xyz[self.skybox_points:], dim=0)[0].cuda(), codes)
        indices = torch.argsort(codes)
        indices = indices.to(torch.int).to(self._xyz.device)
        # Make sure the root node stays in place
        root_index = torch.where(indices==0)[0][0]
        indices[root_index] = indices[0]
        indices[0] = 0
        
        indices += self.skybox_points
        with torch.no_grad():
            self._opacity[indices] = self._opacity[self.skybox_points:self.size].clone().detach().requires_grad_(True)
            self._xyz[indices] = self._xyz[self.skybox_points:self.size].clone().detach().requires_grad_(True)
            self._scaling[indices] = self._scaling[self.skybox_points:self.size].clone().detach().requires_grad_(True)
            self._rotation[indices] = self._rotation[self.skybox_points:self.size].clone().detach().requires_grad_(True)
            self._features_dc[indices] = self._features_dc[self.skybox_points:self.size].clone().detach().requires_grad_(True)
            self._features_rest[indices] = self._features_rest[self.skybox_points:self.size].clone().detach().requires_grad_(True)
            self.nodes[self.skybox_points:self.size, hierarchy_node_parent] = indices[self.nodes[self.skybox_points:self.size, hierarchy_node_parent] - self.skybox_points]
            sibling_mask = self.nodes[:, hierarchy_node_next_sibling] > 0
            #sibling_mask_with_skybox = torch.cat((torch.zeros(self.skybox_points, dtype=torch.bool), sibling_mask))
            self.nodes[sibling_mask, hierarchy_node_next_sibling] = indices[self.nodes[sibling_mask, hierarchy_node_next_sibling] - self.skybox_points]
            child_mask = self.nodes[:, hierarchy_node_first_child] > 0
            #child_mask_with_skybox = torch.cat((torch.zeros(self.skybox_points, dtype=torch.bool), child_mask))
            self.nodes[child_mask, hierarchy_node_first_child] = indices[self.nodes[child_mask, hierarchy_node_first_child]- self.skybox_points]
            self.nodes[self.skybox_points, hierarchy_node_parent] = -1
            self.nodes[self.skybox_points, hierarchy_node_next_sibling] = 0
            self.nodes[indices] = self.nodes.clone()[self.skybox_points:self.size]
            pass
        print("Morton Sort Complete")
            
        
    def merge_gaussians(self, indices):
        ellipse_surface  = lambda scale: scale[0] * scale[1] + scale[0] * scale[2] + scale[1] * scale[2]
        weights = []
        for index in indices:
            weights.append(self.get_opacity[index] * ellipse_surface(self.get_scaling[index]))
        weights /= np.sum(weights)
        new_position = self._xyz[indices]*weights
        new_features_dc = self._features_dc[indices]*weights
        new_features_rest = self._features_rest[indices]*weights
        pos_differences = self._xyz[indices] - new_position
        

    def compute_bounding_sphere_divergence(self, samples=1000):
        diverged = 0
        for i in range(samples):
            random_node = random.randrange(1, len(self.nodes))
            bounding_sphere_radius = torch.max(self.get_scaling[random_node]).item()
            bounding_sphere_position = self._xyz[random_node].detach()
            parent = self.nodes[random_node, hierarchy_node_parent]
            parent_sphere_radius = torch.max(self.get_scaling[parent])
            parent_sphere_position = self._xyz[parent].detach()
            for j in range(100):
                while True:
                    # Generate a random point in a cube of side 2*radius centered at the origin
                    point = (torch.rand(3)-0.5)*2
                    if torch.linalg.vector_norm(point) <= 1:
                        if torch.linalg.vector_norm((point*bounding_sphere_radius+bounding_sphere_position)-parent_sphere_position) > parent_sphere_radius:
                            diverged += 1
                        break
        print(diverged)
        return diverged/(100*samples)
        
    
    def sanity_check_hierarchy(self, hierarchy = None, root=100000):
        visited=[]
        if hierarchy is None:
            hierarchy = self.nodes
        print("Commencing Sanity Check of Hierarchy")
        self.sanity_counter = 0
        def sanity_check_rec(node):
            visited.append(node)
            self.sanity_counter += 1
            if self.sanity_counter > len(self._xyz):
                print("Infinite Recursion")
            if hierarchy[node][hierarchy_node_child_count] == 0:
                return
            child_iterator = hierarchy[node][hierarchy_node_first_child]
            children = [child_iterator]
            while(hierarchy[child_iterator][hierarchy_node_next_sibling] != 0):
                child_iterator = hierarchy[child_iterator][hierarchy_node_next_sibling]
                children.append(child_iterator)
            if len(children) == 1:
                print(f"Error: Node {node} has single child")
            if len(children) > hierarchy[node][hierarchy_node_child_count]:
                print(f"Error: Parent has more children ({len(children)}) than expected ({hierarchy[node][hierarchy_node_child_count]})")
            if len(children) < hierarchy[node][hierarchy_node_child_count]:
                print(f"Error: Parent has less children ({len(children)}) than expected ({hierarchy[node][hierarchy_node_child_count]})")
            
            for child in children:
                if hierarchy[child][hierarchy_node_parent] != node:
                    print(f"Error: Siblings have different parents ({hierarchy[child][hierarchy_node_parent]} instead of {node})")
                if hierarchy[child][hierarchy_node_depth] != hierarchy[node][hierarchy_node_depth]+1:
                    print(f"Error: Child has depth {hierarchy[child][hierarchy_node_depth]}, but parent has depth {hierarchy[node][hierarchy_node_depth]}")
                #if torch.prod(self.get_scaling[child]).item() > (torch.prod(self.get_scaling[node]).item() * 1.1):
                #    print(f"Warning: Child has max scale {torch.prod(self.get_scaling[child]).item()}, but parent has max scale {torch.prod(self.get_scaling[node]).item()}")
                sanity_check_rec(child)
                
        sanity_check_rec(root)
        if self.sanity_counter != len(hierarchy):
            print(f"Error: Reached {self.sanity_counter} out of {len(self._xyz)} nodes by recursion")
        print(f"Finished Sanity Check of Hierarchy ( {self.sanity_counter} / {len(self._xyz)} nodes)")
        return visited

    def training_setup(self, training_args, our_adam=True):
       self.percent_dense = training_args.percent_dense
       # TODO: Remove?
       self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
       self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

       l = [
           {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
           {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
           {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
           {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
           {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
           {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
       ]

       if our_adam:
           self.optimizer = Adam(l, lr=0.0, eps=1e-15)
       else:
           self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
           
       if self.pretrained_exposures is None:
           self.exposure_optimizer = torch.optim.Adam([self._exposure])
       self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                   lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                   lr_delay_mult=training_args.position_lr_delay_mult,
                                                   max_steps=training_args.position_lr_max_steps)
       self.exposure_scheduler_args = get_expon_lr_func(training_args.exposure_lr_init, training_args.exposure_lr_final, lr_delay_steps=training_args.exposure_lr_delay_steps, lr_delay_mult=training_args.exposure_lr_delay_mult, max_steps=training_args.iterations)

       
    
    
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize




    def __init__(self, sh_degree : int):
        self.is_hierarchy = False
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        #self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.nodes = None
        self.boxes = None

        self.pretrained_exposures = None

        self.skybox_points = 0
        self.skybox_locked = True

        self.setup_functions()

    def get_skybox_indices(self):
        return torch.arange(0, self.skybox_points)
    
    def get_number_of_leaf_nodes(self):
        return torch.sum(self.nodes[:, hierarchy_node_child_count] == 0).item()
    
    def get_number_of_levels(self):
        return torch.max(self.nodes[:, hierarchy_node_depth]).item()
    
    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_exposure(self):
        return self._exposure

    def get_exposure_from_name(self, image_name):
        return self._exposure[self.exposure_mapping[image_name]]
        # return self._exposure

    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1


    def create_from_pcd(
            self, 
            pcd : BasicPointCloud, 
            cam_infos : int,
            spatial_lr_scale : float,
            skybox_points: int,
            scaffold_file: str,
            bounds_file: str,
            skybox_locked: bool):
        
        self.spatial_lr_scale = spatial_lr_scale

        xyz = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = torch.tensor(np.asarray(pcd.colors)).float().cuda()
        
        minimum,_ = torch.min(xyz, axis=0)
        maximum,_ = torch.max(xyz, axis=0)
        mean = 0.5 * (minimum + maximum)

        self.skybox_locked = skybox_locked
        if scaffold_file != "" and skybox_points > 0:
            print(f"Overriding skybox_points: loading skybox from scaffold_file: {scaffold_file}")
            skybox_points = 0
        if skybox_points > 0:
            self.skybox_points = skybox_points
            radius = torch.linalg.norm(maximum - mean)

            theta = (2.0 * torch.pi * torch.rand(skybox_points, device="cuda")).float()
            phi = (torch.arccos(1.0 - 1.4 * torch.rand(skybox_points, device="cuda"))).float()
            skybox_xyz = torch.zeros((skybox_points, 3))
            skybox_xyz[:, 0] = radius * 10 * torch.cos(theta)*torch.sin(phi)
            skybox_xyz[:, 1] = radius * 10 * torch.sin(theta)*torch.sin(phi)
            skybox_xyz[:, 2] = radius * 10 * torch.cos(phi)
            skybox_xyz += mean.cpu()
            xyz = torch.concat((skybox_xyz.cuda(), xyz))
            fused_color = torch.concat((torch.ones((skybox_points, 3)).cuda(), fused_color))
            fused_color[:skybox_points,0] *= 0.7
            fused_color[:skybox_points,1] *= 0.8
            fused_color[:skybox_points,2] *= 0.95

        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = RGB2SH(fused_color)
        features[:, 3:, 1:] = 0.0

        dist2 = torch.clamp_min(distCUDA2(xyz), 0.0000001)
        if scaffold_file == "" and skybox_points > 0:
            dist2[:skybox_points] *= 10
            dist2[skybox_points:] = torch.clamp_max(dist2[skybox_points:], 10) 
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((xyz.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        if scaffold_file == "" and skybox_points > 0:
            opacities = self.inverse_opacity_activation(0.02 * torch.ones((xyz.shape[0], 1), dtype=torch.float, device="cuda"))
            opacities[:skybox_points] = 0.7
        else: 
            opacities = self.inverse_opacity_activation(0.01 * torch.ones((xyz.shape[0], 1), dtype=torch.float, device="cuda"))

        features_dc = features[:,:,0:1].transpose(1, 2).contiguous()
        features_rest = features[:,:,1:].transpose(1, 2).contiguous()

        self.scaffold_points = None
        if scaffold_file != "": 
            scaffold_xyz, features_dc_scaffold, features_extra_scaffold, opacities_scaffold, scales_scaffold, rots_scaffold = self.load_ply_file(scaffold_file + "/point_cloud.ply", 1)
            scaffold_xyz = torch.from_numpy(scaffold_xyz).float()
            features_dc_scaffold = torch.from_numpy(features_dc_scaffold).permute(0, 2, 1).float()
            features_extra_scaffold = torch.from_numpy(features_extra_scaffold).permute(0, 2, 1).float()
            opacities_scaffold = torch.from_numpy(opacities_scaffold).float()
            scales_scaffold = torch.from_numpy(scales_scaffold).float()
            rots_scaffold = torch.from_numpy(rots_scaffold).float()

            with open(scaffold_file + "/pc_info.txt") as f:
                skybox_points = int(f.readline())

            self.skybox_points = skybox_points
            with open(os.path.join(bounds_file, "center.txt")) as centerfile:
                with open(os.path.join(bounds_file, "extent.txt")) as extentfile:
                    centerline = centerfile.readline()
                    extentline = extentfile.readline()

                    c = centerline.split(' ')
                    e = extentline.split(' ')
                    center = torch.Tensor([float(c[0]), float(c[1]), float(c[2])]).cuda()
                    extent = torch.Tensor([float(e[0]), float(e[1]), float(e[2])]).cuda()

            distances1 = torch.abs(scaffold_xyz.cuda() - center)
            selec = torch.logical_and(
                torch.max(distances1[:,0], distances1[:,1]) > 0.5 * extent[0],
                torch.max(distances1[:,0], distances1[:,1]) < 1.5 * extent[0])
            selec[:skybox_points] = True

            self.scaffold_points = selec.nonzero().size(0)

            xyz = torch.concat((scaffold_xyz.cuda()[selec], xyz))
            features_dc = torch.concat((features_dc_scaffold.cuda()[selec,0:1,:], features_dc))

            filler = torch.zeros((features_extra_scaffold.cuda()[selec,:,:].size(0), 15, 3))
            filler[:,0:3,:] = features_extra_scaffold.cuda()[selec,:,:]
            features_rest = torch.concat((filler.cuda(), features_rest))
            scales = torch.concat((scales_scaffold.cuda()[selec], scales))
            rots = torch.concat((rots_scaffold.cuda()[selec], rots))
            opacities = torch.concat((opacities_scaffold.cuda()[selec], opacities))

        self._xyz = nn.Parameter(xyz.requires_grad_(True))
        self._features_dc = nn.Parameter(features_dc.requires_grad_(True))
        self._features_rest = nn.Parameter(features_rest.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        #self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        self.exposure_mapping = {cam_info.image_name: idx for idx, cam_info in enumerate(cam_infos)}

        exposure = torch.eye(3, 4, device="cuda")[None].repeat(len(cam_infos), 1, 1)
        self._exposure = nn.Parameter(exposure.requires_grad_(True))
        print("Number of points at initialisation : ", self._xyz.shape[0])

    def init_exposure_optimization(self, training_args, cam_infos):
        self.pretrained_exposures = None
        self.exposure_mapping = {cam_info.image_name: idx for idx, cam_info in enumerate(cam_infos)}
        
        exposure = torch.eye(3, 4, device="cuda")[None].repeat(len(cam_infos), 1, 1)
        self._exposure = nn.Parameter(exposure.requires_grad_(True))
        self.exposure_optimizer = torch.optim.Adam([self._exposure])
        self.exposure_scheduler_args = get_expon_lr_func(training_args.exposure_lr_init, training_args.exposure_lr_final, lr_delay_steps=training_args.exposure_lr_delay_steps, lr_delay_mult=training_args.exposure_lr_delay_mult, max_steps=training_args.iterations)
        
       
    def load_ply_file(self, path, degree):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        return xyz, features_dc, features_extra, opacities, scales, rots


        
        
    
    # scaffold file is only required for number of skybox points
    def create_from_hier(self, path, spatial_lr_scale : float, scaffold_file : str, device="cpu"):
        global SH_properties, Max_SH_Degree, features_rest2
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.is_hierarchy = True
        self.spatial_lr_scale = spatial_lr_scale
        xyz, shs_all, alpha, scales, rots, nodes = load_dynamic_hierarchy(path)
        # set first child to 0 for all nodes that do not have children (because this is fucked up in some hierarchy files)
        SH_mapping = {4: 1, 9: 2, 16: 3}
        if self.max_sh_degree is None:
            self.max_sh_degree = SH_mapping[shs_all.shape[1]]
        self.active_sh_degree = self.max_sh_degree
        Max_SH_Degree = self.max_sh_degree
        SH_properties = number_SH_properties[Max_SH_Degree] * 3
        features_rest2 = 14 + SH_properties
        
        
        base = os.path.dirname(path)

      
        #retrieve exposure
        
        
        if scaffold_file:
            #retrieve skyboxp
            skybox_points = 100_000
            self.skybox_points = 100_000
            if scaffold_file != "":
                scaffold_xyz, features_dc_scaffold, features_extra_scaffold, opacities_scaffold, scales_scaffold, rots_scaffold = self.load_ply_file(scaffold_file + "/point_cloud.ply", 1)
                scaffold_xyz = torch.from_numpy(scaffold_xyz).float()
                features_dc_scaffold = torch.from_numpy(features_dc_scaffold).permute(0, 2, 1).float()
                features_extra_scaffold = torch.from_numpy(features_extra_scaffold).permute(0, 2, 1).float()
                opacities_scaffold = torch.from_numpy(opacities_scaffold).float()
                scales_scaffold = torch.from_numpy(scales_scaffold).float()
                rots_scaffold = torch.from_numpy(rots_scaffold).float()
                skybox_points = 100_000
                with open(scaffold_file + "/pc_info.txt") as f:
                    skybox_points = int(f.readline())
                    
                self.skybox_points = skybox_points
    
            # add skybox points to the front of array
            if self.skybox_points > 0:
                if scaffold_file != "":
                    skybox_xyz, features_dc_sky, features_rest_sky, opacities_sky, scales_sky, rots_sky = scaffold_xyz[:skybox_points], features_dc_scaffold[:skybox_points], features_extra_scaffold[:skybox_points], opacities_scaffold[:skybox_points], scales_scaffold[:skybox_points], rots_scaffold[:skybox_points]
    
                opacities_sky = torch.sigmoid(opacities_sky)
                xyz = torch.cat((skybox_xyz, xyz))
                alpha = torch.cat((opacities_sky, alpha))
                scales = torch.cat((scales_sky, scales))
                rots = torch.cat((rots_sky, rots))
                filler = torch.zeros(features_dc_sky.size(0), 16, 3)
                filler[:, :1, :] = features_dc_sky
                filler[:, 1:4, :] = features_rest_sky
                shs_all = torch.cat((filler, shs_all))
            nodes[:, hierarchy_node_first_child] += self.skybox_points
            nodes[:, hierarchy_node_parent] += self.skybox_points
            

            nodes[nodes[:,hierarchy_node_next_sibling]>0, hierarchy_node_next_sibling] += self.skybox_points
            nodes[0, hierarchy_node_parent] = -1
            nodes = torch.cat((torch.full((self.skybox_points, 6), -99, dtype=torch.int32), nodes)) 
            nodes[skybox_points:, 3] = torch.where(nodes[skybox_points:, 2]==2, nodes[skybox_points:, 3], torch.zeros_like(nodes[skybox_points:,3]))

        self._xyz = xyz.to(device)
        self._features_dc = shs_all.to(device)[:,:1,:].requires_grad_(True)
        if self.max_sh_degree == 3:
            sh_coefficients = 15
        elif self.max_sh_degree == 2:
            sh_coefficients = 8
        elif self.max_sh_degree == 1:
            sh_coefficients = 3
        elif self.max_sh_degree == 0:
            sh_coefficients = 0
        self._features_rest = shs_all.to(device)[:,1:1+sh_coefficients,:].requires_grad_(True)
        self._opacity = alpha.to(device).requires_grad_(True)
        self._scaling = scales.to(device).requires_grad_(True)
        self._rotation =rots.to(device).requires_grad_(True)
        
        with torch.no_grad():
            self._opacity.clamp_(0, 0.99999)
            self._opacity = self.inverse_opacity_activation(self._opacity)
        #    self._xyz = nn.Parameter(xyz.to(device).requires_grad_(True))
        #    self._features_dc = nn.Parameter(shs_all.to(device)[:,:1,:].requires_grad_(True))
        #    self._features_rest = nn.Parameter(shs_all.to(device)[:,1:16,:].requires_grad_(True))
        #    self._opacity = nn.Parameter(alpha.to(device).requires_grad_(True))
        #    self._scaling = nn.Parameter(scales.to(device).requires_grad_(True))
        #    self._rotation = nn.Parameter(rots.to(device).requires_grad_(True))
        #self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        #self.opacity_activation = torch.abs
        #self.inverse_opacity_activation = torch.abs
        
        self.hierarchy_path = path
        # zero out the last column
        nodes[:, -1] = 0
        self.nodes = nodes.to(device)
        #self.boxes = boxes.to(device)

    def create_from_pt(self, path, spatial_lr_scale : float ):
        self.spatial_lr_scale = spatial_lr_scale

        xyz = torch.load(path + "/done_xyz.pt")
        shs_dc = torch.load(path + "/done_dc.pt")
        shs_rest = torch.load(path + "/done_rest.pt")
        alpha = torch.load(path + "/done_opacity.pt")
        scales = torch.load(path + "/done_scaling.pt")
        rots = torch.load(path + "/done_rotation.pt")

        self._xyz = nn.Parameter(xyz.cuda().requires_grad_(True))
        self._features_dc = nn.Parameter(shs_dc.cuda().requires_grad_(True))
        self._features_rest = nn.Parameter(shs_rest.cuda().requires_grad_(True))
        self._opacity = nn.Parameter(alpha.cuda().requires_grad_(True))
        self._scaling = nn.Parameter(scales.cuda().requires_grad_(True))
        self._rotation = nn.Parameter(rots.cuda().requires_grad_(True))
        #self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def save_hierarchy(self, output_path, file_name="opt"):
        print(f"Result hierarchy written to {output_path}/{file_name}")
        write_dynamic_hierarchy(output_path + f"/{file_name}",
                        self.properties[:self.size, xyz1:xyz2],
                        torch.cat((self.properties[:self.size, features1:features2], self.properties[:self.size, features_rest1:features_rest2]), 1),
                        self.opacity_activation(self.properties[:self.size, opacity1:opacity2]),
                        self.properties[:self.size, scales1:scales2],
                        self.properties[:self.size, rotation1:rotation2],
                        self.nodes[:self.size],
                        self.max_sh_degree)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''

        if self.pretrained_exposures is None:
            for param_group in self.exposure_optimizer.param_groups:
                param_group['lr'] = self.exposure_scheduler_args(iteration)
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(features2 - features1):
            l.append('f_dc_{}'.format(i))
        for i in range(features_rest2 - features_rest1):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(3):
            l.append('scale_{}'.format(i))
        for i in range(4):
            l.append('rot_{}'.format(i))
        return l

    def save_pt(self, path):
        mkdir_p(path)

        torch.save(self._xyz.detach().cpu(), os.path.join(path, "done_xyz.pt"))
        torch.save(self._features_dc.cpu(), os.path.join(path, "done_dc.pt"))
        torch.save(self._features_rest.cpu(), os.path.join(path, "done_rest.pt"))
        torch.save(self._opacity.cpu(), os.path.join(path, "done_opacity.pt"))
        torch.save(self._scaling, os.path.join(path, "done_scaling.pt"))
        torch.save(self._rotation, os.path.join(path, "done_rotation.pt"))

        import struct
        def load_pt(path):
            xyz = torch.load(os.path.join(path, "done_xyz.pt")).detach().cpu()
            features_dc = torch.load(os.path.join(path, "done_dc.pt")).detach().cpu()
            features_rest = torch.load( os.path.join(path, "done_rest.pt")).detach().cpu()
            opacity = torch.load(os.path.join(path, "done_opacity.pt")).detach().cpu()
            scaling = torch.load(os.path.join(path, "done_scaling.pt")).detach().cpu()
            rotation = torch.load(os.path.join(path, "done_rotation.pt")).detach().cpu()

            return xyz, features_dc, features_rest, opacity, scaling, rotation


        xyz, features_dc, features_rest, opacity, scaling, rotation = load_pt(path)

        my_int = xyz.size(0)
        with open(os.path.join(path, "point_cloud.bin"), 'wb') as f:
            f.write(struct.pack('i', my_int))
            f.write(xyz.numpy().tobytes())
            print(features_dc[0])
            print(features_rest[0])
            f.write(torch.cat((features_dc, features_rest), dim=1).numpy().tobytes())
            f.write(opacity.numpy().tobytes())
            f.write(scaling.numpy().tobytes())
            f.write(rotation.numpy().tobytes())


    def save_ply_coarse(self, path, only_leaves=False, indices=None):
        mkdir_p(os.path.dirname(path))
        if only_leaves:
            mask = torch.where(self.nodes[:self.size, hierarchy_node_child_count] == 0)[0]
        elif indices is not None:
            mask = indices
        else:
            # all true mask
            mask = torch.ones(len(self._opacity), dtype=torch.bool)
        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().unsqueeze(1).transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        #opacities = self.inverse_opacity_activation(self._opacity[mask]).detach().cpu().numpy()
        opacities = (self._opacity).detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
    
    def save_ply(self, path, only_leaves=False, indices=None):
        mkdir_p(os.path.dirname(path))
        if only_leaves:
            mask = torch.where(self.nodes[:self.size, hierarchy_node_child_count] == 0)[0]
        elif indices is not None:
            mask = indices
        else:
            # all true mask
            mask = torch.ones(len(self._opacity), dtype=torch.bool)
        xyz = self.properties[mask, xyz1:xyz2].detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self.properties[mask, features1:features2].detach().unsqueeze(1).transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self.properties[mask, features_rest1:features_rest2].detach().reshape((len(mask), 3,3)).transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        #opacities = self.inverse_opacity_activation(self._opacity[mask]).detach().cpu().numpy()
        opacities = (self.properties[mask, opacity1:opacity2]).detach().cpu().numpy()
        scale = self.properties[mask, scales1:scales2].detach().cpu().numpy()
        rotation = self.properties[mask, rotation1:rotation2].detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = torch.cat((self._opacity[:self.skybox_points], inverse_sigmoid(torch.min(self.get_opacity[self.skybox_points:], torch.ones_like(self.get_opacity[self.skybox_points:])*0.01))), 0)
        #opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        xyz, features_dc, features_extra, opacities, scales, rots = self.load_ply_file(path, self.max_sh_degree)

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors


    # removes the pruned gaussians from the optimizer and return tensors with the pruned gaussian properties removed
    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)
        
        
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        
        
        indices = torch.where(mask)
        self._xyz[indices] = 0
        self._scaling[indices] = 0
        self._opacity[indices] = 0
        #
        # nodes[indices]          
            #self.nodes = 
        

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors
    

    def densification_postfix_with_storage(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, reset_params=True):
        new_gaussians = new_xyz.size()[0]
        self.properties[self.size:self.size+new_gaussians, xyz1:xyz2] = new_xyz
        self.properties[self.size:self.size+new_gaussians, scales1:scales2] = new_scaling
        self.properties[self.size:self.size+new_gaussians, rotation1:rotation2] = new_rotation
        self.properties[self.size:self.size+new_gaussians, features1:features2] = new_features_dc
        self.properties[self.size:self.size+new_gaussians, opacity1: opacity2] = new_opacities
        self.properties[self.size:self.size+new_gaussians, features_rest1:features_rest2] = new_features_rest

        
        
    # reset_params is from MCMC
    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, reset_params=True):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        if reset_params:
            self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
            self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
            self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        #self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad * self.max_radii2D * torch.pow(self.get_opacity.flatten(), 1/5.0) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask, self.get_opacity.flatten() > 0.15)

        #selected_pts_mask = torch.logical_and(selected_pts_mask,
        #                                      torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
        # Only densify leaf nodes
        selected_pts_mask = torch.logical_and(selected_pts_mask, self.nodes[:,hierarchy_node_child_count] == 0)        
        
        print(f"Split {len(torch.where(selected_pts_mask)[0])} points")
        # No densification of the scaffold
        if self.scaffold_points is not None:
            selected_pts_mask[:self.scaffold_points] = False


        stds = self.get_scaling[selected_pts_mask].repeat_interleave(repeats=N,dim=0)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat_interleave(repeats=N,dim=0)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat_interleave(repeats=N,dim=0)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat_interleave(repeats=N,dim=0) / (0.7*N))
        new_rotation = self._rotation[selected_pts_mask].repeat_interleave(repeats=N,dim=0)
        new_features_dc = self._features_dc[selected_pts_mask].repeat_interleave(repeats=N,dim=0)
        new_features_rest = self._features_rest[selected_pts_mask].repeat_interleave(repeats=N,dim=0)
        new_opacity = self._opacity[selected_pts_mask].repeat_interleave(repeats=N,dim=0)
        #new_boxes = self._opacity[selected_pts_mask].repeat(N,1)
        
        new_nodes = torch.zeros((len(new_xyz), 6), dtype=torch.int32)
        for index, node in enumerate(torch.where(selected_pts_mask)[0]):
            full_index = len(self._xyz) + index*2
            self.nodes[node][hierarchy_node_child_count] = 2
            self.nodes[node][hierarchy_node_first_child] = full_index
            
            
            new_nodes[index*2][hierarchy_node_depth] = self.nodes[node][hierarchy_node_depth] + 1
            new_nodes[index*2][hierarchy_node_parent] = node
            new_nodes[index*2][hierarchy_node_child_count] = 0
            new_nodes[index*2][hierarchy_node_first_child] = -1
            new_nodes[index*2][hierarchy_node_next_sibling] = full_index + 1
            
            new_nodes[index*2+1][hierarchy_node_depth] = self.nodes[node][hierarchy_node_depth] + 1
            new_nodes[index*2+1][hierarchy_node_parent] = node
            new_nodes[index*2+1][hierarchy_node_child_count] = 0
            new_nodes[index*2+1][hierarchy_node_first_child] = -1
            new_nodes[index*2+1][hierarchy_node_next_sibling] = 0
        self.nodes = torch.cat((self.nodes, new_nodes.to('cuda')))

        
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)
        #prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        #self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent, N=2):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) * self.max_radii2D * torch.pow(self.get_opacity.flatten(), 1/5.0) >= grad_threshold, True, False)
        # Don't densify points that will be pruned
        selected_pts_mask = torch.logical_and(selected_pts_mask, self.get_opacity.flatten() > 0.15)
        # Don't densify points that will be pruned
        #selected_pts_mask = torch.logical_and(selected_pts_mask,
        #                                      torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        # No densification of the scaffold
        if self.scaffold_points is not None:
            selected_pts_mask[:self.scaffold_points] = False
        # Only densify leaf nodes
        selected_pts_mask = torch.logical_and(selected_pts_mask, self.nodes[:,hierarchy_node_child_count] == 0)

        
        print(f"Clone {len(torch.where(selected_pts_mask)[0])} points")
        new_xyz = self._xyz[selected_pts_mask].repeat_interleave(repeats=N,dim=0)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat_interleave(repeats=N,dim=0))
        new_features_dc = self._features_dc[selected_pts_mask].repeat_interleave(repeats=N,dim=0)
        new_features_rest = self._features_rest[selected_pts_mask].repeat_interleave(repeats=N,dim=0)
        new_opacities = self._opacity[selected_pts_mask].repeat_interleave(repeats=N,dim=0)
        
        new_rotation = self._rotation[selected_pts_mask].repeat_interleave(repeats=N,dim=0)


        new_nodes = torch.zeros((len(new_xyz), 6), dtype=torch.int32)
        for index, node in enumerate(torch.where(selected_pts_mask)[0]):
            full_index = len(self._xyz) + index*2
            self.nodes[node][hierarchy_node_child_count] = 2
            self.nodes[node][hierarchy_node_first_child] = full_index
            
            
            new_nodes[index*2][hierarchy_node_depth] = self.nodes[node][hierarchy_node_depth] + 1
            new_nodes[index*2][hierarchy_node_parent] = node
            new_nodes[index*2][hierarchy_node_child_count] = 0
            new_nodes[index*2][hierarchy_node_first_child] = -1
            new_nodes[index*2][hierarchy_node_next_sibling] = full_index + 1
            
            new_nodes[index*2+1][hierarchy_node_depth] = self.nodes[node][hierarchy_node_depth] + 1
            new_nodes[index*2+1][hierarchy_node_parent] = node
            new_nodes[index*2+1][hierarchy_node_child_count] = 0
            new_nodes[index*2+1][hierarchy_node_first_child] = -1
            new_nodes[index*2+1][hierarchy_node_next_sibling] = 0
        self.nodes = torch.cat((self.nodes, new_nodes.to('cuda')))

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)


    def densify(self, grads, grad_threshold, scene_extent, N=2):
        
        # Replace 2D gradients with 3D gradients
        #grads = self._xyz.grad
        
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) * self.max_radii2D * torch.pow(self.get_opacity.flatten(), 1/5.0) >= grad_threshold, True, False)
        # Don't densify points that will be pruned
        selected_pts_mask = torch.logical_and(selected_pts_mask, self.get_opacity.flatten() > 0.15)
        # Don't densify points that will be pruned
        #selected_pts_mask = torch.logical_and(selected_pts_mask,
        #                                      torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        # No densification of the scaffold
        if self.scaffold_points is not None:
            selected_pts_mask[:self.scaffold_points] = False
        # Only densify leaf nodes
        selected_pts_mask = torch.logical_and(selected_pts_mask, self.nodes[:,hierarchy_node_child_count] == 0)
        
        #selected_pts_mask = torch.logical_and(selected_pts_mask, torch.min(self._scaling, dim=1)[0] > -10)
        
        print(f"Densify {len(torch.where(selected_pts_mask)[0])} points")
        new_xyz = self._xyz[selected_pts_mask].repeat_interleave(repeats=N,dim=0)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat_interleave(repeats=N,dim=0) / (0.8*N))
        new_features_dc = self._features_dc[selected_pts_mask].repeat_interleave(repeats=N,dim=0)
        new_features_rest = self._features_rest[selected_pts_mask].repeat_interleave(repeats=N,dim=0)
        #new_opacities = self._opacity[selected_pts_mask].repeat_interleave(repeats=N,dim=0)
        new_opacities = self.inverse_opacity_activation(self.get_opacity[selected_pts_mask].repeat_interleave(repeats=N,dim=0) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat_interleave(repeats=N,dim=0)


        new_nodes = torch.zeros((len(new_xyz), 6), dtype=torch.int32)
        for index, node in enumerate(torch.where(selected_pts_mask)[0]):
            full_index = len(self._xyz) + index*2
            self.nodes[node][hierarchy_node_child_count] = 2
            self.nodes[node][hierarchy_node_first_child] = full_index
            
            
            new_nodes[index*2][hierarchy_node_depth] = self.nodes[node][hierarchy_node_depth] + 1
            new_nodes[index*2][hierarchy_node_parent] = node
            new_nodes[index*2][hierarchy_node_child_count] = 0
            new_nodes[index*2][hierarchy_node_first_child] = -1
            new_nodes[index*2][hierarchy_node_next_sibling] = full_index + 1
            
            new_nodes[index*2+1][hierarchy_node_depth] = self.nodes[node][hierarchy_node_depth] + 1
            new_nodes[index*2+1][hierarchy_node_parent] = node
            new_nodes[index*2+1][hierarchy_node_child_count] = 0
            new_nodes[index*2+1][hierarchy_node_first_child] = -1
            new_nodes[index*2+1][hierarchy_node_next_sibling] = 0
        self.nodes = torch.cat((self.nodes, new_nodes.to('cuda')))

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)
        print("")


    def densify_and_prune(self, max_grad, min_opacity, extent):
        grads = self.xyz_gradient_accum 
        grads[grads.isnan()] = 0.0

        self.densify(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if self.scaffold_points is not None:
            prune_mask[:self.scaffold_points] = False

        #self.prune_points(prune_mask)

        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter, indices, skybox_points = 100000):
        update_filter_indices = torch.where(update_filter)[0]
        update_indices = indices[update_filter_indices]
        #self.xyz_gradient_accum[update_indices] = torch.max(torch.norm(viewspace_point_tensor.grad[indices][update_filter,:2], dim=-1, keepdim=True), self.xyz_gradient_accum[update_indices])
        # I use xyz gradients here instead of screen space gradients
        self.xyz_gradient_accum[indices] = torch.max(torch.norm(viewspace_point_tensor.grad[skybox_points:skybox_points+len(indices)][:,:2], dim=-1, keepdim=True), self.xyz_gradient_accum[indices])
        #print(torch.sum(torch.abs(torch.max(torch.norm(viewspace_point_tensor.grad[indices][update_filter,:2], dim=-1, keepdim=True), self.xyz_gradient_accum[update_indices]))))
        #print(torch.max((self.xyz_gradient_accum[update_indices])))
        self.denom[indices] += 1
        


#region MCMC
    # sets momentum parameters for inds to 0
    def replace_tensors_to_optimizer(self, inds=None):
        tensors_dict = {"xyz": self._xyz,
            "f_dc": self._features_dc,
            "f_rest": self._features_rest,
            "opacity": self._opacity,
            "scaling" : self._scaling,
            "rotation" : self._rotation}
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            
            if inds is not None:
                stored_state["exp_avg"][inds] = 0
                stored_state["exp_avg_sq"][inds] = 0
            else:
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)
            del self.optimizer.state[group['params'][0]]
            group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
            self.optimizer.state[group['params'][0]] = stored_state
            optimizable_tensors[group["name"]] = group["params"][0]
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"] 
        # Thomas Bug
        torch.cuda.empty_cache()
        return optimizable_tensor
    
    def _update_params(self, idxs, ratio):
        new_opacity, new_scaling = compute_relocation_cuda(
            opacity_old=self.opacity_activation(self.properties[idxs, opacity1]).cuda(),
            scale_old=self.scaling_activation(self.properties[idxs, scales1:scales2]).cuda(),
            N=ratio[idxs, 0].cuda() + 1
        )
        new_opacity = torch.clamp(new_opacity.unsqueeze(-1), max=1.0 - 0.00000001 , min=0.005)
        new_opacity = self.inverse_opacity_activation(new_opacity)
        new_scaling = self.scaling_inverse_activation(new_scaling.reshape(-1, 3))
        return self.properties[idxs, xyz1:xyz2], self.properties[idxs, features1:features2], self.properties[idxs, features_rest1:features_rest2], new_opacity, new_scaling, self.properties[idxs, rotation1:rotation2]
    
    def _sample_alives(self, probs, num, alive_indices=None):
        probs = probs / (probs.sum() + torch.finfo(torch.float32).eps)
        sampled_idxs = torch.multinomial(probs, num, replacement=True)
        if alive_indices is not None:
            sampled_idxs = alive_indices[sampled_idxs]
        ratio = torch.bincount(sampled_idxs).unsqueeze(-1)
        return sampled_idxs, ratio
    
    def relocate_gs(self, dead_mask, size, storage_device='cpu', densification=False):
        if dead_mask.sum() == 0:
            return
        alive_mask = ~dead_mask
        dead_indices = dead_mask.nonzero(as_tuple=True)[0]
        alive_mask = torch.logical_and(alive_mask, self.nodes[:size, hierarchy_node_child_count] == 0)
        alive_indices = alive_mask.nonzero(as_tuple=True)[0]
        
        # sample from alive ones based on opacity
        # If a node and its sibling want to die, prevent one sibling from dying
        remove_siblings = self.nodes[dead_indices, hierarchy_node_next_sibling]
        dead_indices = dead_indices[~torch.isin(dead_indices, remove_siblings)]

        first_child_mask = self.nodes[dead_indices, hierarchy_node_next_sibling] > 0
        sibling_indices = torch.zeros_like(dead_indices, dtype=torch.int32)
        # if the dead node is the first child, write next sibling 
        sibling_indices[first_child_mask] = self.nodes[dead_indices[first_child_mask], hierarchy_node_next_sibling]
        # if the dead node is a second child, its sibling is the first child of the parent
        sibling_indices[~first_child_mask] = self.nodes[self.nodes[dead_indices[~first_child_mask], hierarchy_node_parent], hierarchy_node_first_child]



        alive_indices = alive_indices[~torch.isin(alive_indices, sibling_indices)]
        # Torch.multionmial can only handle 16_000_000 elements. If there are more possible respawn locations, uniformly sample 16M
        if len(alive_indices) > 16_000_000:
            alive_indices = alive_indices[torch.randperm(len(alive_indices))[:16_000_000]]
        probs = (self.opacity_activation(self.properties[alive_indices, opacity1])) 

        if densification == "classic":
            probs *= (self._densification_criterium[alive_indices] + 1)

        if 0 in sibling_indices:
            print("Found 0 Sibling!")
        # reinit_idx are the Gaussians where the dead Gaussians are respawned
        reinit_idx, ratio = self._sample_alives(alive_indices=alive_indices, probs=probs, num=dead_indices.shape[0]*2)
        reinit_idx = reinit_idx.unique()
        reinit_idx = reinit_idx[torch.randperm(len(reinit_idx))[:len(dead_indices)]]
        if len(reinit_idx) < len(dead_indices):
            print(f"Could only respawn {len(reinit_idx)} out of {len(dead_indices)} Gaussians")
            dead_indices = dead_indices[:len(reinit_idx)]
            sibling_indices = sibling_indices[:len(reinit_idx)]
        
        (
            self.properties[dead_indices, xyz1:xyz2], 
            self.properties[dead_indices, features1:features2],
            self.properties[dead_indices, features_rest1:features_rest2],
            new_opacity,
            new_scaling,
            self.properties[dead_indices, rotation1:rotation2] 
        ) = self._update_params(reinit_idx, ratio=ratio)
        
        new_opacity = new_opacity.to(storage_device)
        new_scaling = new_scaling.to(storage_device)
        self.properties[dead_indices, opacity1:opacity2] = new_opacity
        self.properties[dead_indices, scales1:scales2] = new_scaling
        
        # propogate the not-dead sibling to parent
        parent_indices = self.nodes[dead_indices, hierarchy_node_parent]
        for depth in range(torch.max(self.nodes[self.skybox_points:self.size, hierarchy_node_depth]), 0, -1):
            depth_mask = self.nodes[sibling_indices, hierarchy_node_depth] == depth
            sibling_depth = sibling_indices[depth_mask]
            parent_depth = parent_indices[depth_mask]
            self.properties[parent_depth, :number_properties] = self.properties[sibling_depth, :number_properties] 
            
            #self._xyz[parent_depth] = self._xyz[sibling_depth]
            #self._opacity[parent_depth] = self._opacity[sibling_depth]
            #self._features_dc[parent_depth] = self._features_dc[sibling_depth]
            #self._features_rest[parent_depth] = self._features_rest[sibling_depth]
            #self._scaling[parent_depth] = self._scaling[sibling_depth]
            #self._rotation[parent_depth] = self._rotation[sibling_depth]
        
        
            self.nodes[parent_depth, hierarchy_node_child_count] = self.nodes[sibling_depth, hierarchy_node_child_count]
            self.nodes[parent_depth, hierarchy_node_first_child] = self.nodes[sibling_depth, hierarchy_node_first_child]

            # propagate children of sibling upward
            first_children = self.nodes[sibling_depth, hierarchy_node_first_child] 
            self.nodes[first_children, hierarchy_node_parent] = parent_depth
            self.nodes[first_children, hierarchy_node_depth] = self.nodes[parent_depth, hierarchy_node_depth] + 1
            second_children = self.nodes[first_children, hierarchy_node_next_sibling] 
            self.nodes[second_children, hierarchy_node_parent] = parent_depth
            self.nodes[second_children, hierarchy_node_depth] = self.nodes[parent_depth, hierarchy_node_depth] + 1
        
         
        # respawn nodes now have 2 children
        self.nodes[reinit_idx, hierarchy_node_child_count] = 2
        self.nodes[reinit_idx, hierarchy_node_first_child] = dead_indices.to(torch.int32)
        
        # dead nodes get new parents from leaf nodes
        self.nodes[dead_indices, hierarchy_node_depth] = self.nodes[reinit_idx, 0] + 1
        self.nodes[dead_indices, hierarchy_node_parent] = reinit_idx.to(torch.int32)
        self.nodes[dead_indices, hierarchy_node_child_count] = 0
        self.nodes[dead_indices, hierarchy_node_first_child] = 0
        self.nodes[dead_indices, hierarchy_node_next_sibling] = sibling_indices
        
        self.nodes[sibling_indices, hierarchy_node_depth] = self.nodes[reinit_idx, 0] + 1
        self.nodes[sibling_indices, hierarchy_node_parent] = reinit_idx.to(torch.int32)
        self.nodes[sibling_indices, hierarchy_node_child_count] = 0
        self.nodes[sibling_indices, hierarchy_node_first_child] = 0
        self.nodes[sibling_indices, hierarchy_node_next_sibling] = 0
        
        # The sibling is a copy
        self.properties[sibling_indices, :number_properties] = self.properties[dead_indices, :number_properties] 

        
        # TODO: Implement Disk Equivalent
        #self.replace_tensors_to_optimizer(inds=sibling_indices)
        
        # Keep momentum for dead indices to encourage exploration?
        for name in ["xyz", "f_dc", "scaling", "rotation", "opacity", "f_rest", "nodes"]:
            self.properties[sibling_indices, number_properties:] = 0
            
    def add_new_gs(self, cap_max, size, densification, densify_percent = 1.05, densify_threshold = 0.01):
        device = self.properties.device
        target_num = min(cap_max, int(densify_percent * size))
        num_gs = max(0, target_num - size)
        if num_gs <= 0:
            return 0
        
        # Only Leaf nodes can be used for respawning 
        alive_indices=torch.where(self.nodes[:size, hierarchy_node_child_count] == 0)[0]
        
        #probs = (self.opacity_activation(self._opacity[alive_indices, 0])) 
        
        if densification == "classic":
            #probs *= (self._densification_criterium[alive_indices] + 1)
            #add_idx = alive_indices[torch.where(self._densification_criterium[alive_indices] > 0.001)]
            #ratio = torch.zeros((self.size, 1), device='cpu', dtype=torch.int32)
            add_idx = alive_indices[self._densification_criterium[alive_indices] > densify_threshold]
            if (len(add_idx) * 2) + self.size > cap_max:
                to_add = max(cap_max - self.size, 0) // 2
                add_idx = add_idx[: to_add]
        else:
            # Torch.multionmial can only handle 16_000_000 elements. If there are more possible respawn locations, uniformly sample 16M
            if len(alive_indices) > 16_000_000:
                alive_indices = alive_indices[torch.randperm(len(alive_indices))[:16_000_000]]
            probs = self.opacity_activation(self.properties[:size, opacity1:opacity2]).squeeze(-1)[alive_indices]
            add_idx, ratio = self._sample_alives(probs=probs, num=num_gs, alive_indices=alive_indices)
            # make sure respawn gaussians are unique
            add_idx = torch.where(ratio == 1)[0]
            if (len(add_idx) * 2) + self.size > cap_max:
                to_add = max(cap_max - self.size, 0) // 2
                add_idx = add_idx[: to_add]
        ratio = torch.zeros((self.size, 1), device=device, dtype=torch.int32)
        ratio[add_idx] = 1
        (   new_xyz, 
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation 
        ) = self._update_params(add_idx, ratio=ratio)
        
        print(f"Spawn {len(add_idx)} new Gaussians")
        new_xyz = new_xyz.repeat_interleave(repeats=2, dim=0)
        new_features_dc = new_features_dc.repeat_interleave(repeats=2, dim=0)
        new_features_rest = new_features_rest.repeat_interleave(repeats=2, dim=0)
        new_opacity = new_opacity.repeat_interleave(repeats=2, dim=0)
        new_scaling = new_scaling.repeat_interleave(repeats=2, dim=0)
        new_rotation = new_rotation.repeat_interleave(repeats=2, dim=0)

        new_opacity = new_opacity.to(device)
        new_scaling = new_scaling.to(device)
        
        
        new_nodes = torch.zeros((len(new_xyz), 6), dtype=torch.int32, device= device)     
        add_idx = add_idx.to(torch.int32)
        add_idx = add_idx.to(device)
        self.nodes[add_idx, hierarchy_node_child_count] = 2
        self.nodes[add_idx, hierarchy_node_first_child] = torch.arange(0, len(add_idx), dtype = torch.int32, device=device) * 2 + size
        
        index = torch.arange(0, len(add_idx), dtype = torch.int32, device=device) * 2
        index_plus_one = torch.arange(0, len(add_idx), dtype = torch.int32, device=device) * 2 + 1
        new_nodes[index, hierarchy_node_depth] = self.nodes[add_idx, hierarchy_node_depth] + 1
        new_nodes[index, hierarchy_node_parent] = add_idx
        new_nodes[index, hierarchy_node_child_count] = 0
        new_nodes[index, hierarchy_node_first_child] = 0
        new_nodes[index, hierarchy_node_next_sibling] = index + size + 1
        
        new_nodes[index_plus_one, hierarchy_node_depth] = self.nodes[add_idx, hierarchy_node_depth] + 1
        new_nodes[index_plus_one, hierarchy_node_parent] = add_idx
        new_nodes[index_plus_one, hierarchy_node_child_count] = 0
        new_nodes[index_plus_one, hierarchy_node_first_child] = 0
        new_nodes[index_plus_one, hierarchy_node_next_sibling] = 0
           
        
        
        self.nodes[size:size+new_nodes.size()[0]] = new_nodes
        self.densification_postfix_with_storage(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, reset_params=False)
        self.size += new_nodes.size()[0]

        
        # With no separation of storage / render cache:
        #    self.nodes = torch.cat((self.nodes, new_nodes.to(self.nodes.device)))  
        #    self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, reset_params=False)      
            
        return num_gs
#endregion