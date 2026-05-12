#pragma once
#include <torch/extension.h>
#include <cstdio>
#include <tuple>
#include <string>

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
LoadHierarchy(std::string filename);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> 
LoadDynamicHierarchy(std::string filename);




void WriteHierarchy(
    std::string filename,
    torch::Tensor &pos,
    torch::Tensor &shs,
    torch::Tensor &opacities,
    torch::Tensor &log_scales,
    torch::Tensor &rotations,
    torch::Tensor &nodes,
    torch::Tensor &boxes);

void WriteDynamicHierarchy(
    std::string filename,
    torch::Tensor &pos,
    torch::Tensor &shs,
    torch::Tensor &opacities,
    torch::Tensor &log_scales,
    torch::Tensor &rotations,
    torch::Tensor &nodes,
    int SH_degree);

torch::Tensor
ExpandToTarget(torch::Tensor& nodes, int target);

int ExpandToSize(
torch::Tensor& nodes, 
torch::Tensor& boxes, 
float size, 
torch::Tensor& viewpoint, 
torch::Tensor& viewdir, 
torch::Tensor& render_indices,
torch::Tensor& parent_indices,
torch::Tensor& nodes_for_render_indices);

int ExpandToSizeDynamic(
torch::Tensor& nodes, 
torch::Tensor& positions,
torch::Tensor& scales, 
float size, 
torch::Tensor& viewpoint, 
torch::Tensor& viewdir, 
torch::Tensor& render_indices,
torch::Tensor& parent_indices,
torch::Tensor& nodes_for_render_indices);

void GetTsIndexed(
torch::Tensor& indices,
float size,
torch::Tensor& nodes,
torch::Tensor& boxes,
torch::Tensor& viewpoint, 
torch::Tensor& viewdir, 
torch::Tensor& ts,
torch::Tensor& num_kids);

void GetTsIndexedDynamic(
torch::Tensor& indices,
float size,
torch::Tensor& nodes,
torch::Tensor& positions,
torch::Tensor& scales,
torch::Tensor& viewpoint, 
torch::Tensor& viewdir, 
torch::Tensor& ts,
torch::Tensor& num_kids);

//int GetHierarchyCut(
//torch::Tensor& nodes, 
//torch::Tensor& positions,
//torch::Tensor& scales, 
//float size, 
//torch::Tensor& viewpoint, 
//torch::Tensor& viewdir, 
//torch::Tensor& render_indices);

void GetMortonCode(torch::Tensor& xyz, torch::Tensor& min, torch::Tensor& max, torch::Tensor& codes);

std::tuple<torch::Tensor, torch::Tensor> GetSPTCut(
	int number_of_SPTs,
	torch::Tensor& gaussian_indices,
	torch::Tensor& SPT_starts,
	torch::Tensor& SPT_max,
	torch::Tensor& SPT_min,
	torch::Tensor& SPT_indices,
	torch::Tensor& SPT_distances);


std::tuple<torch::Tensor, torch::Tensor> ComputeRelocationCUDA(
	torch::Tensor& opacity_old,
	torch::Tensor& scale_old,
	torch::Tensor& N,
	torch::Tensor& binoms,
	const int n_max);