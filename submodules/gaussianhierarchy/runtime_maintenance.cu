/*
 * Copyright (C) 2024, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */


#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <thrust/sequence.h>
#include <thrust/device_vector.h>
#include <cuda.h>
#include <fstream>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
//#include <nvtx3/nvToolsExt.h>
#include <thrust/host_vector.h>
#include <tuple>
#include "types.h"
#include "runtime_maintenance.h"

__device__ int safeexc(int* data, int index)
{
	if (index == 0)
		return 0;
	return data[index - 1];
}

__global__ void rearrange(
	int N,
	int* active_nodes,
	int* new_active_nodes,
	const Node* nodes,
	const Box* boxes,
	const float3* pos_space,
	const float4* rot_space,
	const float* sh_space,
	const float* alpha_space,
	const float3* scale_space,
	int* split_space,
	Node* new_nodes,
	Box* new_boxes,
	float3* new_pos_space,
	float4* new_rot_space,
	float* new_sh_space,
	float* new_alpha_space,
	float3* new_scale_space,
	int* new_split_space,
	int* cuda2cpu_src,
	int* cuda2cpu_dst,
	int* node_indices,
	int* gaussian_indices)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if ((idx/15) >= N)
		return;

	int role = idx / N;
	idx = idx % N;
	bool actor = role == 0;

	int node_id = active_nodes[idx];
	int target_id = safeexc(node_indices, node_id);

	Node node = nodes[node_id];

	if (actor)
	{
		new_boxes[target_id] = boxes[node_id];
		new_split_space[target_id] = split_space[node_id];

		if (split_space[node_id] == 0) // Every unexpanded node is gone
			node.start_children = -1;
		split_space[node_id] = 0; // Clean up after yourself

		int new_parent = node.parent == -1 ? -1 : safeexc(node_indices, node.parent);
		int new_start_children = node.start_children == -1 ? -1 : safeexc(node_indices, node.start_children);

		node.parent = new_parent;
		node.start_children = new_start_children;
	}

	int new_start = safeexc(gaussian_indices, node_id);
	for (int i = 0; i < node.count_leafs + node.count_merged; i++)
	{
		int dst = new_start + i;
		int src = node.start + i;
		if (role == 0)
			new_pos_space[dst] = pos_space[src];
		else if (role == 1)
			new_rot_space[dst] = rot_space[src];
		else if (role >= 2 && role < 14)
			*(((float4*)(new_sh_space + dst * 48)) + role - 2) = *(((float4*)(sh_space + src * 48)) + role - 2);
		else
		{
			new_alpha_space[dst] = alpha_space[src];
			new_scale_space[dst] = scale_space[src];
		}
	}

	if (actor)
	{
		node.start = new_start;
		new_nodes[target_id] = node;
		new_active_nodes[idx] = idx;
		cuda2cpu_dst[target_id] = cuda2cpu_src[node_id];
	}
}

void Maintenance::reorder(
	int N,
	int* active_nodes,
	int* new_active_nodes,
	const int* nodes,
	const float* boxes,
	const float* pos_space,
	const float* rot_space,
	const float* sh_space,
	const float* alpha_space,
	const float* scale_space,
	int* split_space,
	int* new_nodes,
	float* new_boxes,
	float* new_pos_space,
	float* new_rot_space,
	float* new_sh_space,
	float* new_alpha_space,
	float* new_scale_space,
	int* new_split_space,
	int* cuda2cpu_src,
	int* cuda2cpu_dst,
	int* node_indices,
	int* gaussian_indices,
	void* streamy
)
{
	cudaStream_t stream = (cudaStream_t)streamy;
	int num_blocks = (N * 15 + 255) / 256;
	rearrange << <num_blocks, 256, 0, stream >> > (
		N,
		active_nodes,
		new_active_nodes,
		(Node*)nodes,
		(Box*)boxes,
		(float3*)pos_space,
		(float4*)rot_space,
		sh_space,
		alpha_space,
		(float3*)scale_space,
		split_space,
		(Node*)new_nodes,
		(Box*)new_boxes,
		(float3*)new_pos_space,
		(float4*)new_rot_space,
		new_sh_space,
		new_alpha_space,
		(float3*)new_scale_space,
		new_split_space,
		cuda2cpu_src,
		cuda2cpu_dst,
		node_indices,
		gaussian_indices
		);
}

__global__ void mark(
	int N,
	const int* indices,
	Node* nodes,
	int* nodes_count,
	int* gaussians_count)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= N)
		return;

	int node_id = indices[idx];
	Node node = nodes[node_id];
	nodes_count[node_id] = 1;
	gaussians_count[node_id] = node.count_merged + node.count_leafs;
}

__global__ void zero(int N, int* a, int* b)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= N)
		return;

	a[idx] = 0;
	b[idx] = 0;
}

void Maintenance::compactPart1(
	int topN,
	int N,
	int* active_nodes,
	int* new_active_nodes,
	const int* nodes,
	const float* boxes,
	const float* pos_space,
	const float* rot_space,
	const float* sh_space,
	const float* alpha_space,
	const float* scale_space,
	int* split_space,
	int* new_nodes,
	float* new_boxes,
	float* new_pos_space,
	float* new_rot_space,
	float* new_sh_space,
	float* new_alpha_space,
	float* new_scale_space,
	int* new_split_space,
	int* cuda2cpu_src,
	int* cuda2cpu_dst,
	int* NsrcI,
	int* NsrcI2,
	int* NdstI,
	int* NdstI2,
	char*& scratchspace,
	size_t& scratchspacesize,
	void* streamy,
	int* count
)
{
	cudaStream_t stream = (cudaStream_t)streamy;

	zero << <(topN + 255) / 256, 256, 0, stream >> > (topN, NsrcI, NsrcI2);
	mark << <(N + 255) / 256, 256, 0, stream >> > (N, active_nodes, (Node*)nodes, NsrcI, NsrcI2);

	cub::DeviceScan::InclusiveSum(scratchspace, scratchspacesize, NsrcI, NdstI, topN, stream);
	cudaMemcpyAsync(count, NdstI + topN - 1, sizeof(int), cudaMemcpyDeviceToHost, stream);
	cudaStreamSynchronize(stream);
}

__global__ void compressCUDA(
	int N,
	Node* nodes,
	float3* scales,
	float4* rots,
	float* shs,
	float* opacities
)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= N)
		return;

	Node node = nodes[idx];

	int g = node.start;

	int parent = g;
	if (node.parent != -1)
	{
		parent = nodes[node.parent].start;
	}

	__half2 mp;
	float* scale = (float*)&scales[g];
	float* pscale = (float*)&scales[parent];
	for (int i = 0; i < 3; i++)
	{
		mp.x = scale[i];
		mp.y = pscale[i];
		scale[i] = *((float*)&mp); // We are overwriting ourselves
	}
}

void Maintenance::compress(
	int N,
	int* nodes,
	float* scales,
	float* rots,
	float* shs,
	float* opacs
)
{
	int num_blocks = (N + 255) / 256;
	compressCUDA << <num_blocks, 256 >> > (N,
		(Node*)nodes,
		(float3*)scales,
		(float4*)rots,
		shs,
		opacs
		);
}

void Maintenance::compactPart2(
	int topN,
	int N,
	int* active_nodes,
	int* new_active_nodes,
	const int* nodes,
	const float* boxes,
	const float* pos_space,
	const float* rot_space,
	const float* sh_space,
	const float* alpha_space,
	const float* scale_space,
	int* split_space,
	int* new_nodes,
	float* new_boxes,
	float* new_pos_space,
	float* new_rot_space,
	float* new_sh_space,
	float* new_alpha_space,
	float* new_scale_space,
	int* new_split_space,
	int* cuda2cpu_src,
	int* cuda2cpu_dst,
	int* NsrcI,
	int* NsrcI2,
	int* NdstI,
	int* NdstI2,
	char*& scratchspace,
	size_t& scratchspacesize,
	void* streamy,
	int* count
)
{
	cudaStream_t stream = (cudaStream_t)streamy;

	cub::DeviceScan::InclusiveSum(scratchspace, scratchspacesize, NsrcI2, NdstI2, topN, stream);

	cudaMemcpyAsync(count, NdstI2 + topN - 1, sizeof(int), cudaMemcpyDeviceToHost, stream);
	cudaStreamSynchronize(stream);

	reorder(
		N,
		active_nodes,
		new_active_nodes,
		nodes,
		boxes,
		pos_space,
		rot_space,
		sh_space,
		alpha_space,
		scale_space,
		split_space,
		new_nodes,
		new_boxes,
		new_pos_space,
		new_rot_space,
		new_sh_space,
		new_alpha_space,
		new_scale_space,
		new_split_space,
		cuda2cpu_src,
		cuda2cpu_dst,
		NdstI,
		NdstI2,
		stream
	);
}

__global__ void setStarts(Node* nodes, int N, int* indices, int* starts)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= N)
		return;

	int parent_id = indices[idx];
	nodes[parent_id].start_children = starts[idx];
}

void Maintenance::updateStarts(
	int* nodes,
	int num_indices,
	int* indices,
	int* starts,
	void* streamy
)
{
	cudaStream_t stream = (cudaStream_t)streamy;
	int num_blocks = (num_indices + 255) / 256;
	setStarts << <num_blocks, 256, 0, stream >> > ((Node*)nodes, num_indices, indices, starts);
}