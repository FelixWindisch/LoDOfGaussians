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
#include <float.h>
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
#include "runtime_switching.h"


#define CHECK_CUDA(A, debug) \
A; if(debug) { \
auto ret = cudaDeviceSynchronize(); \
if (ret != cudaSuccess) { \
std::cerr << "\n[CUDA ERROR] in " << __FILE__ << "\nLine " << __LINE__ << ": " << cudaGetErrorString(ret); \
throw std::runtime_error(cudaGetErrorString(ret)); \
} \
}

__global__ void markTargetNodes(Node* nodes, int N, int target, int* node_counts)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= N)
		return;

	int count = 0;
	Node node = nodes[idx];
	if (node.depth > target)
		count = node.count_leafs;
	else if (node.parent != -1)
	{
		Node parentnode = nodes[node.parent];
		if (parentnode.depth > target)
		{
			count = node.count_leafs;
			if (node.depth != 0)
				count += node.count_merged;
		}
	}
	node_counts[idx] = count;
}


// render_offsets = inclusive sum
// node_count = render_count (this should be either 0 or 1)
__global__ void putRenderIndices(Node* nodes, int N, int* node_counts, int* node_offsets, int* render_indices, int* parent_indices = nullptr, int* nodes_for_render_indices = nullptr)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= N)
		return;

	Node node = nodes[idx];
	int count = node_counts[idx];
	int offset = idx == 0 ? 0 : node_offsets[idx - 1];
	int start = node.start;
	
	int parentgaussian = -1;
	if (node.parent != -1)
	{
		parentgaussian = nodes[node.parent].start;
	}

	for (int i = 0; i < count; i++)
	{
		render_indices[offset + i] = node.start + i;
		if (parent_indices)
			parent_indices[offset + i] = parentgaussian; 
		if (nodes_for_render_indices)
			nodes_for_render_indices[offset + i] = idx;
	}
}

__global__ void putRenderIndicesDynamic(HierarchyNode* nodes, int N, int* node_counts, int* node_offsets, int* render_indices, int* parent_indices = nullptr, int* nodes_for_render_indices = nullptr)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= N || node_counts[idx] == 0)
		return;
	int offset = idx == 0 ? 0 : node_offsets[idx - 1];
	HierarchyNode node = nodes[idx];
	render_indices[offset] = idx; 
	nodes_for_render_indices[offset] = idx;
	if (node.parent != -1)
	{
		parent_indices[offset] = node.parent;
	}
}

int Switching::expandToTarget(
	int N,
	int target,
	int* nodes,
	int* render_indices
)
{
	thrust::device_vector<int> render_counts(N);
	thrust::device_vector<int> render_offsets(N);

	int num_blocks = (N + 255) / 256;
	markTargetNodes << <num_blocks, 256 >> > ((Node*)nodes, N, target, render_counts.data().get());

	size_t temp_storage_bytes;
	thrust::device_vector<char> temp_storage;
	cub::DeviceScan::InclusiveSum(nullptr, temp_storage_bytes, render_counts.data().get(), render_offsets.data().get(), N);
	temp_storage.resize(temp_storage_bytes);
	cub::DeviceScan::InclusiveSum(temp_storage.data().get(), temp_storage_bytes, render_counts.data().get(), render_offsets.data().get(), N);

	putRenderIndices << <num_blocks, 256 >> > ((Node*)nodes, N, render_counts.data().get(), render_offsets.data().get(), render_indices);

	int count = 0;
	cudaMemcpy(&count, render_offsets.data().get() + N - 1, sizeof(int), cudaMemcpyDeviceToHost);
	return count;
}

__device__ bool inboxCUDA(Box& box, Point viewpoint)
{
	bool inside = true;
	for (int i = 0; i < 3; i++)
	{
		inside &= viewpoint.xyz[i] >= box.minn.xyz[i] && viewpoint.xyz[i] <= box.maxx.xyz[i];
	}
	return inside;
}


__device__ float pointgaussiandistCUDA(float3& position, float3& scale, Point viewpoint, Point zdir)
{
	float3 offset_pos = position;
	offset_pos.x += 3.0 * (position.x < viewpoint.xyz[0] ? scale.x : -scale.x);
	offset_pos.y += 3.0 * (position.y < viewpoint.xyz[1] ? scale.y : -scale.y);
	offset_pos.z += 3.0 * (position.z < viewpoint.xyz[2] ? scale.z : -scale.z);
	Point diff = {
		viewpoint.xyz[0] - position.x,
		viewpoint.xyz[1] - position.y,
		viewpoint.xyz[2] - position.z
	};
	float norm = sqrt(diff.xyz[0] * diff.xyz[0] + diff.xyz[1] * diff.xyz[1] + diff.xyz[2] * diff.xyz[2]);
	return norm;
	Point normalized_diff = {diff.xyz[0]/norm, diff.xyz[1] / norm, diff.xyz[2] / norm};
	float cos_angle = normalized_diff.xyz[0] * zdir.xyz[0] + normalized_diff.xyz[1] * zdir.xyz[1] + normalized_diff.xyz[2] * zdir.xyz[2];
		
}

__device__ bool is_in_frustum(float3& position, float3& scale, Point viewpoint, Point zdir)
{
	float3 offset_pos = position;
	offset_pos.x += 3.0 * (position.x < viewpoint.xyz[0] ? scale.x : -scale.x);
	offset_pos.y += 3.0 * (position.y < viewpoint.xyz[1] ? scale.y : -scale.y);
	offset_pos.z += 3.0 * (position.z < viewpoint.xyz[2] ? scale.z : -scale.z);
	Point diff = {
		viewpoint.xyz[0] - position.x,
		viewpoint.xyz[1] - position.y,
		viewpoint.xyz[2] - position.z
	};
	float norm = sqrt(diff.xyz[0] * diff.xyz[0] + diff.xyz[1] * diff.xyz[1] + diff.xyz[2] * diff.xyz[2]);
	Point normalized_diff = {diff.xyz[0]/norm, diff.xyz[1] / norm, diff.xyz[2] / norm};
	float cos_angle = normalized_diff.xyz[0] * zdir.xyz[0] + normalized_diff.xyz[1] * zdir.xyz[1] + normalized_diff.xyz[2] * zdir.xyz[2];
	if (cos_angle < -0.5)
	{
		return true;
	}
	else
	{
		return false;
	}
}

__device__ float pointboxdistCUDA(Box& box, Point viewpoint)
{
	Point closest = {
		max(box.minn.xyz[0], min(box.maxx.xyz[0], viewpoint.xyz[0])),
		max(box.minn.xyz[1], min(box.maxx.xyz[1], viewpoint.xyz[1])),
		max(box.minn.xyz[2], min(box.maxx.xyz[2], viewpoint.xyz[2]))
	};

	Point diff = {
		viewpoint.xyz[0] - closest.xyz[0],
		viewpoint.xyz[1] - closest.xyz[1],
		viewpoint.xyz[2] - closest.xyz[2]
	};

	return sqrt(diff.xyz[0] * diff.xyz[0] + diff.xyz[1] * diff.xyz[1] + diff.xyz[2] * diff.xyz[2]);
}


// computes the size of a box projected onto a particular viewpoint
// This implicitly assumes that zdir is (0,0,1) ?!
// projected AABB axis lengths?
__device__ float computeSizeGPU(Box& box, Point viewpoint, Point zdir)
{ 

	if (inboxCUDA(box, viewpoint))
		return FLT_MAX;
	// This is not taking view direction into account
	float min_dist = pointboxdistCUDA(box, viewpoint);
	// Somehow, somewhere, I guess this is set to the largest side of the box? => In the ClusterMerger
	return box.minn.xyz[3] / min_dist;
}


__device__ float computeSizeGPUDynamic(float3& position, float3& scale, Point viewpoint, Point zdir)
{
	//if (inboxCUDA(box, viewpoint))
	//	return FLT_MAX;
	// This is not taking view direction into account
	float min_dist = pointgaussiandistCUDA(position, scale, viewpoint, zdir);
	if (min_dist < 0.0)
	{
		return 0;
	}
	return fmaxf(scale.x, fmaxf(scale.y, scale.z)) / min_dist;
}


__global__ void changeNodesOnce(
	Node* nodes,
	int N,
	int* indices,
	Box* boxes,
	Point* viewpoint,
	Point zdir,
	float target_size,
	int* split,
	int* node_counts,
	int* node_ids,
	char* needs_children
)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= N)
		return;

	int node_id = indices[idx];
	Node node = nodes[node_id];
	float size = computeSizeGPU(boxes[node_id], *viewpoint, zdir);

	int count = 1; // repeat yourself
	char need_child = 0;
	if (size >= target_size)
	{
		if (node.depth > 0 && split[node_id] == 0) // split
		{
			if (node.start_children == -1)
			{
				node_ids[idx] = node_id;
				need_child = 1;
			}
			else
			{
				count += node.count_children;
				split[node_id] = 1;
			}
		}
	}
	else
	{
		int parent_node_id = node.parent;
		if (parent_node_id != -1)
		{
			Node parent_node = nodes[parent_node_id];
			float parent_size = computeSizeGPU(boxes[parent_node_id], *viewpoint, zdir);
			if (parent_size < target_size) // collapse
			{
				split[parent_node_id] = 0;
				count = 0; // forget yourself
			}
		}
	}
	needs_children[idx] = need_child;
	node_counts[idx] = count;
}

__global__ void putNodes(
	Node* nodes,
	int N,
	int* indices,
	int* node_counts,
	int* node_offsets,
	int* next_nodes)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= N)
		return;

	int count = node_counts[idx];
	if (count == 0)
		return;

	int node_id = indices[idx];
	Node node = nodes[node_id];
	int offset = idx == 0 ? 0 : node_offsets[idx - 1];

	next_nodes[offset] = node_id;
	for (int i = 1; i < count; i++)
		next_nodes[offset + i] = node.start_children + i - 1;
}

__global__ void countRenderIndicesIndexed(Node* nodes, int* split, int N, int* node_indices, int* render_counts)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= N)
		return;

	int node_idx = node_indices[idx];

	Node node = nodes[node_idx];
	int count = node.count_leafs;
	if (node.depth > 0 && split[node_idx] == 0)
		count += node.count_merged;

	render_counts[idx] = count;
}

__global__ void putRenderIndicesIndexed(Node* nodes, int N, int* node_indices, int* render_counts, int* render_offsets, int* render_indices, int* parent_indices, int* nodes_of_render_indices, Box* boxes, float3* debug)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= N)
		return;

	int node_idx = node_indices[idx];

	Node node = nodes[node_idx];
	int count = render_counts[idx];
	int offset = idx == 0 ? 0 : render_offsets[idx - 1];
	int start = node.start;

	int parentgaussian = -1;
	if (node.parent != -1)
	{
		parentgaussian = nodes[node.parent].start;
	}

	for (int i = 0; i < count; i++)
	{
		render_indices[offset + i] = node.start + i;
		parent_indices[offset + i] = parentgaussian;
		nodes_of_render_indices[offset + i] = node_idx;
	}

	if (debug != nullptr)
	{
		Box box = boxes[node_idx];
		for (int i = 0; i < count; i++)
		{
			float red = min(1.0f, node.depth / 10.0f);
			debug[offset + i] = { red, 1.0f - red, 0 };
			if (node.depth == 0)
				debug[offset + i] = { 0, 0, 1.0f };
		}
	}
}

void Switching::changeToSizeStep(
	float target_size,
	int N,
	int* node_indices,
	int* new_node_indices,
	int* nodes,
	float* boxes,
	float* viewpoint,
	float x, float y, float z,
	int* split,
	int* render_indices,
	int* parent_indices,
	int* nodes_of_render_indices,
	int* nodes_to_expand,
	float* debug,
	char*& scratchspace,
	size_t& scratchspacesize,
	int* NsrcI,
	int* NdstI,
	char* NdstC,
	int* numI,
	int maxN,
	int& add_success,
	int* new_N,
	int* new_R,
	int* need_expansion,
	void* maintenanceStream)
{
	cudaStream_t stream = (cudaStream_t)maintenanceStream;

	int num_node_blocks = (N + 255) / 256;

	Point zdir = { x, y, z };

	int* num_to_expand = numI;
	int* node_counts = NsrcI, * node_offsets = NdstI, * node_ids = NdstI;
	char* need_children = NdstC;
	if (scratchspacesize == 0)
	{
		size_t testsize;

		cub::DeviceScan::InclusiveSum(nullptr, testsize, node_counts, node_offsets, maxN, stream);
		scratchspacesize = testsize;
		cub::DeviceSelect::Flagged(nullptr, testsize, node_ids, need_children, nodes_to_expand, num_to_expand, maxN, stream);
		scratchspacesize = max(testsize, scratchspacesize);

		if (scratchspace)
			cudaFree(scratchspace);
		scratchspacesize = testsize;
		cudaMalloc(&scratchspace, scratchspacesize);
	}

	changeNodesOnce << <num_node_blocks, 256, 0, stream >> > (
		(Node*)nodes, 
		N, 
		node_indices, 
		(Box*)boxes, 
		(Point*)viewpoint, 
		zdir, 
		target_size, 
		split, 
		node_counts, 
		node_ids, 
		need_children
		);

	cub::DeviceSelect::Flagged(scratchspace, scratchspacesize, node_ids, need_children, nodes_to_expand, num_to_expand, N, stream);
	cub::DeviceScan::InclusiveSum(scratchspace, scratchspacesize, node_counts, node_offsets, N, stream);

	cudaMemcpyAsync(need_expansion, num_to_expand, sizeof(int), cudaMemcpyDeviceToHost, stream);
	cudaMemcpyAsync(new_N, node_offsets + N - 1, sizeof(int), cudaMemcpyDeviceToHost, stream);
	cudaStreamSynchronize(stream);

	if (*new_N > maxN)
	{
		add_success = 0;
		return;
	}

	putNodes << <num_node_blocks, 256, 0, stream>> > (
		(Node*)nodes,
		N, 
		node_indices, 
		node_counts, 
		node_offsets, 
		new_node_indices
		);

	int num_render_blocks = (*new_N + 255) / 256;
	int* render_counts = NsrcI, * render_offsets = NdstI;

	countRenderIndicesIndexed << <num_render_blocks, 256, 0, stream >> > (
		(Node*)nodes, 
		split, 
		*new_N, 
		new_node_indices, 
		render_counts
		);

	cub::DeviceScan::InclusiveSum(scratchspace, scratchspacesize, render_counts, render_offsets, *new_N, stream);

	putRenderIndicesIndexed << <num_render_blocks, 256, 0, stream >> > (
		(Node*)nodes, 
		*new_N, 
		new_node_indices, 
		render_counts, 
		render_offsets, 
		render_indices, 
		parent_indices, 
		nodes_of_render_indices, 
		(Box*)boxes,
		(float3*)debug
		);

	cudaMemcpyAsync(new_R, render_offsets + *new_N - 1, sizeof(int), cudaMemcpyDeviceToHost, stream);

	add_success = 1;
}

// computes how often a node is seen from a particular viewpoint. 
// node_markers == seen
__global__ void markNodesForSize(Node* nodes, Box* boxes, int N, Point* viewpoint, Point zdir, float target_size, int* render_counts, int* node_markers)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= N)
		return;

	int node_id = idx;
	Node node = nodes[node_id];
	// how large is this box projected onto a particular viewpoint?
	// this does not use zdir at all
	float size = computeSizeGPU(boxes[node_id], *viewpoint, zdir);

	int count = 0;
	if (size >= target_size)
		// this should be one if it is a leaf, otherwise 0
		count = node.count_leafs;
	// if your parent is greater than the target size, but you are not, you will be rendered
	else if (node.parent != -1)
	{
		float parent_size = computeSizeGPU(boxes[node.parent], *viewpoint, zdir);
		if (parent_size >= target_size)
		{
			// if you are a leaf node, count_leafs will be 1, otherwise count_merged will be 1
			count = node.count_leafs;
			if (node.depth != 0)
				count += node.count_merged;
		}
	}

	if (count != 0 && node_markers != nullptr)
		node_markers[node_id] = 1;

	if (render_counts != nullptr)
		render_counts[node_id] = count;
}

// computes how often a node is seen from a particular viewpoint. 
// node_markers == seen
__global__ void markNodesForSizeDynamic(HierarchyNode* nodes, float3* positions, float3* scales, int N, Point* viewpoint, Point zdir, float target_size, int* render_counts, int* node_markers)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= N)
		return;

	int node_id = idx;
	HierarchyNode node = nodes[node_id];

	if(!is_in_frustum(positions[node_id], scales[node_id], *viewpoint, zdir))
	{
		if (node_markers != nullptr)
			node_markers[node_id] = 0;

		if (render_counts != nullptr)
			render_counts[node_id] = 0;
		return;
	}
	// how large is this box projected onto a particular viewpoint?
	// this does not use zdir at all
	float size = computeSizeGPUDynamic(positions[node_id], scales[node_id], *viewpoint, zdir);

	int count = 0;
	// skybox points are added later
	if (node.depth < 0)
	{
		count = 0;
	}
	// leaf nodes are visible if all parent nodes are too coarse
	else if (size >= target_size && node.child_count == 0)
	{
		count = 1;
	}
	// if your parent is greater than the target size, but you are not, you will be rendered
	else if (node.parent >= 0)
	{
		float parent_size = computeSizeGPUDynamic(positions[node.parent], scales[node.parent], *viewpoint, zdir);
		if (parent_size >= target_size && size < target_size)
		{
			count = 1;
		}
	}

	if (count != 0 && node_markers != nullptr)
		node_markers[node_id] = 1;

	if (render_counts != nullptr)
		render_counts[node_id] = count;
		//printf("%d", count);
}


// compute interpolation weights
// kids returns the number of sibling nodes?
// nodes and boxes are already at this point by the indices in expand_to_size
__global__ void computeTsIndexed(
	Node* nodes,
	Box* boxes,
	int N,
	int* indices,
	Point viewpoint,
	Point zdir,
	float target_size,
	float* ts,
	int* num_siblings
)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= N)
		return;

	int node_id = indices[idx];
	Node node = nodes[node_id];

	float t;
	if (node.parent == -1)
		t = 1.0f;
	else
	{
		float parentsize = computeSizeGPU(boxes[node.parent], viewpoint, zdir);

		if (parentsize > 2.0f * target_size)
			t = 1.0f;
		else
		{
			float size = computeSizeGPU(boxes[node_id], viewpoint, zdir);
			float start = max(0.5f * parentsize, size);
			float diff = parentsize - start;

			if (diff <= 0)
				t = 1.0f;
			else
			{
				float tdiff = max(0.0f, target_size - start);
				t = max(1.0f - (tdiff / diff), 0.0f);
			}
		}
	}

	ts[idx] = t;
	num_siblings[idx] = (node.parent == -1) ? 1 : nodes[node.parent].count_children;
}


__global__ void computeTsIndexedDynamic(
	HierarchyNode* nodes,
	float3* positions,
	float3* scales,
	int N,
	int* indices,
	Point viewpoint,
	Point zdir,
	float target_size,
	float* ts,
	int* num_siblings
)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= N)
		return;

	int node_id = indices[idx];
	HierarchyNode node = nodes[node_id];

	float t;
	if (node.parent < 0)
		t = 1.0f;
	else
	{
		float parentsize = computeSizeGPUDynamic(positions[node.parent],scales[node.parent], viewpoint, zdir);

		if (parentsize > 2.0f * target_size)
			t = 1.0f;
		else
		{
			float size = computeSizeGPUDynamic(positions[node_id], scales[node_id], viewpoint, zdir);
			float start = max(0.5f * parentsize, size);
			float diff = parentsize - start;

			if (diff <= 0)
				t = 1.0f;
			else
			{
				float tdiff = max(0.0f, target_size - start);
				t = max(1.0f - (tdiff / diff), 0.0f);
			}
		}
	}

	ts[idx] = t;
	num_siblings[idx] = (node.parent < 0) ? 1 : nodes[node.parent].child_count;
}

// get interpolation weights
void Switching::getTsIndexed(
	int N,
	int* indices,
	float target_size,
	int* nodes,
	float* boxes,
	// camera position
	float vx, float vy, float vz,
	float x, float y, float z,
	float* ts,
	int* num_siblings,
	void* stream
)
{
	Point zdir = { x, y, z };
	Point cam = { vx, vy, vz };
	int num_blocks = (N + 255) / 256;
	computeTsIndexed<<<num_blocks, 256, 0, (cudaStream_t)stream >>>(
		(Node*)nodes, 
		(Box*)boxes, 
		N, 
		indices, 
		cam,
		zdir, 
		target_size, 
		ts, 
		num_siblings);
}

// get interpolation weights
void Switching::getTsIndexedDynamic(
	int N,
	int* indices,
	float target_size,
	int* nodes,
	float* positions,
	float* scales,
	// camera position
	float vx, float vy, float vz,
	float x, float y, float z,
	float* ts,
	int* num_siblings,
	void* stream
)
{
	Point zdir = { x, y, z };
	Point cam = { vx, vy, vz };
	int num_blocks = (N + 255) / 256;
	computeTsIndexedDynamic<<<num_blocks, 256, 0, (cudaStream_t)stream >>>(
		(HierarchyNode*)nodes, 
		(float3*) positions,
		(float3*) scales, 
		N, 
		indices, 
		cam,
		zdir, 
		target_size, 
		ts, 
		num_siblings);
}

int Switching::expandToSizeDynamic(
	int N,
	float target_size,
	int* nodes,
	float* positions,
	float* scales,
	float* viewpoint,
	float x, float y, float z,
	int* render_indices,
	int* node_markers,
	int* parent_indices,
	int* nodes_for_render_indices)
{

	size_t temp_storage_bytes;
	thrust::device_vector<char> temp_storage;
	thrust::device_vector<int> render_counts(N);
	// inclusive sum of render_counts
	thrust::device_vector<int> render_offsets(N);

	Point zdir = { x, y, z };

	int num_blocks = (N + 255) / 256;
	
	// mark the highest nodes in the hierarchy that are below limit in node_markers and populate render_counts
	markNodesForSizeDynamic << <num_blocks, 256 >> > ((HierarchyNode*)nodes, (float3*)positions, (float3*)scales, N, (Point*)viewpoint, zdir, target_size, render_counts.data().get(), node_markers);
	cub::DeviceScan::InclusiveSum(nullptr, temp_storage_bytes, render_counts.data().get(), render_offsets.data().get(), N);
	temp_storage.resize(temp_storage_bytes);
	cub::DeviceScan::InclusiveSum(temp_storage.data().get(), temp_storage_bytes, render_counts.data().get(), render_offsets.data().get(), N);
	putRenderIndicesDynamic << <num_blocks, 256 >> > ((HierarchyNode*)nodes, N, render_counts.data().get(), render_offsets.data().get(), render_indices, parent_indices, nodes_for_render_indices);

	int count = 0;
	cudaMemcpy(&count, render_offsets.data().get() + N - 1, sizeof(int), cudaMemcpyDeviceToHost);
	return count;
}

__global__ void binary_search_SPTs(int number_of_SPTs, int* SPT_starts, float* SPT_max, int* SPT_indices, float* SPT_distances, int* interval_sizes)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= number_of_SPTs)
		return;
	int index = SPT_indices[idx];
	float distance = SPT_distances[idx];
	
	int low = SPT_starts[index];
	int high = SPT_starts[index+1];
	int pivot = (low+high)/2;
	while((high-low) > 1)
	{
		if (SPT_max[pivot] > distance)
		{
			low = pivot;
		}
		else
		{
			high = pivot;
		}
		pivot = (low+high)/2;
	}
	//binary_search_indices[idx] = high;

	interval_sizes[idx] = max(high - SPT_starts[index]-1, 1);
}

__global__ void populate_SPT
(
	int sum, 
	int number_of_SPTs,
	int* SPT_indices, 
	int* SPT_starts, 
	float* SPT_min, 
	float* SPT_distance, 
	int* prefix_sum, 
	int* gaussian_indices, 
	int* results,
	int* SPT_counts)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= sum)
		return;
	int	low = 0;
	int high = number_of_SPTs;

	int interval_index = number_of_SPTs/2;
	while(high - low > 1)
	{
		if (prefix_sum[interval_index] >= idx)
		{
			high = interval_index;
		}
		else
		{
			low = interval_index;
		}
		interval_index = (high + low)/2;
	}
	//results[idx] = interval_index;
	//return;
	int interval_offset = idx - prefix_sum[interval_index];
	int SPT_index = SPT_indices[interval_index];
	int gaussian_index = SPT_starts[SPT_index] + interval_offset;
	results[idx] = 0;
	if (SPT_min[gaussian_index] < SPT_distance[interval_index])
	{
		atomicAdd(&SPT_counts[interval_index], 1);
		results[idx] = gaussian_indices[gaussian_index];
	}
	//results[idx] = interval_index;
}


struct IsNonZero {
    __host__ __device__ bool operator()(const int x) const {
        return x != 0;
    }
};

template<typename T>
void print_device_array(const T* d_array, int size) {
    int* h_array = new int[size];
	(cudaMemcpy(h_array, d_array, size * sizeof(int), cudaMemcpyDeviceToHost));
	std::cout << "Device array: ";
    for (int i = 0; i < size; i++) {
        std::cout << h_array[i] << " ";
    }
    std::cout << std::endl;
	std::cout << std::endl;
	delete[] h_array;
}

int Switching::getSPTCut(
	int number_of_SPTs,
	int* gaussian_indices,
	int* SPT_starts,
	float* SPT_max,
	float* SPT_min,
	int* SPT_indices,
	float* SPT_distances,
	int** SPT_cut,
	int** SPT_counts_out)
{
	// binary search for each SPT_index
	int* interval_sizes;
	CHECK_CUDA(cudaMalloc((void**)&interval_sizes, sizeof(int) * number_of_SPTs), true);
	int num_blocks = (number_of_SPTs + 255) / 256;
	binary_search_SPTs <<<num_blocks, 256 >>>(number_of_SPTs, SPT_starts, SPT_max, SPT_indices, SPT_distances, interval_sizes);
	//print_device_array<int>(interval_sizes, 10);
	cudaDeviceSynchronize();
	//
	//cudaFree(binary_search_indices);
	//print_device_array<int>(interval_sizes, 100);
	//std::cout << binary_search_indices << std::endl;
	// compute prefix sum of ranges
	//cudaFree(binary_search_indices);
	
	int* prefix_sum = nullptr;
	
	void* d_temp_storage = nullptr;
	size_t temp_storage_bytes = 0;
	cub::DeviceScan::ExclusiveSum(
	  d_temp_storage, temp_storage_bytes,
	  interval_sizes, prefix_sum, number_of_SPTs);
	CHECK_CUDA(cudaMalloc(&d_temp_storage, temp_storage_bytes), true);
	CHECK_CUDA(cudaMalloc(&prefix_sum, sizeof(int) * number_of_SPTs), true);
	CHECK_CUDA(cub::DeviceScan::ExclusiveSum(
  	d_temp_storage, temp_storage_bytes,
  	interval_sizes, prefix_sum, number_of_SPTs), true);
	cudaDeviceSynchronize();
	CHECK_CUDA(cudaFree(d_temp_storage), true);
	
	//*return_value = prefix_sum;
	//return number_of_SPTs;
	//print_device_array<int>(prefix_sum, 100);
	int* SPT_counts;
	CHECK_CUDA(cudaMalloc((void**)&SPT_counts, number_of_SPTs * sizeof(int)), true);
	CHECK_CUDA(cudaMemset(SPT_counts, 0, number_of_SPTs * sizeof(int)), true);

	int sum;
	CHECK_CUDA(cudaMemcpy(&sum, &prefix_sum[number_of_SPTs-1], sizeof(int),cudaMemcpyDeviceToHost), true);
	int addition;
	CHECK_CUDA(cudaMemcpy(&addition, &interval_sizes[number_of_SPTs-1], sizeof(int),cudaMemcpyDeviceToHost), true);
	sum += addition + 1;
	int* result;
	CHECK_CUDA(cudaMalloc((void**)&result, sizeof(int) * sum), true);
	num_blocks = (sum + 255) / 256;
	populate_SPT <<<num_blocks, 256 >>>(sum, number_of_SPTs, SPT_indices, SPT_starts, SPT_min,  SPT_distances, prefix_sum,  gaussian_indices, result, SPT_counts);
	cudaDeviceSynchronize();
	cudaFree(prefix_sum);
	cudaFree(interval_sizes);	
	//*SPT_cut = result;
	//return sum;
	//std::cout << "NUMBER OF RENDER INDICES:" << h_num_indices << std::endl;
	//print_device_array<int>(result, 100);

	int* num_indices;
    d_temp_storage = nullptr;
	int* result_compacted;
	CHECK_CUDA(cudaMalloc(&result_compacted, sum * sizeof(int)), true);
	CHECK_CUDA(cudaMalloc(&num_indices, sizeof(int)), true);
    temp_storage_bytes = 0;
    CHECK_CUDA(cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes, result, result_compacted, num_indices, sum, IsNonZero()), true);
    CHECK_CUDA(cudaMalloc(&d_temp_storage, temp_storage_bytes), true);
    CHECK_CUDA(cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes, result, result_compacted, num_indices, sum, IsNonZero()), true);
	cudaDeviceSynchronize();
	CHECK_CUDA(cudaFree(d_temp_storage), true);
	int h_num_indices;
	CHECK_CUDA(cudaMemcpy(&h_num_indices, num_indices, sizeof(int),cudaMemcpyDeviceToHost), true);
	cudaDeviceSynchronize();
	CHECK_CUDA(cudaFree(num_indices), true);

	//print_device_array<int>(result_compacted, 100);
	//*SPT_cut = result_compacted;
	//return h_num_indices;
	cudaFree(result);
	 

	int* final_result;
	CHECK_CUDA(cudaMalloc(&final_result, h_num_indices * sizeof(int)), true);
	cudaDeviceSynchronize();
	CHECK_CUDA(cudaMemcpy(final_result, result_compacted, h_num_indices * sizeof(int), cudaMemcpyDeviceToDevice), true);
	cudaDeviceSynchronize();
	cudaFree(result_compacted);



	int* SPT_count_prefix_sum = nullptr;
	
	d_temp_storage = nullptr;
	temp_storage_bytes = 0;
	cub::DeviceScan::ExclusiveSum(
	  d_temp_storage, temp_storage_bytes,
	  SPT_counts, SPT_count_prefix_sum, number_of_SPTs);
	cudaMalloc(&d_temp_storage, temp_storage_bytes);
	cudaMalloc(&SPT_count_prefix_sum, sizeof(int) * number_of_SPTs);
	cub::DeviceScan::ExclusiveSum(
  	d_temp_storage, temp_storage_bytes,
  	SPT_counts, SPT_count_prefix_sum, number_of_SPTs);
	cudaDeviceSynchronize();
	cudaFree(d_temp_storage);
	cudaFree(SPT_counts);


	*SPT_counts_out = SPT_count_prefix_sum;
	*SPT_cut = final_result;
	//std::cout << "NUMBER OF RENDER INDICES:" << h_num_indices << std::endl;
	return h_num_indices;
}


int Switching::expandToSize(
	int N,
	float target_size,
	int* nodes,
	float* boxes,
	float* viewpoint,
	float x, float y, float z,
	int* render_indices,
	int* node_markers,
	int* parent_indices,
	int* nodes_for_render_indices)
{
	size_t temp_storage_bytes;
	thrust::device_vector<char> temp_storage;
	thrust::device_vector<int> render_counts(N);
	// inclusive sum of render_counts
	thrust::device_vector<int> render_offsets(N);

	Point zdir = { x, y, z };

	int num_blocks = (N + 255) / 256;
	// mark the highest nodes in the hierarchy that are below limit in node_markers and populate render_counts
	markNodesForSize << <num_blocks, 256 >> > ((Node*)nodes, (Box*)boxes, N, (Point*)viewpoint, zdir, target_size, render_counts.data().get(), node_markers);

	cub::DeviceScan::InclusiveSum(nullptr, temp_storage_bytes, render_counts.data().get(), render_offsets.data().get(), N);
	temp_storage.resize(temp_storage_bytes);
	cub::DeviceScan::InclusiveSum(temp_storage.data().get(), temp_storage_bytes, render_counts.data().get(), render_offsets.data().get(), N);

	putRenderIndices << <num_blocks, 256 >> > ((Node*)nodes, N, render_counts.data().get(), render_offsets.data().get(), render_indices, parent_indices, nodes_for_render_indices);

	int count = 0;
	cudaMemcpy(&count, render_offsets.data().get() + N - 1, sizeof(int), cudaMemcpyDeviceToHost);
	return count;
}




// 
void Switching::markVisibleForAllViewpoints(
	float target_size,
	int* nodes,
	int num_nodes,
	float* boxes,
	float* viewpoints,
	int num_viewpoints,
	int* seen,
	float zx,
	float zy,
	float zz
)
{
	thrust::device_vector<int> seen_cuda(num_nodes);
	thrust::device_vector<Point> viewpoint_cuda(1);
	thrust::device_vector<Node> nodes_cuda(num_nodes);
	thrust::device_vector<Box> boxes_cuda(num_nodes);

	cudaMemcpy(nodes_cuda.data().get(), nodes, sizeof(Node) * num_nodes, cudaMemcpyHostToDevice);
	cudaMemcpy(boxes_cuda.data().get(), boxes, sizeof(Box) * num_nodes, cudaMemcpyHostToDevice);

	Point zdir = { zx, zy, zz };

	Point* points = (Point*)viewpoints;
	int num_blocks = (num_nodes + 255) / 256;
	for (int i = 0; i < num_viewpoints; i++)
	{
		Point viewpoint = points[i];
		cudaMemcpy(viewpoint_cuda.data().get(), &viewpoint, sizeof(Point), cudaMemcpyHostToDevice);

		markNodesForSize << <num_blocks, 256 >> > (
			nodes_cuda.data().get(),
			boxes_cuda.data().get(),
			num_nodes,
			viewpoint_cuda.data().get(),
			zdir,
			target_size,
			nullptr,
			seen_cuda.data().get());
	}
	cudaMemcpy(seen, seen_cuda.data().get(), sizeof(int) * num_nodes, cudaMemcpyDeviceToHost);

	if (cudaDeviceSynchronize())
		std::cout << "Errors: " << cudaDeviceSynchronize() << std::endl;
}