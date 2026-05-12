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

#pragma once
#include <cstdio>
#include <tuple>
#include <string>

class Maintenance
{
public:

	static void reorder(
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
	);

	static void compress(
		int N,
		int* nodes,
		float* scales,
		float* rots,
		float* shs,
		float* opacs
	);

	static void compactPart1(
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
	);

	static void compactPart2(
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
	);

	static void updateStarts(
		int* nodes,
		int num_indices,
		int* indices,
		int* starts,
		void* streamy
	);
};