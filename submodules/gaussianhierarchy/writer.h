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

#include "common.h"
#include <map>

class Writer
{
public:
	static void writeHierarchy
	(
		const char* filename, 
		const std::vector<Gaussian>& gaussian, 
		ExplicitTreeNode* root, 
		bool compressed = true);

	static void writeDynamicHierarchy
	(
		const char* filename, 
		const std::vector<Gaussian>& gaussian, 
		ExplicitTreeNode* root);

	static void makeDynamicHierarchy(
		const std::vector<Gaussian>& gaussians,
		ExplicitTreeNode* root,
		std::vector<Eigen::Vector3f>& positions,
		std::vector<Eigen::Vector4f>& rotations,
		std::vector<Eigen::Vector3f>& log_scales,
		std::vector<float>& opacities,
		std::vector<SHs>& shs,
		std::vector<HierarchyNode>& basenodes,
		std::map<int, const ExplicitTreeNode*>* base2tree = nullptr);

	static void makeHierarchy(
		const std::vector<Gaussian>& gaussians,
		const ExplicitTreeNode* root,
		std::vector<Eigen::Vector3f>& positions,
		std::vector<Eigen::Vector4f>& rotations,
		std::vector<Eigen::Vector3f>& log_scales,
		std::vector<float>& opacities,
		std::vector<SHs>& shs,
		std::vector<Node>& basenodes,
		std::vector<Box>& boxes,
		std::map<int, const ExplicitTreeNode*>* base2tree = nullptr);


	static void writeHierarchyGDF(
		const char* filename, 
		const ExplicitTreeNode* root, int max_depth);
};