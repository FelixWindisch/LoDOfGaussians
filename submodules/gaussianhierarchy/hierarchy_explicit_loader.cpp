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


#include "hierarchy_loader.h"
#include "hierarchy_explicit_loader.h"
#include <vector>
#include <Eigen/Dense>
#include "common.h"
#include <iostream>
#include <fstream>
#include "half.hpp"

float getWeight(Eigen::Vector3f& pos, int chunk_id,
	std::vector<Eigen::Vector3f>& chunk_centers)
{
	float dist_to_current_center = (pos - chunk_centers[chunk_id]).norm();
	float min_dist_to_other_center = 1e12f;
	for (int other_chunk_id(0); other_chunk_id < chunk_centers.size(); other_chunk_id++)
	{
		if (other_chunk_id != chunk_id)
		{
			float dist_to_other_center = (pos - chunk_centers[other_chunk_id]).norm();
			if (min_dist_to_other_center > dist_to_other_center)
			{
				min_dist_to_other_center = dist_to_other_center;
			}
		}
	}
	float falloff = 0.05f;
	if (dist_to_current_center <= (1.f - falloff) * min_dist_to_other_center)
	{
		return 1.f;
	}
	else if (dist_to_current_center > (1.f + falloff) * min_dist_to_other_center)
	{
		return 0.f;
	}
	else
	{
		float a = -1.f / (2 * falloff * min_dist_to_other_center);
		float b = (1.f + falloff) / (2 * falloff);
		return a * dist_to_current_center + b;
	}
}

std::vector<ExplicitTreeNode*> buildTreeRec(ExplicitTreeNode* explicitNode,
	std::vector<Gaussian>& gaussians,
	int chunk_id,
	std::vector<Eigen::Vector3f>& chunk_centers,
	Node& node,
	int node_id,
	std::vector<Eigen::Vector3f>& pos,
	std::vector<SHs>& shs,
	std::vector<float>& alphas,
	std::vector<Eigen::Vector3f>& scales,
	std::vector<Eigen::Vector4f>& rot,
	std::vector<Node>& nodes,
	std::vector<Box>& boxes)
{
	explicitNode->depth = node.depth;
	explicitNode->bounds = boxes[node_id];
	int n_valid_gaussians = 0;
	if (node.depth > 0)
	{
		for (int n(0); n < node.count_merged; n++)
		{
			float weigth = getWeight(pos[node.start + n], chunk_id, chunk_centers);
			if (weigth > 0.f)
			{
				n_valid_gaussians++;
				Gaussian g;
				g.position = pos[node.start + n];
				g.rotation = rot[node.start + n];
				g.opacity = alphas[node.start + n] * weigth;
				g.scale = scales[node.start + n].array().exp();
				g.shs = shs[node.start + n];

				explicitNode->merged.push_back(g);
			}
		}
	}
	else
	{
		for (int n(0); n < node.count_leafs; n++)
		{
			float weigth = getWeight(pos[node.start + n], chunk_id, chunk_centers);
			if (weigth > 0.f)
			{
				n_valid_gaussians++;
				Gaussian g;
				g.position = pos[node.start + n];
				g.rotation = rot[node.start + n];
				g.opacity = alphas[node.start + n] * weigth;
				g.scale = scales[node.start + n].array().exp();
				g.shs = shs[node.start + n];

				explicitNode->leaf_indices.push_back(gaussians.size());
				gaussians.push_back(g);
			}
		}
	}

	std::vector<ExplicitTreeNode*> children;
	for (int i = 0; i < node.count_children; i++)
	{
		ExplicitTreeNode* newNode = new ExplicitTreeNode;
		std::vector<ExplicitTreeNode*> newChildren = buildTreeRec(newNode, gaussians, chunk_id, chunk_centers,
			nodes[node.start_children + i], node.start_children + i,
			pos, shs, alphas, scales, rot, nodes, boxes
		);
		children.insert(children.end(), newChildren.begin(), newChildren.end());
	}

	if (n_valid_gaussians > 0)
	{
		explicitNode->children = children;
		return std::vector<ExplicitTreeNode*>(1, explicitNode);
	}
	else
	{
		return children;
	}
}

void HierarchyExplicitLoader::loadExplicit(
	const char* filename, std::vector<Gaussian>& gaussians, ExplicitTreeNode* root,
	int chunk_id, std::vector<Eigen::Vector3f>& chunk_centers)
{
	std::vector<Eigen::Vector3f> pos;
	std::vector<SHs> shs;
	std::vector<float> alphas;
	std::vector<Eigen::Vector3f> scales;
	std::vector<Eigen::Vector4f> rot;
	std::vector<Node> nodes;
	std::vector<Box> boxes;
	HierarchyLoader::load(filename, pos, shs, alphas, scales, rot, nodes, boxes);

	pos[0] = chunk_centers[chunk_id];
	buildTreeRec(root, gaussians, chunk_id, chunk_centers, nodes[0], 0, pos, shs, alphas, scales, rot, nodes, boxes);
}
