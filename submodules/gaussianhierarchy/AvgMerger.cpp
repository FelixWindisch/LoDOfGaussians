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

#include "AvgMerger.h"

void AvgMerger::mergeRec(ExplicitTreeNode* node, const std::vector<Gaussian>& leaf_gaussians)
{
	Gaussian g;
	g.position = Eigen::Vector3f::Zero();
	g.rotation = Eigen::Vector4f::Zero();
	g.opacity = 0;
	g.scale = Eigen::Vector3f::Zero();
	g.shs = SHs::Zero();

	float div = node->children.size() + node->leaf_indices.size();

	auto avgmerge = [&div](Gaussian& g, const Gaussian& x) {
		g.position += x.position / div;
		g.opacity += x.opacity / div;
		g.scale += x.scale;
		g.rotation += x.rotation / div;
		g.shs += x.shs / div;
	};

	for (auto& child : node->children)
	{
		mergeRec(child, leaf_gaussians);
		avgmerge(g, child->merged[0]);

		for (auto& child_leaf : child->leaf_indices)
			avgmerge(g, leaf_gaussians[child_leaf]);
	}

	g.rotation = g.rotation.normalized();

	node->merged.push_back(g);
}

void AvgMerger::merge(ExplicitTreeNode* root, const std::vector<Gaussian>& leaf_gaussians)
{
	mergeRec(root, leaf_gaussians);
}