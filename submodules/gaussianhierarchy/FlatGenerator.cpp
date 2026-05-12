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

#include "FlatGenerator.h"

ExplicitTreeNode* FlatGenerator::generate(const std::vector<Gaussian>& gaussians)
{
	auto node = new ExplicitTreeNode();

	Point minn = { FLT_MAX, FLT_MAX, FLT_MAX };
	Point maxx = { -FLT_MAX, -FLT_MAX, -FLT_MAX };
	for (int i = 0; i < gaussians.size(); i++)
	{
		const Gaussian& g = gaussians[i];
		minn = minn.cwiseMin(g.position);
		maxx = maxx.cwiseMax(g.position);
		node->leaf_indices.push_back(i);
	}
	node->bounds = { minn, maxx };
	node->depth = 0;

	return std::move(node);
}