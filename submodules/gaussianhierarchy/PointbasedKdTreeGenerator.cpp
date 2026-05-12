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


#include "PointbasedKdTreeGenerator.h"
#include <numeric>

ExplicitTreeNode* recKdTree(const std::vector<Gaussian>& gaussians, int* g_indices, int start, int num)
{
	auto node = new ExplicitTreeNode;
	Point minn = { FLT_MAX, FLT_MAX, FLT_MAX };
	Point maxx = { -FLT_MAX, -FLT_MAX, -FLT_MAX };
	for (int i = 0; i < num; i++)
	{
		const Gaussian& g = gaussians[g_indices[start + i]];
		// this is just a maximum value
		float r = 3.0f * g.scale.maxCoeff();
		auto gmin = g.position;
		gmin.array() -= r;
		auto gmax = g.position;
		gmax.array() += r;
		minn = minn.cwiseMin(gmin);
		maxx = maxx.cwiseMax(gmax);
	}
	node->bounds = { minn, maxx };
	
	if (num == 1)
	{
		node->depth = 0;
		node->leaf_indices.push_back(g_indices[start]);
		return node;
		// this is where the recursion ends
	}
	else
	{
		int axis = 0;
		float greatest_dist = 0;
		for (int i = 0; i < 3; i++)
		{
			float dist = maxx[i] - minn[i];
			if (dist > greatest_dist)
			{
				greatest_dist = dist;
				axis = i;
			}
		}

		int* range = g_indices + start;
		int pivot = num / 2 - 1;
		std::nth_element(range, range + pivot, range + num,
			[&](const int a, const int b) { return gaussians[a].position[axis] < gaussians[b].position[axis]; }
		);

		node->children.push_back(recKdTree(gaussians, g_indices, start, pivot + 1));
		node->children.push_back(recKdTree(gaussians, g_indices, start + pivot + 1, num - (pivot + 1)));
		node->depth = std::max(node->children[0]->depth, node->children[1]->depth) + 1;
	}

	return node;
}

ExplicitTreeNode* PointbasedKdTreeGenerator::generate(const std::vector<Gaussian>& gaussians)
{
	std::vector<int> indices(gaussians.size());
	std::iota(indices.begin(), indices.end(), 0);
	return recKdTree(gaussians, indices.data(), 0, gaussians.size());
}
