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


#include "traversal.h"

void recExpand(int node_id, Node* nodes, std::vector<int>& indices, int target)
{
	Node& node = nodes[node_id];

	for (int i = 0; i < node.count_leafs; i++)
		indices.push_back(node.start + i);

	if (node.depth <= target) // We are below target. Nodes will not be expanded, content approximated
	{
		for (int i = 0; i < node.count_merged; i++)
			indices.push_back(node.start + node.count_leafs + i);
	}
	else // we keep expanding and adding visible leaves
	{
		for (int i = 0; i < node.count_children; i++)
			recExpand(node.start_children + i, nodes, indices, target);
	}
}

std::vector<int> Traversal::expandToTarget(Node* nodes, int target)
{
	std::vector<int> indices;
	recExpand(0, nodes, indices, target);
	return indices;
}