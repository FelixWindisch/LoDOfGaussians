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

class AvgMerger
{
private:
	void mergeRec(ExplicitTreeNode* node, const std::vector<Gaussian>& leaf_gaussians);
public:
	void merge(ExplicitTreeNode* root, const std::vector<Gaussian>& gaussians);
};