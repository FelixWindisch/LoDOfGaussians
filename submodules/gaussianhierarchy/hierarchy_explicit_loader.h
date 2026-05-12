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

#include <vector>
#include <Eigen/Dense>
#include "common.h"
#include <iostream>
#include <fstream>

class HierarchyExplicitLoader
{
public:

	static void loadExplicit(const char* filename,
		std::vector<Gaussian>& gaussian, ExplicitTreeNode* root,
		int chunk_id, std::vector<Eigen::Vector3f>& chunk_centers);
};