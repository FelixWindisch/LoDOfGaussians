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

class AppearanceFilter
{
private:

	struct CameraParametersColmap {
		size_t id;
		size_t width;
		size_t height;
		float  fx;
		float  fy;
		float  dx;
		float  dy;
	};

	struct Camera
	{
		CameraParametersColmap& params;
		Eigen::Vector3f position;
	};

	std::vector<CameraParametersColmap> camera_params;
	std::vector<Camera> cameras;

public:

	void init(const char* colmappath);

	void filter(ExplicitTreeNode* root, const std::vector<Gaussian>& gaussians, float orig_limit, float layermultiplier);

	void writeAnchors(const char* filename, ExplicitTreeNode* root, const std::vector<Gaussian>& gaussians, float limit);

};