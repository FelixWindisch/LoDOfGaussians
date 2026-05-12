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


#include "loader.h"
#include "writer.h"
#include "FlatGenerator.h"
#include "PointbasedKdTreeGenerator.h"
#include "AvgMerger.h"
#include "ClusterMerger.h"
#include "common.h"
#include "dependencies/json.hpp"
#include "hierarchy_explicit_loader.h"
#include <vector>
#include <iostream>
#include <fstream>
#include <filesystem>
#include "appearance_filter.h"
#include "rotation_aligner.h"

using json = nlohmann::json;

void recTraverse(ExplicitTreeNode* node, int& zerocount)
{
	if (node->depth == 0)
		zerocount++;
	if (node->children.size() > 0 && node->depth == 0)
		throw std::runtime_error("Leaf nodes should never have children!");

	for (auto c : node->children)
	{
		recTraverse(c, zerocount);
	}
}

int main(int argc, char* argv[])
{
	if (argc < 5)
		throw std::runtime_error("Failed to pass filename");

	int chunk_count(argc - 5);
	std::string rootpath(argv[1]);
	if (std::stoi(argv[2]))
	{
		for (int chunk_id(0); chunk_id < chunk_count; chunk_id++)
		{
			int argidx(chunk_id + 5);
			std::cout << "Building hierarchy for chunk " << argv[chunk_id + 5] << std::endl;

			AppearanceFilter filter;
			std::string inpath(argv[3]);
			filter.init((inpath + "/" + argv[argidx]).c_str());

			std::vector<Gaussian> gaussians;
			ExplicitTreeNode *root;

			// TODO: MAKE THIS DEPEND ON SOME INPUT ARGUMENT
			Loader::loadPly<LessRichPoint>((rootpath + "/" + argv[argidx] + "/point_cloud/iteration_30000/point_cloud.ply").c_str(), gaussians, 0);

			std::cout << "Generating" << std::endl;
			PointbasedKdTreeGenerator generator;
			root = generator.generate(gaussians);

			std::cout << "Merging" << std::endl;
			ClusterMerger merger;
			merger.merge(root, gaussians);

			std::cout << "Fixing rotations" << std::endl;
			RotationAligner::align(root, gaussians);

			std::cout << "Filtering" << std::endl;
			filter.filter(root, gaussians, 0.0005f, 4.0f);

			std::cout << "Writing" << std::endl;

			Writer::writeHierarchy(
				(rootpath + "/" + argv[argidx] + "/chunk.hier").c_str(), 
				gaussians, 
				root);
		}
	}
	else
	{
		// Read chunk centers
		std::string inpath(argv[3]);
		std::vector<Eigen::Vector3f> chunk_centers(chunk_count);
		for (int chunk_id(0); chunk_id < chunk_count; chunk_id++)
		{
			int argidx(chunk_id + 5);
			std::ifstream f(inpath + "/" + argv[argidx] + "/center.txt");
			Eigen::Vector3f chunk_center(0.f, 0.f, 0.f);
			f >> chunk_center[0]; f >> chunk_center[1]; f >> chunk_center[2];
			chunk_centers[chunk_id] = chunk_center;
		}

		// Read per chunk hierarchies and discard unwanted primitives 
		// based on the distance to the chunk's center
		std::vector<Gaussian> gaussians; 
		ExplicitTreeNode* root = new ExplicitTreeNode;

		for (int chunk_id(0); chunk_id < chunk_count; chunk_id++)
		{
			int argidx(chunk_id + 5);
			std::cout << "Adding hierarchy for chunk " << argv[argidx] << std::endl;
			
			ExplicitTreeNode* chunkRoot = new ExplicitTreeNode;
			HierarchyExplicitLoader::loadExplicit(
				(rootpath + "/" + argv[argidx] + "/hierarchy.hier_opt").c_str(),
				gaussians, chunkRoot, chunk_id, chunk_centers);

			if (chunk_id == 0)
			{
				root->bounds = chunkRoot->bounds;
			}
			else
			{	
				for (int idx(0); idx < 3; idx++)
				{
					root->bounds.minn[idx] = std::min(root->bounds.minn[idx], chunkRoot->bounds.minn[idx]);
					root->bounds.maxx[idx] = std::max(root->bounds.maxx[idx], chunkRoot->bounds.maxx[idx]);
				}
			}
			root->depth = std::max(root->depth, chunkRoot->depth + 1);
			root->children.push_back(chunkRoot);
			root->merged.push_back(chunkRoot->merged[0]);
			root->bounds.maxx[3] = 1e9f;
			root->bounds.minn[3] = 1e9f;
		}

		Writer::writeHierarchy(
			(argv[4]), 
			gaussians, root, true);
	}
}