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
#include <vector>
#include <Eigen/Dense>
#include "common.h"
#include <iostream>
#include <fstream>
#include "half.hpp"

struct HalfBox2
{
	half_float::half minn[4];
	half_float::half maxx[4];
};

void HierarchyLoader::load(const char* filename,
	std::vector<Eigen::Vector3f>& pos,
	std::vector<SHs>& shs,
	std::vector<float>& alphas,
	std::vector<Eigen::Vector3f>& scales,
	std::vector<Eigen::Vector4f>& rot,
	std::vector<Node>& nodes,
	std::vector<Box>& boxes)
{
	std::ifstream infile(filename, std::ios_base::binary);

	if (!infile.good())
		throw std::runtime_error("File not found!");

	int P;
	infile.read((char*)&P, sizeof(int));

	if (P >= 0)
	{
		pos.resize(P);
		shs.resize(P);
		alphas.resize(P);
		scales.resize(P);
		rot.resize(P);

		infile.read((char*)pos.data(), P * sizeof(Eigen::Vector3f));
		infile.read((char*)rot.data(), P * sizeof(Eigen::Vector4f));
		infile.read((char*)scales.data(), P * sizeof(Eigen::Vector3f));
		infile.read((char*)alphas.data(), P * sizeof(float));
		infile.read((char*)shs.data(), P * sizeof(SHs));

		int N;
		infile.read((char*)&N, sizeof(int));

		nodes.resize(N);
		boxes.resize(N);

		infile.read((char*)nodes.data(), N * sizeof(Node));
		infile.read((char*)boxes.data(), N * sizeof(Box));
	}
	else
	{
		size_t allP = -P;

		pos.resize(allP);
		shs.resize(allP);
		alphas.resize(allP);
		scales.resize(allP);
		rot.resize(allP);

		std::vector<half_float::half> half_rotations(allP * 4);
		std::vector<half_float::half> half_scales(allP * 3);
		std::vector<half_float::half> half_opacities(allP);
		std::vector<half_float::half> half_shs(allP * 48);

		infile.read((char*)pos.data(), allP * sizeof(Eigen::Vector3f));
		infile.read((char*)half_rotations.data(), allP * 4 * sizeof(half_float::half));
		infile.read((char*)half_scales.data(), allP * 3 * sizeof(half_float::half));
		infile.read((char*)half_opacities.data(), allP * sizeof(half_float::half));
		infile.read((char*)half_shs.data(), allP * 48 * sizeof(half_float::half));

		for (size_t i = 0; i < allP; i++)
		{
			for (size_t j = 0; j < 4; j++)
				rot[i][j] = half_rotations[i * 4 + j];
			for (size_t j = 0; j < 3; j++)
				scales[i][j] = half_scales[i * 3 + j];
			alphas[i] = half_opacities[i];
			for (size_t j = 0; j < 48; j++)
				shs[i][j] = half_shs[i * 48 + j];
		}

		int N;
		infile.read((char*)&N, sizeof(int));
		size_t allN = N;

		nodes.resize(allN);
		boxes.resize(allN);

		std::vector<HalfNode> half_nodes(allN);
		std::vector<HalfBox2> half_boxes(allN);

		infile.read((char*)half_nodes.data(), allN * sizeof(HalfNode));
		infile.read((char*)half_boxes.data(), allN * sizeof(HalfBox2));

		for (int i = 0; i < allN; i++)
		{
			nodes[i].parent = half_nodes[i].parent;
			nodes[i].start = half_nodes[i].start;
			nodes[i].start_children = half_nodes[i].start_children;
			nodes[i].depth = half_nodes[i].dccc[0];
			nodes[i].count_children = half_nodes[i].dccc[1];
			nodes[i].count_leafs = half_nodes[i].dccc[2];
			nodes[i].count_merged = half_nodes[i].dccc[3];

			for (int j = 0; j < 4; j++)
			{
				boxes[i].minn[j] = half_boxes[i].minn[j];
				boxes[i].maxx[j] = half_boxes[i].maxx[j];
			}
		}
	}
}
	void HierarchyLoader::loadDynamic(const char* filename,
		std::vector<Eigen::Vector3f>& pos,
		std::vector<SHs>& shs,
		std::vector<float>& alphas,
		std::vector<Eigen::Vector3f>& scales,
		std::vector<Eigen::Vector4f>& rot,
		std::vector<HierarchyNode>& nodes,
		int& sh_degree)
	{
		std::ifstream infile(filename, std::ios_base::binary);

		if (!infile.good())
			throw std::runtime_error("File not found!");

		

		int number_of_gaussians;
		infile.read((char*)&number_of_gaussians, sizeof(int));
		int SH_Degree;
		infile.read((char*)&SH_Degree, sizeof(int));
		sh_degree = SH_Degree;
		

		pos.resize(number_of_gaussians);
		shs.resize(number_of_gaussians);
		alphas.resize(number_of_gaussians);
		scales.resize(number_of_gaussians);
		rot.resize(number_of_gaussians);


		
		infile.read((char*)pos.data(), number_of_gaussians * sizeof(Eigen::Vector3f));
		infile.read((char*)rot.data(), number_of_gaussians * sizeof(Eigen::Vector4f));
		infile.read((char*)scales.data(), number_of_gaussians * sizeof(Eigen::Vector3f));
		infile.read((char*)alphas.data(), number_of_gaussians * sizeof(float));
		if(SH_Degree == 0)
		{
			infile.read((char*)shs.data(), number_of_gaussians * sizeof(SHs0));
		}
		else if(SH_Degree == 1)
		{
			infile.read((char*)shs.data(), number_of_gaussians * sizeof(SHs1));
		}
		else if(SH_Degree == 2)
		{
			infile.read((char*)shs.data(), number_of_gaussians * sizeof(SHs2));
		}
		else if(SH_Degree == 3)
		{
			infile.read((char*)shs.data(), number_of_gaussians * sizeof(SHs3));
		}

		int number_of_nodes;
		infile.read((char*)&number_of_nodes, sizeof(int));

		number_of_nodes = number_of_gaussians;
		nodes.resize(number_of_nodes);
		printf("G: %d, SH: %d, N: %d\n", number_of_gaussians, SH_Degree, number_of_nodes);


		infile.read((char*)nodes.data(), number_of_nodes * sizeof(HierarchyNode));
	}

