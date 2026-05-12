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


#include "hierarchy_writer.h"
#include <vector>
#include <Eigen/Dense>
#include "common.h"
#include <iostream>
#include <fstream>
#include "half.hpp"

struct HalfBox
{
	half_float::half minn[4];
	half_float::half maxx[4];
};

void HierarchyWriter::write(const char* filename,
	int allG, int allNB,
	Eigen::Vector3f* positions,
	SHs* shs,
	float* opacities,
	Eigen::Vector3f* log_scales,
	Eigen::Vector4f* rotations,
	Node* nodes,
	Box* boxes,
	bool compressed)
{
	std::ofstream outfile(filename, std::ios_base::binary);

	if (!outfile.good())
		throw std::runtime_error("File not created!");

	size_t allP = allG;
	size_t allN = allNB;

	if (!compressed)
	{
		outfile.write((char*)(&allG), sizeof(int));
		outfile.write((char*)positions, allP * sizeof(Eigen::Vector3f));
		outfile.write((char*)rotations, allP * sizeof(Eigen::Vector4f));
		outfile.write((char*)log_scales, allP * sizeof(Eigen::Vector3f));
		outfile.write((char*)opacities, allP * sizeof(float));
		outfile.write((char*)shs, allP * sizeof(SHs));

		outfile.write((char*)(&allNB), sizeof(int));
		outfile.write((char*)nodes, allN * sizeof(Node));
		outfile.write((char*)boxes, allN * sizeof(Box));
	}
	else
	{
		int indi = -allG;
		outfile.write((char*)(&indi), sizeof(int));

		std::vector<half_float::half> half_rotations(allP * 4);
		std::vector<half_float::half> half_scales(allP * 3);
		std::vector<half_float::half> half_opacities(allP);
		std::vector<half_float::half> half_shs(allP * 48);

		int checksum = 0;

		for (size_t i = 0; i < allP; i++)
		{
			for (size_t j = 0; j < 4; j++)
				half_rotations[i * 4 + j] = rotations[i][j];
			for (size_t j = 0; j < 3; j++)
				half_scales[i * 3 + j] = log_scales[i][j];
			half_opacities[i] = opacities[i];
			for (size_t j = 0; j < 48; j++)
				half_shs[i * 48 + j] = shs[i][j];
		}
		outfile.write((char*)positions, allP * sizeof(Eigen::Vector3f));
		outfile.write((char*)half_rotations.data(), allP * 4 * sizeof(half_float::half));
		outfile.write((char*)half_scales.data(), allP * 3 * sizeof(half_float::half));
		outfile.write((char*)half_opacities.data(), allP * sizeof(half_float::half));
		outfile.write((char*)half_shs.data(), allP * 48 * sizeof(half_float::half));

		checksum += allP * 4 * 3 + allP * 8 + allP * 6 + allP * 2 + allP * 96;
		std::cout << checksum / (1024 * 1024) << " " << checksum / (1000000) << std::endl;

		std::vector<HalfNode> half_nodes(allN);
		std::vector<HalfBox> half_boxes(allN);

		for (int i = 0; i < allN; i++)
		{
			half_nodes[i].parent = nodes[i].parent;
			half_nodes[i].start = nodes[i].start;
			half_nodes[i].start_children = nodes[i].start_children;
			if (nodes[i].depth > 32000 || nodes[i].count_children > 32000 || nodes[i].count_leafs > 32000 || nodes[i].count_merged > 32000)
				throw std::runtime_error("Would lose information!");
			half_nodes[i].dccc[0] = (short)nodes[i].depth;
			half_nodes[i].dccc[1] = (short)nodes[i].count_children;
			half_nodes[i].dccc[2] = (short)nodes[i].count_leafs;
			half_nodes[i].dccc[3] = (short)nodes[i].count_merged;

			for (int j = 0; j < 4; j++)
			{
				half_boxes[i].minn[j] = boxes[i].minn[j];
				half_boxes[i].maxx[j] = boxes[i].maxx[j];
			}
		}

		outfile.write((char*)(&allNB), sizeof(int));
		outfile.write((char*)half_nodes.data(), allN * sizeof(HalfNode));
		outfile.write((char*)half_boxes.data(), allN * sizeof(HalfBox));

		checksum += allN * sizeof(HalfNode) + allN * sizeof(HalfBox);
		std::cout << checksum / (1024 * 1024) << " " << checksum / (1000000) << std::endl;
	}
}


void HierarchyWriter::writeDynamic(const char* filename,
	int number_of_gaussians, int number_of_nodes,
	Eigen::Vector3f* positions,
	void* shs,
	float* opacities,
	Eigen::Vector3f* log_scales,
	Eigen::Vector4f* rotations,
	HierarchyNode* nodes, int sh_degree)
{
	std::ofstream outfile(filename, std::ios_base::binary);

	if (!outfile.good())
		throw std::runtime_error("File not created!");


	int sh_sizes [] = {sizeof(SHs0), sizeof(SHs1), sizeof(SHs2), sizeof(SHs3)};

	outfile.write((char*)(&number_of_gaussians), sizeof(int));
	outfile.write((char*)(&sh_degree), sizeof(int));
	
	outfile.write((char*)positions, number_of_gaussians * sizeof(Eigen::Vector3f));
	outfile.write((char*)rotations, number_of_gaussians * sizeof(Eigen::Vector4f));
	outfile.write((char*)log_scales, number_of_gaussians * sizeof(Eigen::Vector3f));
	outfile.write((char*)opacities, number_of_gaussians * sizeof(float));

	//  For the record, I am aware that this is stupid
	if(sh_degree == 0)
	{
		outfile.write((char*)shs, number_of_gaussians * sizeof(SHs0));
	}
	else if(sh_degree == 1)
	{
		outfile.write((char*)shs, number_of_gaussians * sizeof(SHs1));
	}
	else if(sh_degree == 2)
	{
		outfile.write((char*)shs, number_of_gaussians * sizeof(SHs2));
	}
	else if(sh_degree == 3)
	{
		outfile.write((char*)shs, number_of_gaussians * sizeof(SHs3));
	}

	outfile.write((char*)(&number_of_nodes), sizeof(int));
	outfile.write((char*)nodes, number_of_nodes * sizeof(HierarchyNode));
}