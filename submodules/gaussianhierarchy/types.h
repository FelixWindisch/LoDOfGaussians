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

#ifndef __CUDACC__

#include <Eigen/Core>

struct Box
{
	Box(Eigen::Vector3f minn, Eigen::Vector3f maxx) : minn(minn.x(), minn.y(), minn.z(), 0), maxx(maxx.x(), maxx.y(), maxx.z(), 0)
	{}

	Box() {};

	Eigen::Vector4f minn;
	Eigen::Vector4f maxx;
};



typedef Eigen::Matrix<float, 48, 1> SHs;
typedef Eigen::Matrix<float, 48, 1> SHs3;
typedef Eigen::Matrix<float, 27, 1> SHs2;
typedef Eigen::Matrix<float, 12, 1> SHs1;
typedef Eigen::Matrix<float, 3, 1> SHs0;

#else



struct Point
{
	float xyz[3];
};

struct Point4
{
	float xyz[4];
};

struct Box
{
	Point4 minn;
	Point4 maxx;
};

#endif

// By Felix
struct HierarchyNode
{
	int depth = -1;
	int parent = -1;
	int child_count;
	int first_child;
	int next_sibling;
	int max_side_length;

	friend std::ostream& operator<<(std::ostream& os, const HierarchyNode& node) {
	os << "==============================="  << "\n";
    os << "depth: " << node.depth << "\n";
	os << "parent: " << node.parent << "\n";
	os << "child_count: " << node.child_count << "\n";
	os << "first_child: " << node.first_child << "\n";
	os << "next_sibling: " << node.next_sibling << "\n";
	os << "==============================="  << "\n";
    return os;
}
};



struct Node
{
	int depth = -1;
	int parent = -1;
	int start;
	int count_leafs;
	int count_merged;
	int start_children;
	int count_children;
};

struct HalfNode
{
	int parent = -1;
	int start;
	int start_children;
	short dccc[4];
};