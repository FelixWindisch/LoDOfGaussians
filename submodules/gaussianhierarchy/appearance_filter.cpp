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

#include "appearance_filter.h"
#include <vector>
#include <Eigen/Dense>
#include "common.h"
#include <iostream>
#include <fstream>
#include <map>
#include "writer.h"
#include <numeric>
#include "runtime_switching.h"

template <typename T>
T ReadBinaryLittleEndian(std::ifstream* infile)
{
	T val;
	infile->read((char*)&val, sizeof(T));
	return val;
}

template <typename T>
void ReadBinaryLittleEndian(std::ifstream* infile, std::vector<T>* vals)
{
	infile->read((char*)vals->data(), sizeof(T) * vals->size());
}

void AppearanceFilter::init(const char* colmappath)
{
	const std::string basePathName = std::string(colmappath) + "/sparse/0/";
	const std::string camerasListing = basePathName + "/cameras.bin";
	const std::string imagesListing = basePathName + "/images.bin";

	std::ifstream camerasFile(camerasListing, std::ios::binary);
	std::ifstream imagesFile(imagesListing, std::ios::binary);

	if (camerasFile.good())
	{
		std::string line;

		std::map<size_t, CameraParametersColmap> cameraParameters;
		const size_t num_cameras = ReadBinaryLittleEndian<uint64_t>(&camerasFile);

		for (size_t i = 0; i < num_cameras; ++i)
		{
			CameraParametersColmap params;

			params.id = ReadBinaryLittleEndian<uint32_t>(&camerasFile);
			int model_id = ReadBinaryLittleEndian<int>(&camerasFile);
			params.width = ReadBinaryLittleEndian<uint64_t>(&camerasFile);
			params.height = ReadBinaryLittleEndian<uint64_t>(&camerasFile);
			std::vector<double> Params(4);

			ReadBinaryLittleEndian<double>(&camerasFile, &Params);
			params.fx = float(Params[0]);
			params.fy = float(Params[1]);
			params.dx = float(Params[2]);
			params.dy = float(Params[3]);
			cameraParameters[params.id] = params;
		}

		const size_t num_reg_images = ReadBinaryLittleEndian<uint64_t>(&imagesFile);
		for (size_t i = 0; i < num_reg_images; ++i)
		{
			unsigned int	cId = ReadBinaryLittleEndian<unsigned int>(&imagesFile);
			float			qw = float(ReadBinaryLittleEndian<double>(&imagesFile));
			float			qx = float(ReadBinaryLittleEndian<double>(&imagesFile));
			float			qy = float(ReadBinaryLittleEndian<double>(&imagesFile));
			float			qz = float(ReadBinaryLittleEndian<double>(&imagesFile));
			float			tx = float(ReadBinaryLittleEndian<double>(&imagesFile));
			float			ty = float(ReadBinaryLittleEndian<double>(&imagesFile));
			float			tz = float(ReadBinaryLittleEndian<double>(&imagesFile));
			size_t			id = ReadBinaryLittleEndian<unsigned int>(&imagesFile);

			char name_char;
			do {
				imagesFile.read(&name_char, 1);
			} while (name_char != '\0');

			// ignore the 2d points
			const size_t num_points2D = ReadBinaryLittleEndian<uint64_t>(&imagesFile);
			for (size_t j = 0; j < num_points2D; ++j) {
				ReadBinaryLittleEndian<double>(&imagesFile);
				ReadBinaryLittleEndian<double>(&imagesFile);
				ReadBinaryLittleEndian<uint64_t>(&imagesFile);
			}

			if (cameraParameters.find(id) == cameraParameters.end())
			{
				id = 1;
			}

			const Eigen::Quaternionf quat(qw, qx, qy, qz);
			const Eigen::Matrix3f orientation = quat.toRotationMatrix().transpose();
			Eigen::Vector3f translation(tx, ty, tz);

			Eigen::Vector3f position = -(orientation * translation);

			cameras.push_back({ cameraParameters[id], position });
		}
	}
}

bool verify_rec(const ExplicitTreeNode* node, const std::map<const ExplicitTreeNode*, int>& tree2base, const std::vector<int>& seen, int parent_seen)
{
	auto entry = tree2base.find(node);
	if (entry == tree2base.end())
		throw std::runtime_error("Looking for an entry that does not exist in tree!");

	int id = entry->second;
	
	if (seen[id])
	{
		if (parent_seen != -1)
		{
			return false;
		}
		parent_seen = id;
	}

	for (auto child : node->children)
	{
		if (!verify_rec(child, tree2base, seen, parent_seen))
			return false;
	}
	return true;
}



// returns in bottom, all nodes that are marked in seen and all nodes, where all child leaves are seen.
bool bottomRec(ExplicitTreeNode* node, const std::map<const ExplicitTreeNode*, int>& tree2base, const std::vector<int>& seen, std::vector<ExplicitTreeNode*>& bottom)
{
	auto entry = tree2base.find(node);
	if (entry == tree2base.end())
		throw std::runtime_error("Looking for an entry that does not exist in tree!");

	int id = entry->second;

	if (node->children.size() != 0)
	{
		bool some = false;
		bool all = true;
		for (auto child : node->children)
		{
			bool result = bottomRec(child, tree2base, seen, bottom);
			all &= result;
			some |= result;
		}

		if (some && !all)
			throw std::runtime_error("Incomplete cut!");

		if (all)
			return true;
	}

	if (seen[id])
	{
		bottom.push_back(node);
		return true;
	}
	else
		return false;
}

// returns all marked nodes and their children in bottomandbelow
void andBelowRec(ExplicitTreeNode* node, const std::map<const ExplicitTreeNode*, int>& tree2base, const std::vector<int>& marked, std::vector<ExplicitTreeNode*>& bottomandbelow, bool bebelow = false)
{
	auto entry = tree2base.find(node);
	if (entry == tree2base.end())
		throw std::runtime_error("Looking for an entry that does not exist in tree!");

	int id = entry->second;

	if (marked[id])
		bebelow = true;

	if (bebelow)
		bottomandbelow.push_back(node);

	if (node->children.size() != 0)
	{
		for (auto child : node->children)
		{
			andBelowRec(child, tree2base, marked, bottomandbelow, bebelow);
		}
	}
}

void recCollapse(
	ExplicitTreeNode* topnode,
	ExplicitTreeNode* node,
	std::map<const ExplicitTreeNode*, int>& tree2base,
	std::vector<int>& marked)
{
	auto it = tree2base.find(node);
	if (it == tree2base.end())
		throw std::runtime_error("Looking for an entry that does not exist in tree!");
	// dont collapse marked or leaf nodes
	if (marked[it->second] || node->depth == 0)
	{		
		topnode->children.push_back(node);
	}
	else
	{
		for (auto& child : node->children)
			recCollapse(topnode, child, tree2base, marked);
	}
}

void collapseUnused(
	std::vector<ExplicitTreeNode*>& bottom,
	std::map<const ExplicitTreeNode*, int>& tree2base,
	std::vector<int>& marked
	)
{
	for (auto& node : bottom)
	{
		auto it = tree2base.find(node);
		if (it == tree2base.end())
			throw std::runtime_error("Looking for an entry that does not exist in tree!");
		// don't collapse if marked or leaf node
		if (node->depth == 0 || marked[it->second])
			continue;

		auto backup = node->children;
		node->children.clear();
		std::cout << "collapse!\n";
		for (auto& child : backup)
			recCollapse(node, child, tree2base, marked);
	}
}

/// only counts number of nodes and leaves
void recVisitAndCount(ExplicitTreeNode* node, int& nodes, int& leaves)
{
	nodes++;
	if (node->depth == 0)
		leaves++;
	for (auto& child : node->children)
	{
		recVisitAndCount(child, nodes, leaves);
	}
}

// computes the size of each node relative to their parent in sizes (DFS order) 
void recRelSizes(ExplicitTreeNode* node, std::vector<float>& sizes, float parentsize)
{
	auto extent = node->bounds.maxx - node->bounds.minn;
	Eigen::Vector3f extent3 = { extent.x(), extent.y(), extent.z() };
	float mysize = extent3.norm();

	if (parentsize != -1)
	{
		sizes.push_back(mysize / parentsize);
	}

	for (auto child : node->children)
		recRelSizes(child, sizes, mysize);
}

void AppearanceFilter::filter(ExplicitTreeNode* root, const std::vector<Gaussian>& gaussians, float orig_limit, float layermultiplier)
{
	{
		std::vector<Eigen::Vector3f> positions;
		std::vector<Eigen::Vector4f> rotations;
		std::vector<Eigen::Vector3f> log_scales;
		std::vector<float> opacities;
		std::vector<SHs> shs;
		std::vector<Node> basenodes;
		std::vector<Box> boxes;

		std::map<int, const ExplicitTreeNode*> base2tree;

		Writer::makeHierarchy(
			gaussians,
			root,
			positions,
			rotations,
			log_scales,
			opacities,
			shs,
			basenodes,
			boxes,
			&base2tree);

		std::map<const ExplicitTreeNode*, int> tree2base;
		for (auto entry : base2tree)
		{
			tree2base.insert(std::make_pair(entry.second, entry.first));
		}

		std::vector<Point> campositions;
		for (int i = 0; i < cameras.size(); i++)
		{
			campositions.push_back(cameras[i].position);
		}

		std::vector<int> seen(basenodes.size());
		std::vector<int> marked(basenodes.size(), 0);

		int nodes_count = 0, leaves_count = 0;
		recVisitAndCount(root, nodes_count, leaves_count);

		int last_bottom_size = 0;
		// limit == target granularity
		float limit = orig_limit;
		


		while (true)
		{

			// ============ I think this is only for debugging purposes?
			std::vector<float> v;
			recRelSizes(root, v, -1);

			double sum = std::accumulate(std::begin(v), std::end(v), 0.0);
			double mu = sum / v.size();
			double accum = 0.0;
			std::for_each(std::begin(v), std::end(v), [&](const double d) {
				accum += (d - mu) * (d - mu);
				});
			double stdev = sqrt(accum / (v.size() - 1));
			std::cout << "Child-to-parent size relation (mean, std):" << mu << " " << stdev << std::endl;
			//================= end


			// set a flag all Gaussians that are seen from some viewpoint in seen
			Switching::markVisibleForAllViewpoints(limit,
				(int*)basenodes.data(),
				basenodes.size(),
				(float*)boxes.data(),
				(float*)campositions.data(),
				campositions.size(),
				seen.data(),
				0, 0, 0
			);

			std::vector<ExplicitTreeNode*> bottom;
			bottomRec(root, tree2base, seen, bottom);

			if (limit > 1)
				break;

			//entirely unnecessary
			last_bottom_size = bottom.size();

			collapseUnused(bottom, tree2base, marked);

			// all nodes of bottom are marked
			for (int i = 0; i < bottom.size(); i++)
				marked[tree2base[bottom[i]]] = 1;

			nodes_count = 0;
			leaves_count = 0;
			recVisitAndCount(root, nodes_count, leaves_count);
			std::cout << "After collapse: " << nodes_count << " nodes, " << leaves_count << " leaves reachable" << std::endl;

			limit *= layermultiplier;
			//marked is not reset
		}
	}
}


void AppearanceFilter::writeAnchors(const char* filename, ExplicitTreeNode * root, const std::vector<Gaussian>&gaussians, float limit)
{
	std::cout << "Identifying and writing anchor points" << std::endl;

	std::vector<Eigen::Vector3f> positions;
	std::vector<Eigen::Vector4f> rotations;
	std::vector<Eigen::Vector3f> log_scales;
	std::vector<float> opacities;
	std::vector<SHs> shs;
	std::vector<Node> basenodes;
	std::vector<Box> boxes;

	std::map<int, const ExplicitTreeNode*> base2tree;

	Writer::makeHierarchy(
		gaussians,
		root,
		positions,
		rotations,
		log_scales,
		opacities,
		shs,
		basenodes,
		boxes,
		&base2tree);

	std::map<const ExplicitTreeNode*, int> tree2base;
	for (auto entry : base2tree)
		tree2base.insert(std::make_pair(entry.second, entry.first));

	std::vector<Point> campositions;
	for (int i = 0; i < cameras.size(); i++)
		campositions.push_back(cameras[i].position);

	std::vector<int> seen(basenodes.size());
	std::vector<int> marked(basenodes.size(), 0);

	Switching::markVisibleForAllViewpoints(limit,
		(int*)basenodes.data(),
		basenodes.size(),
		(float*)boxes.data(),
		(float*)campositions.data(),
		campositions.size(),
		seen.data(),
		0, 0, 0
	);

	std::vector<ExplicitTreeNode*> bottom;
	// find bottom cut of visible nodes
	bottomRec(root, tree2base, seen, bottom);

	// bottom nodes are marked
	for (int i = 0; i < bottom.size(); i++)
		marked[tree2base[bottom[i]]] = 1;

	// anchors are all bottom nodes and their children
	std::vector<ExplicitTreeNode*> bottomandbelow;
	andBelowRec(root, tree2base, marked, bottomandbelow);

	std::ofstream anchors(filename, std::ios_base::binary);

	int size = 0;
	for (int i = 0; i < bottomandbelow.size(); i++)
	{
		int index = tree2base[bottomandbelow[i]];
		size += basenodes[index].count_leafs + basenodes[index].count_merged;
	}
	anchors.write((char*)&size, sizeof(int));

	for (int i = 0; i < bottomandbelow.size(); i++)
	{
		int index = tree2base[bottomandbelow[i]];

		for (int j = basenodes[index].start; j < basenodes[index].start + basenodes[index].count_leafs + basenodes[index].count_merged; j++)
		{
			anchors.write((char*)&j, sizeof(int));
		}
	}
}