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


#include "writer.h"
#include <iostream>
#include <fstream>
#include "hierarchy_writer.h"
#include <map>
#include <cassert>

// populate entries of flattened hierarchy in DFS order
void populateRec(
	const ExplicitTreeNode* treenode,
	int id,
	const std::vector<Gaussian>& gaussians, 
	std::vector<Eigen::Vector3f>& positions,
	std::vector<Eigen::Vector4f>& rotations,
	std::vector<Eigen::Vector3f>& log_scales,
	std::vector<float>& opacities,
	std::vector<SHs>& shs,
	std::vector<Node>& basenodes,
	std::vector<Box>& boxes,
	std::map<int, const ExplicitTreeNode*>* base2tree = nullptr)
{
	if(base2tree)
		base2tree->insert(std::make_pair(id, treenode));

	boxes[id] = treenode->bounds;
	basenodes[id].start = positions.size();
	for (auto& i : treenode->leaf_indices)
	{
		const Gaussian& g = gaussians[i];
		positions.push_back(g.position);
		rotations.push_back(g.rotation);
		log_scales.push_back(g.scale.array().log());
		opacities.push_back(g.opacity);
		shs.push_back(g.shs);
	}
	assert(treenode->leaf_indices.size() <= 1);
	basenodes[id].count_leafs = treenode->leaf_indices.size();

	for (auto& g : treenode->merged)
	{
		positions.push_back(g.position);
		rotations.push_back(g.rotation);
		log_scales.push_back(g.scale.array().log());
		opacities.push_back(g.opacity);
		shs.push_back(g.shs);
	}
	assert(treenode->merged.size() <= 1);

	basenodes[id].count_merged = treenode->merged.size();
	if(treenode->leaf_indices.size() == 0)
	{
		basenodes[id].start_children = -1;
	}
	else
	{
		basenodes[id].start_children = basenodes.size();
		for (int n = 0; n < treenode->children.size(); n++)
		{
			basenodes.push_back(Node());
			basenodes.back().parent = id;
			boxes.push_back(Box());
		}
		basenodes[id].count_children = treenode->children.size();
	}
	

	basenodes[id].depth = treenode->depth;

	for (int n = 0; n < treenode->children.size(); n++)
	{
		populateRec(
			treenode->children[n],
			basenodes[id].start_children + n,
			gaussians, 
			positions, 
			rotations, 
			log_scales, 
			opacities,
			shs, 
			basenodes,
			boxes,
			base2tree);
	}
}

// populate entries of flattened hierarchy in Preorder
void populateDynamicRec(
	const ExplicitTreeNode* treenode,
	int id,
	const std::vector<Gaussian>& gaussians, 
	std::vector<Eigen::Vector3f>& positions,
	std::vector<Eigen::Vector4f>& rotations,
	std::vector<Eigen::Vector3f>& log_scales,
	std::vector<float>& opacities,
	std::vector<SHs>& shs,
	std::vector<HierarchyNode>& basenodes, int depth,
	std::map<int, const ExplicitTreeNode*>* base2tree = nullptr)
{
	if(base2tree)
		base2tree->insert(std::make_pair(id, treenode));
	// Add exactly one Gaussian for each node
	for (auto& i : treenode->leaf_indices)
	{
		const Gaussian& g = gaussians[i];
		positions.push_back(g.position);
		rotations.push_back(g.rotation);
		log_scales.push_back(g.scale.array().log());
		opacities.push_back(g.opacity);
		shs.push_back(g.shs);

		basenodes[id].max_side_length = i;
	}
	assert(treenode->leaf_indices.size() + treenode->merged.size() == 1);
	for (auto& g : treenode->merged)
	{
		positions.push_back(g.position);
		rotations.push_back(g.rotation);
		log_scales.push_back(g.scale.array().log());
		opacities.push_back(g.opacity);
		shs.push_back(g.shs);

		basenodes[id].max_side_length = -1;
	}

	basenodes[id].first_child = basenodes.size();

	//for (int n = 0; n < treenode->children.size(); n++)
	//{
		
	//}
	basenodes[id].child_count = treenode->children.size();

	basenodes[id].depth = depth;
	//std::cout << treenode->depth  << std::endl;
	//std::cout << id << std::endl;
	//std::cout << treenode->depth << std::endl;
	for (int n = 0; n < treenode->children.size(); n++)
	{
		basenodes.push_back(HierarchyNode());
		int prev_node = basenodes.size()-1;
		basenodes[prev_node].parent = id;
		
		populateDynamicRec(
			treenode->children[n],
			basenodes.size()-1,
			gaussians, 
			positions, 
			rotations, 
			log_scales, 
			opacities,
			shs, 
			basenodes, depth + 1,
			base2tree);
		if (n < treenode->children.size()-1)
			basenodes[prev_node].next_sibling = basenodes.size();
		else
			basenodes[prev_node].next_sibling = 0;
	}
}

void recTraverse(int id, std::vector<Node>& nodes, int& count)
{
	if (nodes[id].depth == 0)
		count++;
	if (nodes[id].count_children != 0 && nodes[id].depth == 0)
		throw std::runtime_error("An error occurred in traversal");
	for (int i = 0; i < nodes[id].count_children; i++)
	{
		recTraverse(nodes[id].start_children + i, nodes, count);
	}
}


void Writer::makeDynamicHierarchy(
	const std::vector<Gaussian>& gaussians,
	ExplicitTreeNode* root,
	std::vector<Eigen::Vector3f>& positions,
	std::vector<Eigen::Vector4f>& rotations,
	std::vector<Eigen::Vector3f>& log_scales,
	std::vector<float>& opacities,
	std::vector<SHs>& shs,
	std::vector<HierarchyNode>& basenodes,
	std::map<int, const ExplicitTreeNode*>* base2tree)
{
	basenodes.resize(1);
	root->depth = 0;
	populateDynamicRec(
		root,
		0,
		gaussians,
		positions, rotations, log_scales, opacities, shs, basenodes, 0, base2tree);
}


void Writer::makeHierarchy(
	const std::vector<Gaussian>& gaussians,
	const ExplicitTreeNode* root,
	std::vector<Eigen::Vector3f>& positions,
	std::vector<Eigen::Vector4f>& rotations,
	std::vector<Eigen::Vector3f>& log_scales,
	std::vector<float>& opacities,
	std::vector<SHs>& shs,
	std::vector<Node>& basenodes,
	std::vector<Box>& boxes,
	std::map<int, const ExplicitTreeNode*>* base2tree)
{
	basenodes.resize(1);
	boxes.resize(1);

	populateRec(
		root,
		0,
		gaussians,
		positions, rotations, log_scales, opacities, shs, basenodes, boxes,
		base2tree);
}

void Writer::writeHierarchy(const char* filename, const std::vector<Gaussian>& gaussians, ExplicitTreeNode* root, bool compressed)
{
	std::vector<Eigen::Vector3f> positions;
	std::vector<Eigen::Vector4f> rotations;
	std::vector<Eigen::Vector3f> log_scales;
	std::vector<float> opacities;
	std::vector<SHs> shs;
	std::vector<Node> basenodes;
	std::vector<Box> boxes;

	makeHierarchy(gaussians, root, positions, rotations, log_scales, opacities, shs, basenodes, boxes);

	HierarchyWriter writer;
	writer.write(
		filename,
		positions.size(),
		basenodes.size(),
		positions.data(),
		shs.data(),
		opacities.data(),
		log_scales.data(),
		rotations.data(),
		basenodes.data(),
		boxes.data(),
		compressed
	);
	std::string graphFileName = std::string(filename);
	graphFileName.pop_back();
	graphFileName.pop_back();
	graphFileName.pop_back();
	graphFileName.pop_back();
	graphFileName.append("gdf");
	writeHierarchyGDF(graphFileName.c_str(), root, 3000);
}


void Writer::writeDynamicHierarchy(const char* filename, const std::vector<Gaussian>& gaussians, ExplicitTreeNode* root)
{
	std::vector<Eigen::Vector3f> positions;
	std::vector<Eigen::Vector4f> rotations;
	std::vector<Eigen::Vector3f> log_scales;
	std::vector<float> opacities;
	std::vector<SHs> shs;
	std::vector<HierarchyNode> basenodes;

	makeDynamicHierarchy(gaussians, root, positions, rotations, log_scales, opacities, shs, basenodes, nullptr);
	HierarchyWriter writer;
	writer.writeDynamic(
		filename,
		positions.size(),
		basenodes.size(),
		positions.data(),
		shs.data(),
		opacities.data(),
		log_scales.data(),
		rotations.data(),
		basenodes.data(), 3
	);
}

void writeRec(std::ofstream& outfile, const ExplicitTreeNode* node, int* index, int parent_index, std::vector<std::pair<int, int>>& edges, int depth, int max_depth)
{
	
	std::string s = std::to_string(*index);
	outfile.write(s.c_str(), s.size());
	outfile.write("\n", 1);
	if(node->children.size() == 0 || depth >= max_depth)
	{
		return;
	}	
	for(ExplicitTreeNode* child : node->children)
	{
		edges.push_back(std::make_pair(*index, parent_index));
		int current_index = *index;
		*index += 1;
		writeRec(outfile, child, index, current_index, edges, depth+1, max_depth);
	}
}	

void Writer::writeHierarchyGDF(const char* filename, const ExplicitTreeNode* root, int max_depth)
{
	std::ofstream outfile(filename);

	if (!outfile.good())
		throw std::runtime_error("File not created!");
	std::string node_header = "nodedef>name VARCHAR \n";
	outfile.write(node_header.c_str(), node_header.size());
	//outfile.write(",", 1);
	//outfile << root->merged.;
	std::vector<std::pair<int, int>> edges; 
	int index = 0;
	writeRec(outfile, root, &index, -1, edges, 0, max_depth);

	std::string edge_header = "edgedef>node1 VARCHAR,node2 VARCHAR\n";
	outfile.write(edge_header.c_str(), edge_header.size());
	for(std::pair<int, int> edge : edges)
	{
		std::string s = std::to_string(edge.first);
		outfile.write(s.c_str(), s.size());
		outfile.write(",", 1);
		s = std::to_string(edge.second);
		outfile.write(s.c_str(), s.size());
		outfile.write("\n", 1);
	}
	outfile.close();
}





