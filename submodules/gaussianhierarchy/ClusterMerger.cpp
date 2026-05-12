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

#include "ClusterMerger.h"
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

float ellipseSurface(Eigen::Vector3f scale)
{
	return scale[0] * scale[1] +
		scale[0] * scale[2] +
		scale[1] * scale[2];
}

void ClusterMerger::mergeRec(ExplicitTreeNode* node, const std::vector<Gaussian>& leaf_gaussians)
{
	Gaussian clustered;
	clustered.position = Eigen::Vector3f::Zero();
	clustered.rotation = Eigen::Vector4f::Zero();
	clustered.opacity = 0;
	clustered.scale = Eigen::Vector3f::Zero();
	clustered.shs = SHs::Zero();
	clustered.covariance = Cov::Zero();

	std::vector<const Gaussian*> toMerge;
	for (auto& child : node->children)
	{
		if (child->leaf_indices.size() == 0)
		{
			mergeRec(child, leaf_gaussians);
		}
		if(child->merged.size())
			toMerge.push_back(&child->merged[0]);

		for (auto& child_leaf : child->leaf_indices)
			toMerge.push_back(&leaf_gaussians[child_leaf]);
	}

	if (node->depth == 0)
		return;

	float weight_sum = 0;
	std::vector<float> weights;
	for (const Gaussian* g : toMerge)
	{
		float w = g->opacity * ellipseSurface(g->scale);
		weights.push_back(w);
		weight_sum += w;
	}
	for (int i = 0; i < weights.size(); i++)
		weights[i] = weights[i] / weight_sum;

	for (int i = 0; i < toMerge.size(); i++)
	{
		const Gaussian* g = toMerge[i];
		float a = weights[i];

		clustered.position += a * g->position;
		clustered.shs += a * g->shs;
	}

	for (int i = 0; i < toMerge.size(); i++)
	{
		const Gaussian* g = toMerge[i];
		float a = weights[i];

		Eigen::Vector3f diff = g->position - clustered.position;

		clustered.covariance[0] += a * (g->covariance[0] + diff.x() * diff.x());
		clustered.covariance[1] += a * (g->covariance[1] + diff.y() * diff.x());
		clustered.covariance[2] += a * (g->covariance[2] + diff.z() * diff.x());
		clustered.covariance[3] += a * (g->covariance[3] + diff.y() * diff.y());
		clustered.covariance[4] += a * (g->covariance[4] + diff.z() * diff.y());
		clustered.covariance[5] += a * (g->covariance[5] + diff.z() * diff.z());
	}

	Eigen::Matrix3f matrix;
	matrix <<
		clustered.covariance[0], clustered.covariance[1], clustered.covariance[2],
		clustered.covariance[1], clustered.covariance[3], clustered.covariance[4],
		clustered.covariance[2], clustered.covariance[4], clustered.covariance[5];

	

	Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigensolver(matrix);
	auto eigenvalues = eigensolver.eigenvalues();
	auto eigenvectors = eigensolver.eigenvectors();

	for (int i = 0; eigenvalues.hasNaN() && i < 5; i++)
	{
		throw std::runtime_error("Found Nans!");
	}

	int loops = 0;
	while(eigenvalues[0] == 0 || eigenvalues[1] == 0 || eigenvalues[2] == 0)
	{
		matrix(0, 0) += std::max(matrix(0, 0) * 0.0001f, std::numeric_limits<float>::epsilon());
		matrix(1, 1) += std::max(matrix(1, 1) * 0.0001f, std::numeric_limits<float>::epsilon());
		matrix(2, 2) += std::max(matrix(2, 2) * 0.0001f, std::numeric_limits<float>::epsilon());

		Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigensolver2(matrix);
		eigenvalues = eigensolver2.eigenvalues();
		eigenvectors = eigensolver2.eigenvectors();

		loops++;
		if (loops % 10 == 0)
			std::cout << "Working hard..." << std::endl;
	}

	auto v1 = eigenvectors.col(0);
	auto v2 = eigenvectors.col(1);
	auto v3 = eigenvectors.col(2);

	auto test = v1.cross(v2);
	if (test.dot(v3) < 0)
		eigenvectors.col(2) *= -1;

	float a = sqrt(abs(eigenvalues.x()));
	float b = sqrt(abs(eigenvalues.y()));
	float c = sqrt(abs(eigenvalues.z()));

	auto q = quatFromMatrix(eigenvectors);

	clustered.scale = { a, b, c };
	clustered.rotation = { q.w(), -q.x(), -q.y(), -q.z() };

	auto q2 = Eigen::Quaternionf(eigenvectors);
	clustered.rotation = { q2.w(), q2.x(), q2.y(), q2.z() };

	clustered.opacity = weight_sum / (ellipseSurface(clustered.scale));

	node->merged.push_back(clustered);

	// The rest here just finds the longest box side length
	Gaussian g;
	if (node->depth == 0)
	{
		g = leaf_gaussians[node->leaf_indices[0]];
	}
	else
	{
		g = node->merged[0];
	}
	float minextent = g.scale.array().minCoeff();
	float maxextent = g.scale.array().maxCoeff();

	if (node->depth != 0)
	{
		for (int i = 0; i < 2; i++)
		{
			minextent = std::max(minextent, node->children[i]->bounds.minn.w());
			maxextent = std::max(maxextent, node->children[i]->bounds.maxx.w());
		}
	}

	auto diff = node->bounds.maxx - node->bounds.minn;
	// Set the w value to the longest box size for projection later
	node->bounds.minn.w() = std::max(std::max(diff.x(), diff.y()), diff.z());
	node->bounds.maxx.w() = std::max(std::max(diff.x(), diff.y()), diff.z());
}

void ClusterMerger::merge(ExplicitTreeNode* root, const std::vector<Gaussian>& leaf_gaussians)
{
	mergeRec(root, leaf_gaussians);
}