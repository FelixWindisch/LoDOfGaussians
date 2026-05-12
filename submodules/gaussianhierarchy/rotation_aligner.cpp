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

#include "rotation_aligner.h"

float frobenius(Eigen::Matrix3f& A, Eigen::Matrix3f& B)
{
	float sum = 0.f;
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
			sum += A(i, j) * B(i, j);
	return sum;
}

void matchExhaustive(const Gaussian& ref, Gaussian& match)
{
	auto Rref = matrixFromQuat(ref.rotation);
	auto Rmatch = matrixFromQuat(match.rotation);

	auto ror = match.rotation;

	Cov pre;
	computeCovariance(match.scale, match.rotation, pre);

	bool noscore = true;
	float best_score = 0;
	int best_i, best_j, best_k;
	Eigen::Matrix3f base, test, best;
	for (int i = 0; i < 3; i++)
	{
		base.col(0) = Rmatch.col(i);
		for (int j = 0; j < 3; j++)
		{
			if (j == i)
				continue;

			base.col(1) = Rmatch.col(j);
			for (int k = 0; k < 3; k++)
			{
				if (k == j || k == i)
					continue;

				base.col(2) = Rmatch.col(k);

				int state = 0;
				while (state < 8)
				{
					for (int w = 0; w < 3; w++)
						if ((state & (1 << w)))
							test.col(w) = -base.col(w);
						else
							test.col(w) = base.col(w);

					state++;

					if (test.col(0).cross(test.col(1)).dot(test.col(2)) < 0)
						continue;

					float score = frobenius(test, Rref);
					if (score > best_score)
					{
						best_i = i;
						best_j = j;
						best_k = k;
						best = test;
						best_score = score;
						noscore = false;
					}
				}
			}
		}
	}

	auto q = Eigen::Quaternionf(best);
	match.rotation = { q.w(), q.x(), q.y(), q.z() };
	match.scale = Eigen::Vector3f(match.scale[best_i], match.scale[best_j], match.scale[best_k]);

	Cov cov;
	computeCovariance(match.scale, match.rotation, cov);
}

void topDownAlign(ExplicitTreeNode* node, std::vector<Gaussian>& gaussians)
{
	if (node->merged.size() != 0)
	{
		Gaussian& ref = node->merged[0];

		for (auto& child : node->children)
		{
			for (auto& m : child->merged)
			{
				matchExhaustive(ref, m);
			}
			for (auto& i : child->leaf_indices)
			{
				matchExhaustive(ref, gaussians[i]);
			}

			topDownAlign(child, gaussians);
		}
	}
}

void RotationAligner::align(ExplicitTreeNode* root, std::vector<Gaussian>& gaussians)
{
	topDownAlign(root, gaussians);
}