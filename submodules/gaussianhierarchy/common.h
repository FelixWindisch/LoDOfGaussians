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

#include <Eigen/Dense>
#include <iostream>
#include "types.h"
#include <float.h>
#include <memory>

static float sigmoid(const float m1)
{
	return 1.0f / (1.0f + exp(-m1));
}


typedef Eigen::Matrix<float, 6, 1> Cov;
typedef Eigen::Vector3f Point;

struct RichPoint
{
	Eigen::Vector3f position;
	Eigen::Vector3f normal;
	float shs[48];
	float opacity;
	Eigen::Vector3f scale;
	float rotation[4];
};

struct LessRichPoint
{
	Eigen::Vector3f position;
	Eigen::Vector3f normal;
	float shs[12];
	float opacity;
	Eigen::Vector3f scale;
	float rotation[4];
};

struct Gaussian
{
	Eigen::Vector3f position;
	SHs shs;
	float opacity;
	Eigen::Vector3f scale;
	Eigen::Vector4f rotation;
	Cov covariance;

	Box bounds99(Eigen::Vector3f& minn, Eigen::Vector3f& maxx) const
	{
		Eigen::Vector3f off(3.f * sqrt(covariance[0]), 3.f * sqrt(covariance[3]), 3.f * sqrt(covariance[5]));
		return Box(position - off, position + off);
	}

    friend std::ostream& operator<<(std::ostream& os, const Gaussian& G)
	{
		os << "Gaussian: ==================" << std::endl;
		os << "Position: " << G.position[0] << " " << G.position[1] << " " << G.position[2] << std::endl;
		os << "Scale: " << G.scale[0] << " " << G.scale[1] << " " << G.scale[2] << std::endl;
		os << "Opacity: " << G.opacity << std::endl;
    	return os;
	};


};


struct ExplicitTreeNode
{
	int depth = -1;
	Box bounds = Box({FLT_MAX, FLT_MAX, FLT_MAX}, {-FLT_MAX, -FLT_MAX, -FLT_MAX});
	std::vector<ExplicitTreeNode*> children;
	// why are these vectors???
	std::vector<int> leaf_indices;
	std::vector<Gaussian> merged;
};

static Box getBounds(const std::vector<Gaussian>& gaussians)
{
	Eigen::Vector3f minn(FLT_MAX, FLT_MAX, FLT_MAX);
	Eigen::Vector3f maxx = -minn;

	Eigen::Vector3f gMax, gMin;
	for (int i = 0; i < gaussians.size(); i++)
	{
		gaussians[i].bounds99(gMin, gMax);
		maxx = maxx.cwiseMax(gMax);
		minn = minn.cwiseMin(gMin);
	}

	return Box(minn, maxx);
}

static Eigen::Matrix3f matrixFromQuat(Eigen::Vector4f rot)
{
	float s = rot.x();
	float x = rot.y();
	float y = rot.z();
	float z = rot.w();

	Eigen::Matrix3f R;
	R <<
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - s * z), 2.f * (x * z + s * y),
		2.f * (x * y + s * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - s * x),
		2.f * (x * z - s * y), 2.f * (y * z + s * x), 1.f - 2.f * (x * x + y * y);

	return R;
}

static void computeCovariance(const Eigen::Vector3f& scale, const Eigen::Vector4f& rot, Cov& covariance, bool debug = false)
{
	Eigen::Matrix3f L = Eigen::Matrix3f::Identity();

	L(0, 0) = scale.x();
	L(1, 1) = scale.y();
	L(2, 2) = scale.z();

	auto R = matrixFromQuat(rot);

	Eigen::Matrix3f T = R * L;
	Eigen::Matrix3f Tf;
	Tf <<
		T(0, 0), T(1, 0), T(2, 0),
		T(0, 1), T(1, 1), T(2, 1),
		T(0, 2), T(1, 2), T(2, 2);
	Eigen::Matrix3f T2 = T * Tf;

	covariance[0] = T2(0, 0);
	covariance[1] = T2(0, 1);
	covariance[2] = T2(0, 2);
	covariance[3] = T2(1, 1);
	covariance[4] = T2(1, 2);
	covariance[5] = T2(2, 2);

	if (debug)
		std::cout << covariance << std::endl;
}

template <typename T, int Options>
Eigen::Quaternion<T, 0> quatFromMatrix(const Eigen::Matrix<T, 3, 3, Options, 3, 3>& m) 
{
	Eigen::Quaternion<T, 0> q;
	float trace = m(0, 0) + m(1, 1) + m(2, 2) + 1.f;
	if (trace > 0)
	{
		float s = 0.5f / sqrtf(trace);
		q.x() = (m(1, 2) - m(2, 1)) * s;
		q.y() = (m(2, 0) - m(0, 2)) * s;
		q.z() = (m(0, 1) - m(1, 0)) * s;
		q.w() = 0.25f / s;
	}
	else
	{
		if ((m(0, 0) > m(1, 1)) && (m(0, 0) > m(2, 2)))
		{
			float s = sqrtf(1.f + m(0, 0) - m(1, 1) - m(2, 2)) * 2.f;
			q.x() = 0.5f / s;
			q.y() = (m(1, 0) + m(0, 1)) / s;
			q.z() = (m(2, 0) + m(0, 2)) / s;
			q.w() = (m(2, 1) + m(1, 2)) / s;
		}
		else if (m(1, 1) > m(2, 2))
		{
			float s = sqrtf(1.f - m(0, 0) + m(1, 1) - m(2, 2)) * 2.f;
			q.x() = (m(1, 0) + m(0, 1)) / s;
			q.y() = 0.5f / s;
			q.z() = (m(2, 1) + m(1, 2)) / s;
			q.w() = (m(2, 0) + m(0, 2)) / s;
		}
		else
		{
			float s = sqrtf(1.f - m(0, 0) - m(1, 1) + m(2, 2)) * 2.f;
			q.x() = (m(2, 0) + m(0, 2)) / s;
			q.y() = (m(2, 1) + m(1, 2)) / s;
			q.z() = 0.5f / s;
			q.w() = (m(1, 0) + m(0, 1)) / s;
		}
	}
	return q;
}
