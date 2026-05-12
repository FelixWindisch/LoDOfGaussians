#ifndef CUDA_RASTERIZER_MORTON_H_INCLUDED
#define CUDA_RASTERIZER_MORTON_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//#define GLM_FORCE_CUDA
//#include <glm/glm.hpp>

namespace MORTON {
    void getMortonCode(
        const float* xyz,
        const float* min,
        const float* max,
        int64_t* codes,
        const int P);
}

#endif