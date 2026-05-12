//#include "auxiliary.h"
#include "morton.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#define SCALING (1 << 21)

__global__
void getMortonCodeCUDA(
    const float3* xyz,
    const float3* min,
    const float3* max,
    int64_t* codes,
    const int P) {

    auto idx = cg::this_grid().thread_rank();
    if (idx >= P) return;

    float3 mn = min[0];
    float3 mx = max[0];
    //TODO
    float3 box = {mx.x - mn.x, mx.y - mn.y, mx.z - mn.z}                                                                                     ;

    float3 point = xyz[idx];
    //point = (point - mn) / box;
    point = {(point.x - mn.x) / box.x, (point.y - mn.y) / box.y, (point.z - mn.z) / box.z};
    
    point = {point.x * SCALING, point.y * SCALING, point.z * SCALING};

    int64_t point_int64[3] = {point.x, point.y, point.z};
    int64_t code = 0;

    #pragma unroll
    for (int i = 0; i < 21; ++i) {
        code = code | (point_int64[0] >> i & 1) << (3 * i);
        code = code | (point_int64[1] >> i & 1) << (3 * i + 1);
        code = code | (point_int64[2] >> i & 1) << (3 * i + 2);
    }

    codes[idx] = code;
}

// xyz are the Gaussian positions, min and max are Vector3s that show the extent of the Gaussians
void MORTON::getMortonCode(
    const float* xyz,
    const float* min,
    const float* max,
    int64_t* codes,
    const int P
) {
    getMortonCodeCUDA<<<(P + 255)/256, 256>>>(
        (float3 *)xyz,
        (float3 *)min,
        (float3 *)max,
        codes,
        P
    );
}