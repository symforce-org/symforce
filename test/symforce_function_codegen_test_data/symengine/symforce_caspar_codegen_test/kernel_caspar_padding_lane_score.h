#pragma once

#include <cuda_runtime.h>

#include "shared_indices.h"

namespace caspar {

void CasparPaddingLaneScore(float* cam_T_world, unsigned int cam_T_world_num_alloc,
                            SharedIndex* cam_T_world_indices, float* point,
                            unsigned int point_num_alloc, SharedIndex* point_indices,
                            float* calibration, unsigned int calibration_num_alloc,
                            SharedIndex* calibration_indices, float* pixel,
                            unsigned int pixel_num_alloc, float* const out_rTr,
                            size_t problem_size);

}  // namespace caspar