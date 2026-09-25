#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include "kernel_caspar_padding_lane_score.h"
#include "memops.cuh"

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1) CasparPaddingLaneScoreKernel(
    float* cam_T_world, unsigned int cam_T_world_num_alloc, SharedIndex* cam_T_world_indices,
    float* point, unsigned int point_num_alloc, SharedIndex* point_indices, float* calibration,
    unsigned int calibration_num_alloc, SharedIndex* calibration_indices, float* pixel,
    unsigned int pixel_num_alloc, float* const out_rTr, size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex cam_T_world_indices_loc[1024];
  cam_T_world_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size ? cam_T_world_indices[global_thread_idx]
                                        : SharedIndex{0xffffffff, 0xffff, 0xffff});
  __shared__ SharedIndex point_indices_loc[1024];
  point_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size ? point_indices[global_thread_idx]
                                        : SharedIndex{0xffffffff, 0xffff, 0xffff});
  __shared__ SharedIndex calibration_indices_loc[1024];
  calibration_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size ? calibration_indices[global_thread_idx]
                                        : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ float out_rTr_local[1];

  float r0 = 0, r1 = 0, r2 = 0, r3 = 0, r4 = 0, r5 = 0, r6 = 0, r7 = 0, r8 = 0, r9 = 0, r10 = 0,
        r11 = 0, r12 = 0, r13 = 0, r14 = 0, r15 = 0, r16 = 0, r17 = 0, r18 = 0, r19 = 0, r20 = 0,
        r21 = 0, r22 = 0, r23 = 0, r24 = 0, r25 = 0, r26 = 0, r27 = 0, r28 = 0;
  LoadShared<4, float, float>(calibration, 0 * calibration_num_alloc, calibration_indices_loc,
                              (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float*)inout_shared, calibration_indices_loc[threadIdx.x].target, r0, r1,
                       r2, r3);
  };
  __syncthreads();
  LoadShared<4, float, float>(calibration, 4 * calibration_num_alloc, calibration_indices_loc,
                              (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float*)inout_shared, calibration_indices_loc[threadIdx.x].target, r4, r5,
                       r6, r7);
  };
  __syncthreads();
  LoadShared<3, float, float>(cam_T_world, 4 * cam_T_world_num_alloc, cam_T_world_indices_loc,
                              (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float*)inout_shared, cam_T_world_indices_loc[threadIdx.x].target, r8, r9,
                       r10);
  };
  __syncthreads();
  LoadShared<3, float, float>(point, 0 * point_num_alloc, point_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float*)inout_shared, point_indices_loc[threadIdx.x].target, r11, r12, r13);
  };
  __syncthreads();
  LoadShared<4, float, float>(cam_T_world, 0 * cam_T_world_num_alloc, cam_T_world_indices_loc,
                              (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float*)inout_shared, cam_T_world_indices_loc[threadIdx.x].target, r14, r15,
                       r16, r17);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r18 = r14 * r14;
    r19 = -2.00000000000000000e+00;
    r18 = r18 * r19;
    r20 = 1.00000000000000000e+00;
    r21 = r15 * r15;
    r21 = fmaf(r19, r21, r20);
    r22 = r18 + r21;
    r22 = fmaf(r13, r22, r10);
    r10 = 2.00000000000000000e+00;
    r23 = r14 * r10;
    r24 = r16 * r23;
    r25 = r17 * r19;
    r26 = fmaf(r15, r25, r24);
    r27 = r15 * r16;
    r27 = r27 * r10;
    r28 = fmaf(r17, r23, r27);
    r22 = fmaf(r11, r26, r22);
    r22 = fmaf(r12, r28, r22);
    r28 = r22 * r22;
    r28 = 1.0 / r28;
    r18 = r20 + r18;
    r26 = r16 * r16;
    r26 = r19 * r26;
    r18 = r18 + r26;
    r18 = fmaf(r12, r18, r9);
    r9 = r16 * r17;
    r23 = r15 * r23;
    r9 = fmaf(r10, r9, r23);
    r14 = fmaf(r14, r25, r27);
    r18 = fmaf(r11, r9, r18);
    r18 = fmaf(r13, r14, r18);
    r14 = r18 * r18;
    r14 = r28 * r14;
    r9 = 3.00000000000000000e+00;
    r21 = r26 + r21;
    r21 = fmaf(r11, r21, r8);
    r25 = fmaf(r16, r25, r23);
    r23 = r15 * r17;
    r23 = fmaf(r10, r23, r24);
    r21 = fmaf(r12, r25, r21);
    r21 = fmaf(r13, r23, r21);
    r23 = r21 * r21;
    r23 = r28 * r23;
    r13 = fmaf(r9, r23, r14);
    r25 = r14 + r23;
    r4 = fmaf(r4, r25, r20);
    r25 = r25 * r25;
    r4 = fmaf(r5, r25, r4);
    r22 = 1.0 / r22;
    r22 = r4 * r22;
    r13 = fmaf(r21, r22, r7 * r13);
    r4 = r6 * r10;
    r28 = r18 * r28;
    r4 = r4 * r21;
    r13 = fmaf(r28, r4, r13);
    r13 = fmaf(r0, r13, r2);
    ReadIdx2<1024, float, float, float2>(pixel, 0 * pixel_num_alloc, global_thread_idx, r0, r2);
    r4 = -1.00000000000000000e+00;
    r13 = fmaf(r0, r4, r13);
    r14 = fmaf(r9, r14, r23);
    r22 = fmaf(r18, r22, r6 * r14);
    r18 = r7 * r10;
    r18 = r18 * r21;
    r22 = fmaf(r28, r18, r22);
    r22 = fmaf(r1, r22, r3);
    r22 = fmaf(r2, r4, r22);
    r22 = fmaf(r22, r22, r13 * r13);
  };
  SumStore<float>(out_rTr_local, (float*)inout_shared, 0, global_thread_idx < problem_size, r22);
  SumFlushFinal<float>(out_rTr_local, out_rTr, 1);
}

void CasparPaddingLaneScore(float* cam_T_world, unsigned int cam_T_world_num_alloc,
                            SharedIndex* cam_T_world_indices, float* point,
                            unsigned int point_num_alloc, SharedIndex* point_indices,
                            float* calibration, unsigned int calibration_num_alloc,
                            SharedIndex* calibration_indices, float* pixel,
                            unsigned int pixel_num_alloc, float* const out_rTr,
                            size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  CasparPaddingLaneScoreKernel<<<n_blocks, 1024>>>(
      cam_T_world, cam_T_world_num_alloc, cam_T_world_indices, point, point_num_alloc,
      point_indices, calibration, calibration_num_alloc, calibration_indices, pixel,
      pixel_num_alloc, out_rTr, problem_size);
}

}  // namespace caspar