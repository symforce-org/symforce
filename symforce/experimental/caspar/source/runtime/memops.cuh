/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2025, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#pragma once

#include <stdio.h>

#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include "shared_indices.h"

namespace cg = cooperative_groups;

// READ INDEXED
namespace caspar {

__forceinline__ __device__ void read_idx_1(const float* const input, const uint offset,
                                           const uint idx, float& x) {
  x = input[offset + idx];
}

__forceinline__ __device__ void read_idx_2(const float* const input, const uint offset,
                                           const uint idx, float& x, float& y) {
  const float2 tmp = *reinterpret_cast<const float2*>(&input[offset + idx * 2]);
  x = tmp.x;
  y = tmp.y;
}

__forceinline__ __device__ void read_idx_3(const float* const input, const uint offset,
                                           const uint idx, float& x, float& y, float& z) {
  const float3 tmp = *reinterpret_cast<const float3*>(&input[offset + idx * 4]);
  x = tmp.x;
  y = tmp.y;
  z = tmp.z;
}

__forceinline__ __device__ void read_idx_4(const float* const input, const uint offset,
                                           const uint idx, float& x, float& y, float& z, float& w) {
  const float4 tmp = *reinterpret_cast<const float4*>(&input[offset + idx * 4]);
  x = tmp.x;
  y = tmp.y;
  z = tmp.z;
  w = tmp.w;
}

// WRITE INDEXED

__forceinline__ __device__ void write_idx_1(float* const output, const uint offset, const uint idx,
                                            const float x) {
  output[offset + idx] = x;
}

__forceinline__ __device__ void write_idx_2(float* const output, const uint offset, const uint idx,
                                            const float x, const float y) {
  float2 tmp;
  tmp.x = x;
  tmp.y = y;
  *reinterpret_cast<float2*>(&output[offset + idx * 2]) = tmp;
}

__forceinline__ __device__ void write_idx_3(float* const output, const uint offset, const uint idx,
                                            const float x, const float y, const float z) {
  float3 tmp;
  tmp.x = x;
  tmp.y = y;
  tmp.z = z;
  *reinterpret_cast<float3*>(&output[offset + idx * 4]) = tmp;
}

__forceinline__ __device__ void write_idx_4(float* const output, const uint offset, const uint idx,
                                            const float x, const float y, const float z,
                                            const float w) {
  float4 tmp;
  tmp.x = x;
  tmp.y = y;
  tmp.z = z;
  tmp.w = w;
  *reinterpret_cast<float4*>(&output[offset + idx * 4]) = tmp;
}

// WRITE ADD
__forceinline__ __device__ void add_idx_1(float* const output, const uint offset, const uint idx,
                                          const float x) {
  output[offset + idx] += x;
}

__forceinline__ __device__ void add_idx_2(float* const output, const uint offset, const uint idx,
                                          const float x, const float y) {
  const float2 existing = *reinterpret_cast<const float2*>(&output[offset + idx * 2]);
  float2 tmp;
  tmp.x = x + existing.x;
  tmp.y = y + existing.y;
  *reinterpret_cast<float2*>(&output[offset + idx * 2]) = tmp;
}

__forceinline__ __device__ void add_idx_3(float* const output, const uint offset, const uint idx,
                                          const float x, const float y, const float z) {
  const float3 existing = *reinterpret_cast<const float3*>(&output[offset + idx * 4]);
  float3 tmp;
  tmp.x = x + existing.x;
  tmp.y = y + existing.y;
  tmp.z = z + existing.z;
  *reinterpret_cast<float3*>(&output[offset + idx * 4]) = tmp;
}

__forceinline__ __device__ void add_idx_4(float* const output, const uint offset, const uint idx,
                                          const float x, const float y, const float z,
                                          const float w) {
  const float4 existing = *reinterpret_cast<const float4*>(&output[offset + idx * 4]);
  float4 tmp;
  tmp.x = x + existing.x;
  tmp.y = y + existing.y;
  tmp.z = z + existing.z;
  tmp.w = w + existing.w;
  *reinterpret_cast<float4*>(&output[offset + idx * 4]) = tmp;
}

// WRITE SUM
/**
 * The write_sum_x writes data to shared local data.
 * flush_sum or flush_sum_block should be called after all write_sum_x calls to perform reduction
 * and write to the output.
 */
__forceinline__ __device__ void write_sum_1(float* const inout_shared, const float x) {
  inout_shared[threadIdx.x] = x;
}

__forceinline__ __device__ void write_sum_2(float* const inout_shared, const float x,
                                            const float y) {
  inout_shared[threadIdx.x * 2 + 0] = x;
  inout_shared[threadIdx.x * 2 + 1] = y;
}

__forceinline__ __device__ void write_sum_3(float* const inout_shared, const float x, const float y,
                                            const float z) {
  inout_shared[threadIdx.x * 3 + 0] = x;
  inout_shared[threadIdx.x * 3 + 1] = y;
  inout_shared[threadIdx.x * 3 + 2] = z;
}

__forceinline__ __device__ void write_sum_4(float* const inout_shared, const float x, const float y,
                                            const float z, const float w) {
  inout_shared[threadIdx.x * 4 + 0] = x;
  inout_shared[threadIdx.x * 4 + 1] = y;
  inout_shared[threadIdx.x * 4 + 2] = z;
  inout_shared[threadIdx.x * 4 + 3] = w;
}

/**
 * Function used to perform collaborative reductions. Read more on caspar.argtypes.accessors.Sum.
 */
template <uint dim_target>
__forceinline__ __device__ void flush_sum(float* const output, const uint offset,
                                          const SharedIndex* const indices,
                                          float* const inout_shared) {
  __syncthreads();

  const SharedIndex idx = indices[threadIdx.x];
  uint unique = 0xffffffff;
  if (idx.argsort != 0xffff) {  // 0xffff indicates the thread is not used.
    unique = indices[indices[idx.argsort].target].unique;
  }
  const cg::coalesced_group group = cg::labeled_partition(cg::coalesced_threads(), unique);

#pragma unroll
  for (int i = 0; i < dim_target; i++) {
    const SharedIndex idx = indices[threadIdx.x];
    float tot;
    if (idx.argsort != 0xffff) {  // 0xffff indicates the thread is not used.
      tot = cg::reduce(group, inout_shared[idx.argsort * dim_target + i], cg::plus<float>());
    }
    __syncthreads();
    inout_shared[threadIdx.x * dim_target + i] = 0.0f;
    __syncthreads();

    // 0xffff indicates the thread is not used.
    if (idx.argsort != 0xffff && group.thread_rank() == 0) {
      atomicAdd_block(&inout_shared[indices[idx.argsort].target * dim_target + i], tot);
    }
    __syncthreads();
  }

  constexpr uint dim_aligned = dim_target == 3 ? 4 : dim_target;
  for (int i = 0; i < dim_target; i++) {
    const uint val_idx = threadIdx.x + blockDim.x * i;
    const uint obj_idx = indices[val_idx / dim_target].unique;
    const uint elem = val_idx % dim_target;
    // 0xffffffff indicates this thread has nothing to write.
    if (obj_idx == 0xffffffff) {
      break;
    }
    atomicAdd(&output[offset + obj_idx * dim_aligned + elem], inout_shared[val_idx]);
  }
  __syncthreads();
}

/**
 * Function used to perform a reduction over the block.
 *
 * Read more on caspar.argtypes.accessors.BlockSum.
 */
template <uint dim_target>
__forceinline__ __device__ void flush_sum_block(float* const output, float* const inout_shared,
                                                const bool valid) {
  __syncthreads();
  const cg::coalesced_group group = cg::binary_partition(cg::coalesced_threads(), valid);
  constexpr uint dim_aligned = dim_target == 3 ? 4 : dim_target;

#pragma unroll
  for (int i = 0; i < dim_target; i++) {
    float tot;

    if (valid) {
      tot = cg::reduce(group, inout_shared[threadIdx.x * dim_target + i], cg::plus<float>());
    }
    __syncthreads();
    inout_shared[threadIdx.x * dim_target + i] = 0.0f;
    __syncthreads();

    if (valid && group.thread_rank() == 0) {
      // TODO(Emil Martes): use first warp to reduce instead of atomicAdd_block
      atomicAdd_block(&inout_shared[i], tot);
    }
    __syncthreads();
  }
  for (int i = threadIdx.x; i < dim_target; i += blockDim.x) {
    output[blockIdx.x * dim_aligned + i] = inout_shared[i];
  }
}

/**
 * Function used to perform a reduction over a block and add to the output.
 *
 * Read more on caspar.argtypes.accessors.BlockSumAdd.
 */
template <uint dim_target>
__forceinline__ __device__ void flush_sum_block_add(float* const output, float* const inout_shared,
                                                    const bool valid) {
  __syncthreads();
  const cg::coalesced_group group = cg::binary_partition(cg::coalesced_threads(), valid);
  constexpr uint dim_aligned = dim_target == 3 ? 4 : dim_target;

#pragma unroll
  for (int i = 0; i < dim_target; i++) {
    float tot;

    if (valid) {
      tot = cg::reduce(group, inout_shared[threadIdx.x * dim_target + i], cg::plus<float>());
    }
    __syncthreads();
    inout_shared[threadIdx.x * dim_target + i] = 0.0f;
    __syncthreads();

    if (valid && group.thread_rank() == 0) {
      // TODO(Emil Martes): use first warp to reduce instead of atomicAdd_block
      atomicAdd_block(&inout_shared[i], tot);
    }
    __syncthreads();
  }
  for (int i = threadIdx.x; i < dim_target; i += blockDim.x) {
    output[blockIdx.x * dim_aligned + i] += inout_shared[i];
  }
}

// READ SHARED

/**
 * Function used to load data from global memory into shared memory.
 *
 * Read more on caspar.argtypes.accessors.Shared.
 * Should be followed by read_shared_x.
 */
template <uint dim_target>
__forceinline__ __device__ void load_shared(const float* const input, const uint offset,
                                            const SharedIndex* const indices,
                                            float* const inout_shared) {
  __syncthreads();
  constexpr uint dim_aligned = dim_target == 3 ? 4 : dim_target;

#pragma unroll
  for (int i = 0; i < dim_target; i++) {
    const uint val_idx = blockDim.x * i + threadIdx.x;
    const uint obj_idx = indices[val_idx / dim_target].unique;
    const uint elem = val_idx % dim_target;

    // 0xffffffff indicates this thread has nothing to read.
    if (obj_idx == 0xffffffff) {
      break;
    }
    inout_shared[val_idx] = input[offset + obj_idx * dim_aligned + elem];
  }
  __syncthreads();
}

/**
 * Function used to load data from a unique global element into shared memory.
 *
 * Read more on caspar.argtypes.accessors.Unique.
 * Should be followed by read_shared_x.
 */
template <uint dim_target>
__forceinline__ __device__ void load_unique(const float* const input, const uint offset,
                                            float* const inout_shared) {
  __syncthreads();
  if (threadIdx.x < dim_target) {
    inout_shared[threadIdx.x] = input[offset + threadIdx.x];
  }
  __syncthreads();
}

// READ SHARED
__forceinline__ __device__ void read_shared_1(const float* const inout_shared, const uint target,
                                              float& x) {
  x = inout_shared[target];
}

__forceinline__ __device__ void read_shared_2(const float* const inout_shared, const uint target,
                                              float& x, float& y) {
  x = inout_shared[target * 2 + 0];
  y = inout_shared[target * 2 + 1];
}

__forceinline__ __device__ void read_shared_3(const float* const inout_shared, const uint target,
                                              float& x, float& y, float& z) {
  x = inout_shared[target * 3 + 0];
  y = inout_shared[target * 3 + 1];
  z = inout_shared[target * 3 + 2];
}

__forceinline__ __device__ void read_shared_4(const float* const inout_shared, const uint target,
                                              float& x, float& y, float& z, float& w) {
  x = inout_shared[target * 4 + 0];
  y = inout_shared[target * 4 + 1];
  z = inout_shared[target * 4 + 2];
  w = inout_shared[target * 4 + 3];
}

}  // namespace caspar
