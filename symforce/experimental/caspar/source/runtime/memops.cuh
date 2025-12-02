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
  const float4 tmp = *reinterpret_cast<const float4*>(&input[offset + idx * 4]);
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
  float4 tmp;
  tmp.x = x;
  tmp.y = y;
  tmp.z = z;
  *reinterpret_cast<float4*>(&output[offset + idx * 4]) = tmp;
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
  const float4 existing = *reinterpret_cast<const float4*>(&output[offset + idx * 4]);
  float4 tmp;
  tmp.x = x + existing.x;
  tmp.y = y + existing.y;
  tmp.z = z + existing.z;
  *reinterpret_cast<float4*>(&output[offset + idx * 4]) = tmp;
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
__forceinline__ __device__ void flush_sum_shared(float* const output, const uint offset,
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
__forceinline__ __device__ void sum_store(float* const shared_tmp, float* const inout_shared,
                                          const uint offset, const bool valid, float data) {
  auto group = cg::tiled_partition<32>(cg::this_thread_block());

  __syncthreads();
  float tot = cg::reduce(group, valid ? data : 0.0f, cg::plus<float>());
  if (group.thread_rank() == 0) {
    inout_shared[group.meta_group_rank()] = tot;
  }
  __syncthreads();
  if (group.meta_group_rank() == 0) {
    tot = cg::reduce(group, inout_shared[group.thread_rank()], cg::plus<float>());
    if (group.thread_rank() == 0) {
      shared_tmp[offset] = tot;
    }
  }
}

__forceinline__ __device__ void sum_flush_final(const float* const shared_tmp, float* const output,
                                                const uint dim) {
  __syncthreads();
  if (threadIdx.x < dim) {
    atomicAdd(&output[threadIdx.x], shared_tmp[threadIdx.x]);
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

// READ OVERLAPPED
__forceinline__ __device__ void read_and_shuffle_1(float* const inout_shared,
                                                   const float* const input, const uint offset,
                                                   const uint idx, bool is_last, bool should_read,
                                                   float& x0, float& x1) {
  if (should_read) {
    x0 = input[offset + idx];
    inout_shared[threadIdx.x] = x0;
  }
  __syncthreads();
  if (should_read && !is_last) {
    x1 = inout_shared[threadIdx.x + 1];
  }
  __syncthreads();
  if (should_read && is_last) {
    x1 = input[offset + idx + 1];
  }
}
__forceinline__ __device__ void read_and_shuffle_2(float* const inout_shared,
                                                   const float* const input, const uint offset,
                                                   const uint idx, bool is_last, bool should_read,
                                                   float& x0, float& y0, float& x1, float& y1) {
  float2 tmp;
  if (should_read) {
    tmp = reinterpret_cast<const float2*>(&input[offset])[idx];
    x0 = tmp.x;
    y0 = tmp.y;
    reinterpret_cast<float2*>(inout_shared)[threadIdx.x] = tmp;
  }
  __syncthreads();
  if (should_read && !is_last) {
    tmp = reinterpret_cast<float2*>(inout_shared)[threadIdx.x + 1];
  }
  __syncthreads();
  if (should_read && is_last) {
    tmp = reinterpret_cast<const float2*>(&input[offset])[idx + 1];
  }
  x1 = tmp.x;
  y1 = tmp.y;
}

__forceinline__ __device__ void read_and_shuffle_3(float* const inout_shared,
                                                   const float* const input, const uint offset,
                                                   const uint idx, bool is_last, bool should_read,
                                                   float& x0, float& y0, float& z0, float& x1,
                                                   float& y1, float& z1) {
  float4 tmp;
  if (should_read) {
    tmp = reinterpret_cast<const float4*>(&input[offset])[idx];
    x0 = tmp.x;
    y0 = tmp.y;
    z0 = tmp.z;
    reinterpret_cast<float4*>(inout_shared)[threadIdx.x] = tmp;
  }
  __syncthreads();
  if (should_read && !is_last) {
    tmp = reinterpret_cast<float4*>(inout_shared)[threadIdx.x + 1];
  }
  __syncthreads();
  if (should_read && is_last) {
    tmp = reinterpret_cast<const float4*>(&input[offset])[idx + 1];
  }
  x1 = tmp.x;
  y1 = tmp.y;
  z1 = tmp.z;
}

__forceinline__ __device__ void read_and_shuffle_4(float* const inout_shared,
                                                   const float* const input, const uint offset,
                                                   const uint idx, bool is_last, bool valid,
                                                   float& x0, float& y0, float& z0, float& w0,
                                                   float& x1, float& y1, float& z1, float& w1) {
  float4 tmp;
  if (valid) {
    tmp = reinterpret_cast<const float4*>(&input[offset])[idx];
    x0 = tmp.x;
    y0 = tmp.y;
    z0 = tmp.z;
    w0 = tmp.w;
    reinterpret_cast<float4*>(inout_shared)[threadIdx.x] = tmp;
  }
  __syncthreads();
  if (valid && !is_last) {
    tmp = reinterpret_cast<float4*>(inout_shared)[threadIdx.x + 1];
  }
  __syncthreads();
  if (valid && is_last) {
    tmp = reinterpret_cast<const float4*>(&input[offset])[idx + 1];
  }
  x1 = tmp.x;
  y1 = tmp.y;
  z1 = tmp.z;
  w1 = tmp.w;
}

__forceinline__ __device__ void shuffle_and_add_1(float* const inout_shared, float* const output,
                                                  const uint offset, const uint gtidx,
                                                  uint problem_size, const float x0,
                                                  const float x1) {
  if (gtidx < problem_size) {
    inout_shared[threadIdx.x + 1] = x1;
  }
  if (threadIdx.x == 0) {
    inout_shared[0] = x0;
  }
  __syncthreads();
  float from_prev;
  if (gtidx <= problem_size) {
    from_prev = inout_shared[threadIdx.x];
  }
  if (threadIdx.x == 0) {
    from_prev = 0.0f;
  }

  const uint block_start = gtidx & (-1u << 10);
  {
    bool help_intrablock_down = block_start != 0 && threadIdx.x < 1;
    bool help_intrablock_up =
        block_start + 1024 <= problem_size && threadIdx.x >= 1 && threadIdx.x < 2;

    float data;
    if (help_intrablock_down) {
      data = inout_shared[threadIdx.x * 1025];
    }
    if (help_intrablock_up) {
      data = inout_shared[(threadIdx.x) * 1025 + 1024 - 1025];
    }
    __syncthreads();
    if (help_intrablock_down || help_intrablock_up) {
      if (block_start != 0 && block_start + 1024 <= problem_size) {
        uint idx = threadIdx.x < 1 ? threadIdx.x : threadIdx.x - 1u + 1024u;
        atomicAdd(&output[offset + block_start + idx], data);
      } else if (block_start == 0) {
        atomicAdd(&output[offset + block_start + threadIdx.x - 1u + 1024u], data);
      } else {
        atomicAdd(&output[offset + block_start + threadIdx.x], data);
      }
    }
  }
  if ((threadIdx.x > 0 || block_start == 0) && gtidx <= problem_size) {
    float tmp = output[offset + gtidx];
    tmp += from_prev;
    if (gtidx < problem_size) {
      tmp += x0;
    }
    output[offset + gtidx] = tmp;
  }
}
__forceinline__ __device__ void shuffle_and_add_2(float* const inout_shared, float* const output,
                                                  const uint offset, const uint gtidx,
                                                  uint problem_size, const float x0, const float y0,
                                                  const float x1, const float y1) {
  if (gtidx < problem_size) {
    inout_shared[threadIdx.x + 1] = x1;
    inout_shared[threadIdx.x + 1 + 1025] = y1;
  }
  if (threadIdx.x == 0) {
    inout_shared[0] = x0;
    inout_shared[1025] = y0;
  }
  __syncthreads();
  float2 from_prev;
  if (gtidx <= problem_size) {
    from_prev.x = inout_shared[threadIdx.x];
    from_prev.y = inout_shared[threadIdx.x + 1025];
  }
  if (threadIdx.x == 0) {
    from_prev.x = 0.0f;
    from_prev.y = 0.0f;
  }

  const uint block_start = gtidx & (-1u << 10);
  {
    bool help_intrablock_down = block_start != 0 && threadIdx.x < 2;
    bool help_intrablock_up =
        block_start + 1024 <= problem_size && threadIdx.x >= 2 && threadIdx.x < 4;

    float data;
    if (help_intrablock_down) {
      data = inout_shared[threadIdx.x * 1025];
    }
    if (help_intrablock_up) {
      data = inout_shared[(threadIdx.x) * 1025 + 1024 - 2 * 1025];
    }
    __syncthreads();
    if (help_intrablock_down || help_intrablock_up) {
      if (block_start != 0 && block_start + 1024 <= problem_size) {
        uint idx = threadIdx.x < 2 ? threadIdx.x : threadIdx.x - 2u + 2u * 1024u;
        atomicAdd(&output[offset + 2 * block_start + idx], data);
      } else if (block_start == 0) {
        atomicAdd(&output[offset + 2 * block_start + threadIdx.x - 2u + 2u * 1024u], data);
      } else {
        atomicAdd(&output[offset + 2 * block_start + threadIdx.x], data);
      }
    }
  }
  if ((threadIdx.x > 0 || block_start == 0) && gtidx <= problem_size) {
    float2 tmp2 = reinterpret_cast<float2*>(&output[offset])[gtidx];
    tmp2.x += from_prev.x;
    tmp2.y += from_prev.y;
    if (gtidx < problem_size) {
      tmp2.x += x0;
      tmp2.y += y0;
    }
    reinterpret_cast<float2*>(&output[offset])[gtidx] = tmp2;
  }
}
__forceinline__ __device__ void shuffle_and_add_3(float* const inout_shared, float* const output,
                                                  const uint offset, const uint gtidx,
                                                  uint problem_size, const float x0, const float y0,
                                                  const float z0, const float x1, const float y1,
                                                  const float z1) {
  if (gtidx < problem_size) {
    inout_shared[threadIdx.x + 1] = x1;
    inout_shared[threadIdx.x + 1 + 1025] = y1;
    inout_shared[threadIdx.x + 1 + 2 * 1025] = z1;
  }
  if (threadIdx.x == 0) {
    inout_shared[0] = x0;
    inout_shared[1025] = y0;
    inout_shared[2 * 1025] = z0;
  }
  __syncthreads();
  float from_prev_x, from_prev_y, from_prev_z;
  if (gtidx <= problem_size) {
    from_prev_x = inout_shared[threadIdx.x];
    from_prev_y = inout_shared[threadIdx.x + 1025];
    from_prev_z = inout_shared[threadIdx.x + 2 * 1025];
  }
  if (threadIdx.x == 0) {
    from_prev_x = 0.0f;
    from_prev_y = 0.0f;
    from_prev_z = 0.0f;
  }

  const uint block_start = gtidx & (-1u << 10);
  {
    bool help_intrablock_down = block_start != 0 && threadIdx.x < 3;
    bool help_intrablock_up =
        block_start + 1024 <= problem_size && threadIdx.x >= 3 && threadIdx.x < 6;

    float data;
    if (help_intrablock_down) {
      data = inout_shared[threadIdx.x * 1025];
    }
    if (help_intrablock_up) {
      data = inout_shared[(threadIdx.x) * 1025 + 1024 - 3 * 1025];
    }
    __syncthreads();
    if (help_intrablock_down || help_intrablock_up) {
      if (block_start != 0 && block_start + 1024 <= problem_size) {
        uint idx = threadIdx.x < 3 ? threadIdx.x : threadIdx.x - 3u + 4u * 1024u;
        atomicAdd(&output[offset + 4 * block_start + idx], data);
      } else if (block_start == 0) {
        atomicAdd(&output[offset + 4 * block_start + threadIdx.x - 3u + 4u * 1024u], data);
      } else {
        atomicAdd(&output[offset + 4 * block_start + threadIdx.x], data);
      }
    }
  }
  if ((threadIdx.x > 0 || block_start == 0) && gtidx <= problem_size) {
    float4 tmp4 = reinterpret_cast<float4*>(&output[offset])[gtidx];
    tmp4.x += from_prev_x;
    tmp4.y += from_prev_y;
    tmp4.z += from_prev_z;
    if (gtidx < problem_size) {
      tmp4.x += x0;
      tmp4.y += y0;
      tmp4.z += z0;
    }
    reinterpret_cast<float4*>(&output[offset])[gtidx] = tmp4;
  }
}

__forceinline__ __device__ void shuffle_and_add_4(float* const inout_shared, float* const output,
                                                  const uint offset, const uint gtidx,
                                                  uint problem_size, const float x0, const float y0,
                                                  const float z0, const float w0, const float x1,
                                                  const float y1, const float z1, const float w1) {
  if (gtidx < problem_size) {
    inout_shared[threadIdx.x + 1] = x1;
    inout_shared[threadIdx.x + 1 + 1025] = y1;
    inout_shared[threadIdx.x + 1 + 2 * 1025] = z1;
    inout_shared[threadIdx.x + 1 + 3 * 1025] = w1;
  }
  if (threadIdx.x == 0) {
    inout_shared[0] = x0;
    inout_shared[1025] = y0;
    inout_shared[2 * 1025] = z0;
    inout_shared[3 * 1025] = w0;
  }
  __syncthreads();
  float4 from_prev;
  if (gtidx <= problem_size) {
    from_prev.x = inout_shared[threadIdx.x];
    from_prev.y = inout_shared[threadIdx.x + 1025];
    from_prev.z = inout_shared[threadIdx.x + 2 * 1025];
    from_prev.w = inout_shared[threadIdx.x + 3 * 1025];
  }
  if (threadIdx.x == 0) {
    from_prev.x = 0.0f;
    from_prev.y = 0.0f;
    from_prev.z = 0.0f;
    from_prev.w = 0.0f;
  }

  const uint block_start = gtidx & (-1u << 10);
  {
    bool help_intrablock_down = block_start != 0 && threadIdx.x < 4;
    bool help_intrablock_up =
        block_start + 1024 <= problem_size && threadIdx.x >= 4 && threadIdx.x < 8;

    float data;
    if (help_intrablock_down) {
      data = inout_shared[threadIdx.x * 1025];
    }
    if (help_intrablock_up) {
      data = inout_shared[(threadIdx.x) * 1025 + 1024 - 4 * 1025];
    }
    __syncthreads();
    if (help_intrablock_down || help_intrablock_up) {
      if (block_start != 0 && block_start + 1024 <= problem_size) {
        uint idx = threadIdx.x < 4 ? threadIdx.x : threadIdx.x - 4u + 4u * 1024u;
        atomicAdd(&output[offset + 4 * block_start + idx], data);
      } else if (block_start == 0) {
        atomicAdd(&output[offset + 4 * block_start + threadIdx.x - 4u + 4u * 1024u], data);
      } else {
        atomicAdd(&output[offset + 4 * block_start + threadIdx.x], data);
      }
    }
  }
  if ((threadIdx.x > 0 || block_start == 0) && gtidx <= problem_size) {
    float4 tmp4 = reinterpret_cast<float4*>(&output[offset])[gtidx];
    tmp4.x += from_prev.x;
    tmp4.y += from_prev.y;
    tmp4.z += from_prev.z;
    tmp4.w += from_prev.w;
    if (gtidx < problem_size) {
      tmp4.x += x0;
      tmp4.y += y0;
      tmp4.z += z0;
      tmp4.w += w0;
    }
    reinterpret_cast<float4*>(&output[offset])[gtidx] = tmp4;
  }
}

// SHUFFLE AND WRITE
__forceinline__ __device__ void shuffle_and_write_1(float* const inout_shared, float* const output,
                                                    const uint offset, const uint gtidx,
                                                    uint problem_size, const float x0,
                                                    const float x1) {
  if (gtidx < problem_size) {
    inout_shared[threadIdx.x + 1] = x1;
  }
  if (threadIdx.x == 0) {
    inout_shared[0] = x0;
  }
  __syncthreads();
  float from_prev;
  if (gtidx <= problem_size) {
    from_prev = inout_shared[threadIdx.x];
  }
  if (threadIdx.x == 0) {
    from_prev = 0.0f;
  }

  const uint block_start = gtidx & (-1u << 10);
  {
    bool help_intrablock_down = block_start != 0 && threadIdx.x < 1;
    bool help_intrablock_up =
        block_start + 1024 <= problem_size && threadIdx.x >= 1 && threadIdx.x < 2;

    float data;
    if (help_intrablock_down) {
      data = inout_shared[threadIdx.x * 1025];
    }
    if (help_intrablock_up) {
      data = inout_shared[(threadIdx.x) * 1025 + 1024 - 1025];
    }
    __syncthreads();
    if (help_intrablock_down || help_intrablock_up) {
      if (block_start != 0 && block_start + 1024 <= problem_size) {
        uint idx = threadIdx.x < 1 ? threadIdx.x : threadIdx.x - 1u + 1024u;
        atomicAdd(&output[offset + block_start + idx], data);
      } else if (block_start == 0) {
        atomicAdd(&output[offset + block_start + threadIdx.x - 1u + 1024u], data);
      } else {
        atomicAdd(&output[offset + block_start + threadIdx.x], data);
      }
    }
  }
  if ((threadIdx.x > 0 || block_start == 0) && gtidx <= problem_size) {
    float tmp;
    if (gtidx < problem_size) {
      tmp = x0 + from_prev;
    } else {
      tmp = from_prev;
    }
    output[offset + gtidx] = tmp;
  }
}

__forceinline__ __device__ void shuffle_and_write_2(float* const inout_shared, float* const output,
                                                    const uint offset, const uint gtidx,
                                                    uint problem_size, const float x0,
                                                    const float y0, const float x1,
                                                    const float y1) {
  if (gtidx < problem_size) {
    inout_shared[threadIdx.x + 1] = x1;
    inout_shared[threadIdx.x + 1 + 1025] = y1;
  }
  if (threadIdx.x == 0) {
    inout_shared[0] = x0;
    inout_shared[1025] = y0;
  }
  __syncthreads();
  float2 from_prev;
  if (gtidx <= problem_size) {
    from_prev.x = inout_shared[threadIdx.x];
    from_prev.y = inout_shared[threadIdx.x + 1025];
  }
  if (threadIdx.x == 0) {
    from_prev.x = 0.0f;
    from_prev.y = 0.0f;
  }

  const uint block_start = gtidx & (-1u << 10);
  {
    bool help_intrablock_down = block_start != 0 && threadIdx.x < 2;
    bool help_intrablock_up =
        block_start + 1024 <= problem_size && threadIdx.x >= 2 && threadIdx.x < 4;

    float data;
    if (help_intrablock_down) {
      data = inout_shared[threadIdx.x * 1025];
    }
    if (help_intrablock_up) {
      data = inout_shared[(threadIdx.x) * 1025 + 1024 - 2 * 1025];
    }
    __syncthreads();
    if (help_intrablock_down || help_intrablock_up) {
      if (block_start != 0 && block_start + 1024 <= problem_size) {
        uint idx = threadIdx.x < 2 ? threadIdx.x : threadIdx.x - 2u + 2u * 1024u;
        atomicAdd(&output[offset + 2 * block_start + idx], data);
      } else if (block_start == 0) {
        atomicAdd(&output[offset + 2 * block_start + threadIdx.x - 2u + 2u * 1024u], data);
      } else {
        atomicAdd(&output[offset + 2 * block_start + threadIdx.x], data);
      }
    }
  }
  if ((threadIdx.x > 0 || block_start == 0) && gtidx <= problem_size) {
    float2 tmp2;
    if (gtidx < problem_size) {
      tmp2.x = x0 + from_prev.x;
      tmp2.y = y0 + from_prev.y;
    } else {
      tmp2.x = from_prev.x;
      tmp2.y = from_prev.y;
    }
    reinterpret_cast<float2*>(&output[offset])[gtidx] = tmp2;
  }
}
__forceinline__ __device__ void shuffle_and_write_3(float* const inout_shared, float* const output,
                                                    const uint offset, const uint gtidx,
                                                    uint problem_size, const float x0,
                                                    const float y0, const float z0, const float x1,
                                                    const float y1, const float z1) {
  if (gtidx < problem_size) {
    inout_shared[threadIdx.x + 1] = x1;
    inout_shared[threadIdx.x + 1 + 1025] = y1;
    inout_shared[threadIdx.x + 1 + 2 * 1025] = z1;
  }
  if (threadIdx.x == 0) {
    inout_shared[0] = x0;
    inout_shared[1025] = y0;
    inout_shared[2 * 1025] = z0;
  }
  __syncthreads();
  float from_prev_x, from_prev_y, from_prev_z;
  if (gtidx <= problem_size) {
    from_prev_x = inout_shared[threadIdx.x];
    from_prev_y = inout_shared[threadIdx.x + 1025];
    from_prev_z = inout_shared[threadIdx.x + 2 * 1025];
  }
  if (threadIdx.x == 0) {
    from_prev_x = 0.0f;
    from_prev_y = 0.0f;
    from_prev_z = 0.0f;
  }

  const uint block_start = gtidx & (-1u << 10);
  {
    bool help_intrablock_down = block_start != 0 && threadIdx.x < 3;
    bool help_intrablock_up =
        block_start + 1024 <= problem_size && threadIdx.x >= 3 && threadIdx.x < 6;

    float data;
    if (help_intrablock_down) {
      data = inout_shared[threadIdx.x * 1025];
    }
    if (help_intrablock_up) {
      data = inout_shared[(threadIdx.x) * 1025 + 1024 - 3 * 1025];
    }
    __syncthreads();
    if (help_intrablock_down || help_intrablock_up) {
      if (block_start != 0 && block_start + 1024 <= problem_size) {
        uint idx = threadIdx.x < 3 ? threadIdx.x : threadIdx.x - 3u + 4u * 1024u;
        atomicAdd(&output[offset + 4 * block_start + idx], data);
      } else if (block_start == 0) {
        atomicAdd(&output[offset + 4 * block_start + threadIdx.x - 3u + 4u * 1024u], data);
      } else {
        atomicAdd(&output[offset + 4 * block_start + threadIdx.x], data);
      }
    }
  }
  if ((threadIdx.x > 0 || block_start == 0) && gtidx <= problem_size) {
    float4 tmp4;
    if (gtidx < problem_size) {
      tmp4.x = x0 + from_prev_x;
      tmp4.y = y0 + from_prev_y;
      tmp4.z = z0 + from_prev_z;
    } else {
      tmp4.x = from_prev_x;
      tmp4.y = from_prev_y;
      tmp4.z = from_prev_z;
    }
    reinterpret_cast<float4*>(&output[offset])[gtidx] = tmp4;
  }
}
__forceinline__ __device__ void shuffle_and_write_4(float* const inout_shared, float* const output,
                                                    const uint offset, const uint gtidx,
                                                    uint problem_size, const float x0,
                                                    const float y0, const float z0, const float w0,
                                                    const float x1, const float y1, const float z1,
                                                    const float w1) {
  if (gtidx < problem_size) {
    inout_shared[threadIdx.x + 1] = x1;
    inout_shared[threadIdx.x + 1 + 1025] = y1;
    inout_shared[threadIdx.x + 1 + 2 * 1025] = z1;
    inout_shared[threadIdx.x + 1 + 3 * 1025] = w1;
  }
  if (threadIdx.x == 0) {
    inout_shared[0] = x0;
    inout_shared[1025] = y0;
    inout_shared[2 * 1025] = z0;
    inout_shared[3 * 1025] = w0;
  }
  __syncthreads();
  float4 from_prev;
  if (gtidx <= problem_size) {
    from_prev.x = inout_shared[threadIdx.x];
    from_prev.y = inout_shared[threadIdx.x + 1025];
    from_prev.z = inout_shared[threadIdx.x + 2 * 1025];
    from_prev.w = inout_shared[threadIdx.x + 3 * 1025];
  }
  if (threadIdx.x == 0) {
    from_prev.x = 0.0f;
    from_prev.y = 0.0f;
    from_prev.z = 0.0f;
    from_prev.w = 0.0f;
  }

  const uint block_start = gtidx & (-1u << 10);
  {
    bool help_intrablock_down = block_start != 0 && threadIdx.x < 4;
    bool help_intrablock_up =
        block_start + 1024 <= problem_size && threadIdx.x >= 4 && threadIdx.x < 8;

    float data;
    if (help_intrablock_down) {
      data = inout_shared[threadIdx.x * 1025];
    }
    if (help_intrablock_up) {
      data = inout_shared[(threadIdx.x) * 1025 + 1024 - 4 * 1025];
    }
    __syncthreads();
    if (help_intrablock_down || help_intrablock_up) {
      if (block_start != 0 && block_start + 1024 <= problem_size) {
        uint idx = threadIdx.x < 4 ? threadIdx.x : threadIdx.x - 4u + 4u * 1024u;
        atomicAdd(&output[offset + 4 * block_start + idx], data);
      } else if (block_start == 0) {
        atomicAdd(&output[offset + 4 * block_start + threadIdx.x - 4u + 4u * 1024u], data);
      } else {
        atomicAdd(&output[offset + 4 * block_start + threadIdx.x], data);
      }
    }
  }
  if ((threadIdx.x > 0 || block_start == 0) && gtidx <= problem_size) {
    float4 tmp4;
    if (gtidx < problem_size) {
      tmp4.x = x0 + from_prev.x;
      tmp4.y = y0 + from_prev.y;
      tmp4.z = z0 + from_prev.z;
      tmp4.w = w0 + from_prev.w;
    } else {
      tmp4.x = from_prev.x;
      tmp4.y = from_prev.y;
      tmp4.z = from_prev.z;
      tmp4.w = from_prev.w;
    }
    reinterpret_cast<float4*>(&output[offset])[gtidx] = tmp4;
  }
}

}  // namespace caspar
