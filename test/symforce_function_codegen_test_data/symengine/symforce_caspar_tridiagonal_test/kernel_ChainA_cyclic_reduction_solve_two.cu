#include <stdexcept>
#include <string>

#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include "kernel_ChainA_cyclic_reduction_solve_two.h"
#include "memops.cuh"

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(32, 1) ChainACyclicReductionSolveTwoKernel(
    double* const eq, unsigned int eq_num_alloc, int eq_offset, int eq_stride, double* solu2km1,
    unsigned int solu2km1_num_alloc, int solu2km1_offset, int solu2km1_stride, double* solu2k,
    unsigned int solu2k_num_alloc, int solu2k_offset, int solu2k_stride, size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  // Scratch data, then the shared indices of each shared accessor; dynamic, so
  // that it may exceed the 48 KB static shared memory limit (see the launch
  // below).
  extern __shared__ __align__(16) uint8_t inout_shared[];

  double r0 = 0, r1 = 0, r2 = 0, r3 = 0, r4 = 0, r5 = 0, r6 = 0, r7 = 0, r8 = 0, r9 = 0, r10 = 0,
         r11 = 0, r12 = 0, r13 = 0, r14 = 0, r15 = 0, r16 = 0, r17 = 0, r18 = 0, r19 = 0, r20 = 0,
         r21 = 0, r22 = 0, r23 = 0, r24 = 0, r25 = 0, r26 = 0, r27 = 0, r28 = 0, r29 = 0, r30 = 0,
         r31 = 0;

  if (global_thread_idx < problem_size) {
    r0 = -1.00000000000000000e+00;
  };
  ReadAndShuffleWithDefault2<32, double, double, double2>(
      (double*)inout_shared, eq, 8 * eq_num_alloc, global_thread_idx * eq_stride + eq_offset,
      threadIdx.x == 31, global_thread_idx <= problem_size, eq_num_alloc, eq_stride, r1, r2, r3, r4,
      0.000000f, 0.000000f);
  ReadAndShuffleWithDefault2<32, double, double, double2>(
      (double*)inout_shared, eq, 14 * eq_num_alloc, global_thread_idx * eq_stride + eq_offset,
      threadIdx.x == 31, global_thread_idx <= problem_size, eq_num_alloc, eq_stride, r4, r3, r5, r6,
      1.000000f, 0.000000f);
  if (global_thread_idx < problem_size) {
    r4 = 1.0 / r4;
  };
  ReadAndShuffleWithDefault2<32, double, double, double2>(
      (double*)inout_shared, eq, 18 * eq_num_alloc, global_thread_idx * eq_stride + eq_offset,
      threadIdx.x == 31, global_thread_idx <= problem_size, eq_num_alloc, eq_stride, r7, r8, r9,
      r10, 0.000000f, 1.000000f);
  if (global_thread_idx < problem_size) {
    r3 = r3 * r0;
    r3 = r3 * r4;
    r2 = fma(r1, r3, r2);
    r11 = r7 * r2;
    r8 = 1.0 / r8;
  };
  ReadAndShuffleWithDefault2<32, double, double, double2>(
      (double*)inout_shared, eq, 16 * eq_num_alloc, global_thread_idx * eq_stride + eq_offset,
      threadIdx.x == 31, global_thread_idx <= problem_size, eq_num_alloc, eq_stride, r12, r13, r14,
      r15, 1.000000f, 1.000000f);
  if (global_thread_idx < problem_size) {
    r12 = 1.0 / r12;
    r12 = r8 * r12;
    r8 = r0 * r12;
    r11 = fma(r8, r11, r1 * r4);
    r1 = r0 * r11;
    r13 = 1.0 / r13;
    r16 = 1.00000000000000000e+00;
  };
  ReadAndShuffleWithDefault2<32, double, double, double2>(
      (double*)inout_shared, eq, 2 * eq_num_alloc, global_thread_idx * eq_stride + eq_offset,
      threadIdx.x == 31, global_thread_idx <= problem_size, eq_num_alloc, eq_stride, r17, r18, r19,
      r20, 0.000000f, 0.000000f);
  if (global_thread_idx < problem_size) {
    r6 = r6 * r0;
    r5 = 1.0 / r5;
    r6 = r6 * r5;
    r20 = fma(r19, r6, r20);
    r18 = r9 * r20;
    r14 = 1.0 / r14;
    r10 = 1.0 / r10;
    r10 = r14 * r10;
    r14 = r0 * r10;
    r19 = fma(r19, r5, r14 * r18);
    r18 = r2 * r19;
    r15 = 1.0 / r15;
    r18 = r18 * r15;
    r18 = fma(r8, r18, r16);
  };
  ReadAndShuffleWithDefault2<32, double, double, double2>(
      (double*)inout_shared, eq, 0 * eq_num_alloc, global_thread_idx * eq_stride + eq_offset,
      threadIdx.x == 31, global_thread_idx <= problem_size, eq_num_alloc, eq_stride, r17, r21, r22,
      r23, 0.000000f, 0.000000f);
  if (global_thread_idx < problem_size) {
    r23 = fma(r22, r6, r23);
    r23 = r23 * r14;
    r22 = fma(r22, r5, r9 * r23);
    r22 = r0 * r22;
    r22 = r22 * r13;
    r22 = r22 * r15;
    r18 = fma(r11, r22, r18);
    r18 = 1.0 / r18;
    r21 = r19 * r15;
  };
  ReadAndShuffleWithDefault2<32, double, double, double2>(
      (double*)inout_shared, eq, 10 * eq_num_alloc, global_thread_idx * eq_stride + eq_offset,
      threadIdx.x == 31, global_thread_idx <= problem_size, eq_num_alloc, eq_stride, r17, r24, r25,
      r26, 0.000000f, 0.000000f);
  if (global_thread_idx < problem_size) {
    r24 = fma(r17, r3, r24);
    r24 = r24 * r8;
    r17 = fma(r17, r4, r7 * r24);
    r21 = fma(r17, r22, r24 * r21);
    r26 = r0 * r21;
    r25 = r11 * r13;
    r27 = r2 * r20;
    r27 = r27 * r8;
    r27 = fma(r10, r27, r23 * r25);
    r25 = r0 * r27;
  };
  ReadAndShuffleWithDefault2<32, double, double, double2>(
      (double*)inout_shared, eq, 12 * eq_num_alloc, global_thread_idx * eq_stride + eq_offset,
      threadIdx.x == 31, global_thread_idx <= problem_size, eq_num_alloc, eq_stride, r28, r29, r30,
      r31, 0.000000f, 0.000000f);
  if (global_thread_idx < problem_size) {
    r6 = fma(r30, r6, r31);
    r31 = r9 * r6;
    r31 = fma(r14, r31, r30 * r5);
    r3 = fma(r28, r3, r29);
    r29 = r19 * r3;
    r29 = r29 * r15;
    r29 = fma(r8, r29, r31 * r15);
    r31 = r7 * r3;
    r4 = fma(r28, r4, r8 * r31);
    r29 = fma(r4, r22, r29);
    r25 = r25 * r29;
    r22 = r20 * r3;
    r22 = r22 * r8;
    r22 = fma(r10, r22, r18 * r25);
    r25 = r4 * r13;
    r22 = fma(r23, r25, r22);
    r22 = fma(r6, r10, r22);
    r25 = r17 * r13;
    r25 = fma(r23, r25, r16);
    r16 = r20 * r10;
    r25 = fma(r24, r16, r25);
    r23 = r0 * r27;
    r23 = r23 * r21;
    r25 = fma(r18, r23, r25);
    r25 = 1.0 / r25;
    r25 = r22 * r25;
    r26 = fma(r25, r26, r29);
    r26 = r18 * r26;
    r1 = r1 * r13;
    r1 = fma(r4, r13, r26 * r1);
    r18 = r0 * r17;
    r18 = r18 * r13;
    r1 = fma(r25, r18, r1);
    r12 = fma(r3, r12, r25 * r24);
    r24 = r2 * r8;
    r12 = fma(r26, r24, r12);
    WriteIdx2<32, double, double, double2>(solu2km1, 0 * solu2km1_num_alloc,
                                           global_thread_idx * solu2km1_stride + solu2km1_offset,
                                           r1, r12);
    WriteIdx2<32, double, double, double2>(
        solu2k, 0 * solu2k_num_alloc, global_thread_idx * solu2k_stride + solu2k_offset, r26, r25);
  };
}

void ChainACyclicReductionSolveTwo(double* const eq, unsigned int eq_num_alloc, int eq_offset,
                                   int eq_stride, double* solu2km1, unsigned int solu2km1_num_alloc,
                                   int solu2km1_offset, int solu2km1_stride, double* solu2k,
                                   unsigned int solu2k_num_alloc, int solu2k_offset,
                                   int solu2k_stride, size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 32 - 1) / 32;
  // Kernels using more than 48 KB of shared memory per block must opt in.
  static const cudaError_t attribute_error = cudaFuncSetAttribute(
      ChainACyclicReductionSolveTwoKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 512);
  if (attribute_error != cudaSuccess) {
    throw std::runtime_error(std::string("ChainACyclicReductionSolveTwoKernel: ") +
                             cudaGetErrorString(attribute_error));
  }
  ChainACyclicReductionSolveTwoKernel<<<n_blocks, 32, 512>>>(
      eq, eq_num_alloc, eq_offset, eq_stride, solu2km1, solu2km1_num_alloc, solu2km1_offset,
      solu2km1_stride, solu2k, solu2k_num_alloc, solu2k_offset, solu2k_stride, problem_size);
  // A launch that fails (e.g. from exceeding shared memory or register limits)
  // otherwise leaves its outputs untouched without any error.
  const cudaError_t launch_error = cudaGetLastError();
  if (launch_error != cudaSuccess) {
    throw std::runtime_error(std::string("ChainACyclicReductionSolveTwoKernel launch failed: ") +
                             cudaGetErrorString(launch_error));
  }
}

}  // namespace caspar