#include <stdexcept>
#include <string>

#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include "kernel_ChainA_cyclic_reduction_down.h"
#include "memops.cuh"

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(64, 1)
    ChainACyclicReductionDownKernel(double* eqk, unsigned int eqk_num_alloc, int eqk_offset,
                                    int eqk_stride, double* const eq, unsigned int eq_num_alloc,
                                    int eq_offset, int eq_stride, double* eq_marked,
                                    unsigned int eq_marked_num_alloc, int eq_marked_offset,
                                    int eq_marked_stride, size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  // Scratch data, then the shared indices of each shared accessor; dynamic, so
  // that it may exceed the 48 KB static shared memory limit (see the launch
  // below).
  extern __shared__ __align__(16) uint8_t inout_shared[];

  double r0 = 0, r1 = 0, r2 = 0, r3 = 0, r4 = 0, r5 = 0, r6 = 0, r7 = 0, r8 = 0, r9 = 0, r10 = 0,
         r11 = 0, r12 = 0, r13 = 0, r14 = 0, r15 = 0, r16 = 0, r17 = 0, r18 = 0, r19 = 0, r20 = 0,
         r21 = 0, r22 = 0, r23 = 0, r24 = 0, r25 = 0, r26 = 0, r27 = 0, r28 = 0, r29 = 0, r30 = 0,
         r31 = 0, r32 = 0, r33 = 0, r34 = 0, r35 = 0, r36 = 0, r37 = 0, r38 = 0, r39 = 0, r40 = 0,
         r41 = 0;

  if (global_thread_idx < problem_size) {
    ReadIdx2<64, double, double, double2>(eqk, 0 * eqk_num_alloc,
                                          global_thread_idx * eqk_stride + eqk_offset, r0, r1);
  };
  ReadAndShuffleWithDefault2<64, double, double, double2>(
      (double*)inout_shared, eq, 0 * eq_num_alloc, global_thread_idx * eq_stride + eq_offset,
      threadIdx.x == 63, global_thread_idx <= problem_size, eq_num_alloc, eq_stride, r2, r3, r4, r5,
      0.000000f, 0.000000f);
  if (global_thread_idx < problem_size) {
    r6 = -1.00000000000000000e+00;
    r7 = r2 * r6;
  };
  ReadAndShuffleWithDefault2<64, double, double, double2>(
      (double*)inout_shared, eq, 16 * eq_num_alloc, global_thread_idx * eq_stride + eq_offset,
      threadIdx.x == 63, global_thread_idx <= problem_size, eq_num_alloc, eq_stride, r8, r9, r10,
      r11, 1.000000f, 1.000000f);
  if (global_thread_idx < problem_size) {
    r9 = 1.0 / r9;
  };
  ReadAndShuffleWithDefault2<64, double, double, double2>(
      (double*)inout_shared, eq, 14 * eq_num_alloc, global_thread_idx * eq_stride + eq_offset,
      threadIdx.x == 63, global_thread_idx <= problem_size, eq_num_alloc, eq_stride, r12, r13, r14,
      r15, 1.000000f, 0.000000f);
  if (global_thread_idx < problem_size) {
    ReadIdx2<64, double, double, double2>(eqk, 2 * eqk_num_alloc,
                                          global_thread_idx * eqk_stride + eqk_offset, r16, r17);
  };
  ReadAndShuffleWithDefault2<64, double, double, double2>(
      (double*)inout_shared, eq, 18 * eq_num_alloc, global_thread_idx * eq_stride + eq_offset,
      threadIdx.x == 63, global_thread_idx <= problem_size, eq_num_alloc, eq_stride, r18, r19, r20,
      r21, 0.000000f, 1.000000f);
  if (global_thread_idx < problem_size) {
    r18 = r18 * r6;
    r18 = r18 * r9;
    r16 = fma(r0, r18, r16);
    r19 = 1.0 / r19;
    r8 = 1.0 / r8;
    r8 = r19 * r8;
    r19 = r6 * r8;
    r22 = r16 * r19;
    r0 = fma(r13, r22, r0 * r9);
    r12 = 1.0 / r12;
    r0 = r0 * r12;
    r7 = fma(r3, r22, r0 * r7);
    r23 = r2 * r6;
    r18 = fma(r1, r18, r17);
    r17 = r13 * r18;
    r17 = fma(r19, r17, r1 * r9);
    r23 = r23 * r17;
    r9 = r3 * r18;
    r9 = fma(r19, r9, r12 * r23);
    WriteIdx2<64, double, double, double2>(eq_marked, 0 * eq_marked_num_alloc,
                                           global_thread_idx * eq_marked_stride + eq_marked_offset,
                                           r7, r9);
  };
  ReadAndShuffleWithDefault2<64, double, double, double2>(
      (double*)inout_shared, eq, 2 * eq_num_alloc, global_thread_idx * eq_stride + eq_offset,
      threadIdx.x == 63, global_thread_idx <= problem_size, eq_num_alloc, eq_stride, r9, r7, r23,
      r1, 0.000000f, 0.000000f);
  if (global_thread_idx < problem_size) {
    r24 = r9 * r6;
    r22 = fma(r7, r22, r0 * r24);
    r24 = r9 * r6;
    r24 = r24 * r17;
    r25 = r7 * r18;
    r25 = fma(r19, r25, r12 * r24);
    WriteIdx2<64, double, double, double2>(eq_marked, 2 * eq_marked_num_alloc,
                                           global_thread_idx * eq_marked_stride + eq_marked_offset,
                                           r22, r25);
    ReadIdx2<64, double, double, double2>(eqk, 4 * eqk_num_alloc,
                                          global_thread_idx * eqk_stride + eqk_offset, r25, r22);
    ReadIdx2<64, double, double, double2>(eqk, 8 * eqk_num_alloc,
                                          global_thread_idx * eqk_stride + eqk_offset, r24, r19);
    r11 = 1.0 / r11;
    ReadIdx2<64, double, double, double2>(eqk, 10 * eqk_num_alloc,
                                          global_thread_idx * eqk_stride + eqk_offset, r26, r27);
    r20 = r20 * r6;
    r20 = r20 * r11;
    r26 = fma(r24, r20, r26);
    r21 = 1.0 / r21;
    r10 = 1.0 / r10;
    r10 = r21 * r10;
    r26 = r26 * r10;
    r21 = r6 * r26;
    r24 = fma(r15, r21, r24 * r11);
    r14 = 1.0 / r14;
    r24 = r24 * r14;
    r28 = fma(r5, r26, r4 * r24);
    r28 = fma(r6, r28, r25);
  };
  ReadAndShuffleWithDefault2<64, double, double, double2>(
      (double*)inout_shared, eq, 8 * eq_num_alloc, global_thread_idx * eq_stride + eq_offset,
      threadIdx.x == 63, global_thread_idx <= problem_size, eq_num_alloc, eq_stride, r25, r29, r30,
      r31, 0.000000f, 0.000000f);
  if (global_thread_idx < problem_size) {
    r32 = r29 * r16;
    r32 = fma(r8, r32, r25 * r0);
    r28 = fma(r6, r32, r28);
    r32 = r25 * r17;
    r33 = r29 * r18;
    r33 = fma(r8, r33, r12 * r32);
    r33 = fma(r6, r33, r22);
    r22 = r15 * r6;
    r20 = fma(r19, r20, r27);
    r22 = r22 * r20;
    r22 = fma(r10, r22, r19 * r11);
    r11 = r4 * r22;
    r19 = r5 * r20;
    r19 = fma(r10, r19, r14 * r11);
    r33 = fma(r6, r19, r33);
    WriteIdx2<64, double, double, double2>(eq_marked, 4 * eq_marked_num_alloc,
                                           global_thread_idx * eq_marked_stride + eq_marked_offset,
                                           r28, r33);
    ReadIdx2<64, double, double, double2>(eqk, 6 * eqk_num_alloc,
                                          global_thread_idx * eqk_stride + eqk_offset, r19, r11);
    r27 = fma(r1, r26, r23 * r24);
    r27 = fma(r6, r27, r19);
  };
  ReadAndShuffleWithDefault2<64, double, double, double2>(
      (double*)inout_shared, eq, 10 * eq_num_alloc, global_thread_idx * eq_stride + eq_offset,
      threadIdx.x == 63, global_thread_idx <= problem_size, eq_num_alloc, eq_stride, r19, r32, r34,
      r35, 0.000000f, 0.000000f);
  if (global_thread_idx < problem_size) {
    r36 = r32 * r16;
    r36 = fma(r8, r36, r19 * r0);
    r27 = fma(r6, r36, r27);
    r36 = r19 * r17;
    r37 = r32 * r18;
    r37 = fma(r8, r37, r12 * r36);
    r37 = fma(r6, r37, r11);
    r11 = r23 * r22;
    r36 = r1 * r20;
    r36 = fma(r10, r36, r14 * r11);
    r37 = fma(r6, r36, r37);
    WriteIdx2<64, double, double, double2>(eq_marked, 6 * eq_marked_num_alloc,
                                           global_thread_idx * eq_marked_stride + eq_marked_offset,
                                           r27, r37);
    r36 = r30 * r6;
    r36 = fma(r31, r21, r24 * r36);
    r11 = r30 * r6;
    r11 = r11 * r22;
    r38 = r31 * r6;
    r38 = r38 * r20;
    r38 = fma(r10, r38, r14 * r11);
    WriteIdx2<64, double, double, double2>(eq_marked, 8 * eq_marked_num_alloc,
                                           global_thread_idx * eq_marked_stride + eq_marked_offset,
                                           r36, r38);
    r38 = r34 * r6;
    r21 = fma(r35, r21, r24 * r38);
    r38 = r34 * r6;
    r38 = r38 * r22;
    r36 = r35 * r6;
    r36 = r36 * r20;
    r36 = fma(r10, r36, r14 * r38);
    WriteIdx2<64, double, double, double2>(eq_marked, 10 * eq_marked_num_alloc,
                                           global_thread_idx * eq_marked_stride + eq_marked_offset,
                                           r21, r36);
    ReadIdx2<64, double, double, double2>(eqk, 12 * eqk_num_alloc,
                                          global_thread_idx * eqk_stride + eqk_offset, r36, r21);
  };
  ReadAndShuffleWithDefault2<64, double, double, double2>(
      (double*)inout_shared, eq, 12 * eq_num_alloc, global_thread_idx * eq_stride + eq_offset,
      threadIdx.x == 63, global_thread_idx <= problem_size, eq_num_alloc, eq_stride, r38, r11, r39,
      r40, 0.000000f, 0.000000f);
  if (global_thread_idx < problem_size) {
    r41 = r11 * r16;
    r41 = fma(r8, r41, r38 * r0);
    r41 = fma(r6, r41, r36);
    r26 = fma(r40, r26, r39 * r24);
    r41 = fma(r6, r26, r41);
    r26 = r39 * r22;
    r24 = r40 * r20;
    r24 = fma(r10, r24, r14 * r26);
    r24 = fma(r6, r24, r21);
    r21 = r38 * r17;
    r26 = r11 * r18;
    r26 = fma(r8, r26, r12 * r21);
    r24 = fma(r6, r26, r24);
    WriteIdx2<64, double, double, double2>(eq_marked, 12 * eq_marked_num_alloc,
                                           global_thread_idx * eq_marked_stride + eq_marked_offset,
                                           r41, r24);
    ReadIdx2<64, double, double, double2>(eqk, 14 * eqk_num_alloc,
                                          global_thread_idx * eqk_stride + eqk_offset, r24, r41);
    r41 = 1.00000000000000000e+00;
    r24 = 1.0 / r28;
    r24 = r33 * r24;
    WriteIdx2<64, double, double, double2>(eq_marked, 14 * eq_marked_num_alloc,
                                           global_thread_idx * eq_marked_stride + eq_marked_offset,
                                           r41, r24);
    ReadIdx2<64, double, double, double2>(eqk, 16 * eqk_num_alloc,
                                          global_thread_idx * eqk_stride + eqk_offset, r33, r26);
    WriteIdx2<64, double, double, double2>(eq_marked, 16 * eq_marked_num_alloc,
                                           global_thread_idx * eq_marked_stride + eq_marked_offset,
                                           r41, r28);
    ReadIdx2<64, double, double, double2>(eqk, 18 * eqk_num_alloc,
                                          global_thread_idx * eqk_stride + eqk_offset, r28, r41);
    r41 = r6 * r27;
    r41 = fma(r24, r41, r37);
    WriteIdx2<64, double, double, double2>(eq_marked, 18 * eq_marked_num_alloc,
                                           global_thread_idx * eq_marked_stride + eq_marked_offset,
                                           r27, r41);
  };
}

void ChainACyclicReductionDown(double* eqk, unsigned int eqk_num_alloc, int eqk_offset,
                               int eqk_stride, double* const eq, unsigned int eq_num_alloc,
                               int eq_offset, int eq_stride, double* eq_marked,
                               unsigned int eq_marked_num_alloc, int eq_marked_offset,
                               int eq_marked_stride, size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 64 - 1) / 64;
  // Kernels using more than 48 KB of shared memory per block must opt in.
  static const cudaError_t attribute_error = cudaFuncSetAttribute(
      ChainACyclicReductionDownKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 1024);
  if (attribute_error != cudaSuccess) {
    throw std::runtime_error(std::string("ChainACyclicReductionDownKernel: ") +
                             cudaGetErrorString(attribute_error));
  }
  ChainACyclicReductionDownKernel<<<n_blocks, 64, 1024>>>(
      eqk, eqk_num_alloc, eqk_offset, eqk_stride, eq, eq_num_alloc, eq_offset, eq_stride, eq_marked,
      eq_marked_num_alloc, eq_marked_offset, eq_marked_stride, problem_size);
  // A launch that fails (e.g. from exceeding shared memory or register limits)
  // otherwise leaves its outputs untouched without any error.
  const cudaError_t launch_error = cudaGetLastError();
  if (launch_error != cudaSuccess) {
    throw std::runtime_error(std::string("ChainACyclicReductionDownKernel launch failed: ") +
                             cudaGetErrorString(launch_error));
  }
}

}  // namespace caspar