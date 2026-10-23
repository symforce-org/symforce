#include <stdexcept>
#include <string>

#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include "kernel_ChainB_cyclic_reduction_solve_two.h"
#include "memops.cuh"

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(32, 1) ChainBCyclicReductionSolveTwoKernel(
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
         r31 = 0, r32 = 0, r33 = 0, r34 = 0, r35 = 0, r36 = 0, r37 = 0, r38 = 0, r39 = 0, r40 = 0,
         r41 = 0, r42 = 0, r43 = 0, r44 = 0, r45 = 0, r46 = 0, r47 = 0, r48 = 0, r49 = 0, r50 = 0,
         r51 = 0, r52 = 0, r53 = 0, r54 = 0, r55 = 0, r56 = 0, r57 = 0, r58 = 0, r59 = 0, r60 = 0,
         r61 = 0, r62 = 0, r63 = 0;

  if (global_thread_idx < problem_size) {
    r0 = -1.00000000000000000e+00;
  };
  ReadAndShuffleWithDefault2<32, double, double, double2>(
      (double*)inout_shared, eq, 24 * eq_num_alloc, global_thread_idx * eq_stride + eq_offset,
      threadIdx.x == 31, global_thread_idx <= problem_size, eq_num_alloc, eq_stride, r1, r2, r3, r4,
      0.000000f, 0.000000f);
  ReadAndShuffleWithDefault2<32, double, double, double2>(
      (double*)inout_shared, eq, 30 * eq_num_alloc, global_thread_idx * eq_stride + eq_offset,
      threadIdx.x == 31, global_thread_idx <= problem_size, eq_num_alloc, eq_stride, r4, r3, r5, r6,
      1.000000f, 0.000000f);
  if (global_thread_idx < problem_size) {
    r4 = 1.0 / r4;
  };
  ReadAndShuffleWithDefault2<32, double, double, double2>(
      (double*)inout_shared, eq, 36 * eq_num_alloc, global_thread_idx * eq_stride + eq_offset,
      threadIdx.x == 31, global_thread_idx <= problem_size, eq_num_alloc, eq_stride, r7, r8, r9,
      r10, 1.000000f, 0.000000f);
  if (global_thread_idx < problem_size) {
    r11 = r8 * r0;
  };
  ReadAndShuffleWithDefault2<32, double, double, double2>(
      (double*)inout_shared, eq, 32 * eq_num_alloc, global_thread_idx * eq_stride + eq_offset,
      threadIdx.x == 31, global_thread_idx <= problem_size, eq_num_alloc, eq_stride, r12, r13, r14,
      r15, 1.000000f, 0.000000f);
  if (global_thread_idx < problem_size) {
    r12 = 1.0 / r12;
    r3 = r3 * r0;
    r3 = r3 * r4;
    r2 = fma(r1, r3, r2);
  };
  ReadAndShuffleWithDefault2<32, double, double, double2>(
      (double*)inout_shared, eq, 40 * eq_num_alloc, global_thread_idx * eq_stride + eq_offset,
      threadIdx.x == 31, global_thread_idx <= problem_size, eq_num_alloc, eq_stride, r16, r17, r18,
      r19, 0.000000f, 1.000000f);
  ReadAndShuffleWithDefault2<32, double, double, double2>(
      (double*)inout_shared, eq, 26 * eq_num_alloc, global_thread_idx * eq_stride + eq_offset,
      threadIdx.x == 31, global_thread_idx <= problem_size, eq_num_alloc, eq_stride, r20, r21, r22,
      r23, 0.000000f, 0.000000f);
  ReadAndShuffleWithDefault2<32, double, double, double2>(
      (double*)inout_shared, eq, 34 * eq_num_alloc, global_thread_idx * eq_stride + eq_offset,
      threadIdx.x == 31, global_thread_idx <= problem_size, eq_num_alloc, eq_stride, r22, r24, r25,
      r26, 0.000000f, 1.000000f);
  if (global_thread_idx < problem_size) {
    r22 = r22 * r0;
    r22 = r22 * r12;
    r20 = fma(r2, r22, r20);
    r27 = r1 * r13;
    r27 = r27 * r0;
    r20 = fma(r4, r27, r20);
    r27 = r16 * r20;
    r17 = 1.0 / r17;
    r24 = 1.0 / r24;
    r24 = r17 * r24;
    r17 = r0 * r24;
    r27 = fma(r17, r27, r2 * r12);
  };
  ReadAndShuffleWithDefault2<32, double, double, double2>(
      (double*)inout_shared, eq, 38 * eq_num_alloc, global_thread_idx * eq_stride + eq_offset,
      threadIdx.x == 31, global_thread_idx <= problem_size, eq_num_alloc, eq_stride, r2, r28, r29,
      r30, 0.000000f, 1.000000f);
  if (global_thread_idx < problem_size) {
    r28 = 1.0 / r28;
    r11 = r11 * r27;
    r11 = fma(r28, r11, r1 * r4);
    r31 = r2 * r20;
    r11 = fma(r17, r31, r11);
    r31 = r0 * r11;
    r7 = 1.0 / r7;
  };
  ReadAndShuffleWithDefault2<32, double, double, double2>(
      (double*)inout_shared, eq, 4 * eq_num_alloc, global_thread_idx * eq_stride + eq_offset,
      threadIdx.x == 31, global_thread_idx <= problem_size, eq_num_alloc, eq_stride, r32, r33, r34,
      r35, 0.000000f, 0.000000f);
  if (global_thread_idx < problem_size) {
    r25 = r25 * r0;
    r14 = 1.0 / r14;
    r25 = r25 * r14;
  };
  ReadAndShuffleWithDefault2<32, double, double, double2>(
      (double*)inout_shared, eq, 2 * eq_num_alloc, global_thread_idx * eq_stride + eq_offset,
      threadIdx.x == 31, global_thread_idx <= problem_size, eq_num_alloc, eq_stride, r33, r32, r36,
      r37, 0.000000f, 0.000000f);
  if (global_thread_idx < problem_size) {
    r6 = r6 * r0;
    r5 = 1.0 / r5;
    r6 = r6 * r5;
    r34 = fma(r37, r6, r34);
    r35 = fma(r34, r25, r35);
    r32 = r37 * r15;
    r32 = r32 * r0;
    r35 = fma(r5, r32, r35);
    r26 = 1.0 / r26;
    r19 = 1.0 / r19;
    r19 = r26 * r19;
    r26 = r0 * r19;
    r32 = r35 * r26;
  };
  ReadAndShuffleWithDefault2<32, double, double, double2>(
      (double*)inout_shared, eq, 22 * eq_num_alloc, global_thread_idx * eq_stride + eq_offset,
      threadIdx.x == 31, global_thread_idx <= problem_size, eq_num_alloc, eq_stride, r33, r38, r39,
      r40, 0.000000f, 0.000000f);
  ReadAndShuffleWithDefault2<32, double, double, double2>(
      (double*)inout_shared, eq, 20 * eq_num_alloc, global_thread_idx * eq_stride + eq_offset,
      threadIdx.x == 31, global_thread_idx <= problem_size, eq_num_alloc, eq_stride, r40, r39, r41,
      r42, 0.000000f, 0.000000f);
  if (global_thread_idx < problem_size) {
    r33 = fma(r39, r3, r33);
    r38 = fma(r33, r22, r38);
    r42 = r39 * r13;
    r42 = r42 * r0;
    r38 = fma(r4, r42, r38);
    r38 = r38 * r17;
    r33 = fma(r33, r12, r16 * r38);
    r33 = r33 * r28;
  };
  ReadAndShuffleWithDefault2<32, double, double, double2>(
      (double*)inout_shared, eq, 8 * eq_num_alloc, global_thread_idx * eq_stride + eq_offset,
      threadIdx.x == 31, global_thread_idx <= problem_size, eq_num_alloc, eq_stride, r42, r41, r43,
      r44, 0.000000f, 1.000000f);
  ReadAndShuffleWithDefault2<32, double, double, double2>(
      (double*)inout_shared, eq, 6 * eq_num_alloc, global_thread_idx * eq_stride + eq_offset,
      threadIdx.x == 31, global_thread_idx <= problem_size, eq_num_alloc, eq_stride, r44, r41, r42,
      r45, 0.000000f, 0.000000f);
  if (global_thread_idx < problem_size) {
    r41 = r42 * r15;
    r41 = r41 * r0;
    r41 = fma(r5, r41, r43);
    r45 = fma(r42, r6, r45);
    r41 = fma(r45, r25, r41);
    r43 = r41 * r19;
    r43 = fma(r38, r43, r33 * r32);
    r32 = r8 * r0;
    r32 = fma(r39, r4, r33 * r32);
    r32 = fma(r2, r38, r32);
    r44 = r32 * r7;
  };
  ReadAndShuffleWithDefault2<32, double, double, double2>(
      (double*)inout_shared, eq, 0 * eq_num_alloc, global_thread_idx * eq_stride + eq_offset,
      threadIdx.x == 31, global_thread_idx <= problem_size, eq_num_alloc, eq_stride, r46, r47, r48,
      r49, 0.000000f, 0.000000f);
  if (global_thread_idx < problem_size) {
    r49 = fma(r48, r6, r49);
    r36 = fma(r49, r25, r36);
    r47 = r48 * r15;
    r47 = r47 * r0;
    r36 = fma(r5, r47, r36);
    r36 = r36 * r26;
    r43 = fma(r36, r44, r43);
    r47 = r29 * r35;
    r46 = r18 * r35;
    r34 = fma(r34, r14, r26 * r46);
    r34 = r0 * r34;
    r30 = 1.0 / r30;
    r34 = r34 * r30;
    r47 = fma(r10, r34, r26 * r47);
    r47 = fma(r37, r5, r47);
    r46 = r0 * r47;
    r9 = 1.0 / r9;
    r46 = r46 * r9;
    r50 = r10 * r0;
    r51 = r18 * r41;
    r45 = fma(r45, r14, r26 * r51);
    r50 = r50 * r45;
    r51 = r29 * r41;
    r51 = fma(r26, r51, r30 * r50);
    r51 = fma(r42, r5, r51);
    r50 = r51 * r9;
    r50 = fma(r38, r50, r33 * r46);
    r46 = r10 * r0;
    r49 = fma(r18, r36, r49 * r14);
    r46 = r46 * r49;
    r46 = fma(r30, r46, r29 * r36);
    r46 = fma(r48, r5, r46);
    r46 = r0 * r46;
    r46 = r46 * r7;
    r46 = r46 * r9;
    r50 = fma(r32, r46, r50);
    r52 = r0 * r50;
  };
  ReadAndShuffleWithDefault2<32, double, double, double2>(
      (double*)inout_shared, eq, 18 * eq_num_alloc, global_thread_idx * eq_stride + eq_offset,
      threadIdx.x == 31, global_thread_idx <= problem_size, eq_num_alloc, eq_stride, r53, r54, r55,
      r56, 0.000000f, 0.000000f);
  if (global_thread_idx < problem_size) {
    r54 = fma(r53, r3, r54);
    r40 = fma(r54, r22, r40);
    r56 = r53 * r13;
    r56 = r56 * r0;
    r40 = fma(r4, r56, r40);
    r56 = r41 * r40;
    r56 = r56 * r17;
    r55 = r8 * r0;
    r57 = r16 * r40;
    r54 = fma(r54, r12, r17 * r57);
    r55 = r55 * r54;
    r57 = r2 * r40;
    r57 = fma(r17, r57, r28 * r55);
    r57 = fma(r53, r4, r57);
    r55 = r57 * r7;
    r55 = fma(r36, r55, r19 * r56);
    r56 = r35 * r54;
    r56 = r56 * r28;
    r55 = fma(r26, r56, r55);
    r56 = 1.00000000000000000e+00;
    r58 = r40 * r51;
    r58 = r58 * r9;
    r58 = fma(r17, r58, r56);
    r59 = r0 * r54;
    r59 = r59 * r47;
    r59 = r59 * r28;
    r58 = fma(r9, r59, r58);
    r58 = fma(r57, r46, r58);
    r58 = 1.0 / r58;
    r52 = r52 * r55;
    r43 = fma(r58, r52, r43);
    r52 = r0 * r43;
    r44 = r45 * r40;
    r44 = r44 * r30;
    r59 = r0 * r49;
    r59 = r59 * r57;
    r59 = r59 * r7;
    r59 = fma(r30, r59, r17 * r44);
    r44 = r54 * r28;
    r59 = fma(r34, r44, r59);
    r44 = r0 * r59;
  };
  ReadAndShuffleWithDefault2<32, double, double, double2>(
      (double*)inout_shared, eq, 28 * eq_num_alloc, global_thread_idx * eq_stride + eq_offset,
      threadIdx.x == 31, global_thread_idx <= problem_size, eq_num_alloc, eq_stride, r60, r61, r62,
      r63, 0.000000f, 0.000000f);
  if (global_thread_idx < problem_size) {
    r3 = fma(r21, r3, r60);
    r22 = fma(r3, r22, r61);
    r61 = r21 * r13;
    r61 = r61 * r0;
    r22 = fma(r4, r61, r22);
    r61 = r2 * r22;
    r4 = fma(r21, r4, r17 * r61);
    r61 = r8 * r0;
    r60 = r16 * r22;
    r60 = fma(r17, r60, r3 * r12);
    r61 = r61 * r60;
    r4 = fma(r28, r61, r4);
    r61 = r0 * r47;
    r61 = r61 * r60;
    r61 = r61 * r28;
    r61 = fma(r9, r61, r4 * r46);
    r12 = r51 * r22;
    r12 = r12 * r9;
    r61 = fma(r17, r12, r61);
    r6 = fma(r23, r6, r62);
    r25 = fma(r6, r25, r63);
    r63 = r23 * r15;
    r63 = r63 * r0;
    r25 = fma(r5, r63, r25);
    r63 = r29 * r25;
    r5 = fma(r23, r5, r26 * r63);
    r63 = r10 * r0;
    r62 = r18 * r25;
    r14 = fma(r6, r14, r26 * r62);
    r63 = r63 * r14;
    r5 = fma(r30, r63, r5);
    r61 = fma(r5, r9, r61);
    r44 = r44 * r61;
    r5 = r0 * r49;
    r5 = r5 * r4;
    r5 = r5 * r7;
    r5 = fma(r30, r5, r58 * r44);
    r44 = r45 * r22;
    r44 = r44 * r30;
    r5 = fma(r17, r44, r5);
    r12 = r60 * r28;
    r5 = fma(r34, r12, r5);
    r5 = fma(r14, r30, r5);
    r14 = r45 * r30;
    r14 = fma(r38, r14, r56);
    r12 = r0 * r59;
    r12 = r12 * r50;
    r14 = fma(r58, r12, r14);
    r44 = r0 * r49;
    r44 = r44 * r32;
    r44 = r44 * r7;
    r14 = fma(r30, r44, r14);
    r14 = fma(r34, r33, r14);
    r14 = 1.0 / r14;
    r52 = r52 * r5;
    r44 = r41 * r22;
    r44 = r44 * r17;
    r44 = fma(r19, r44, r14 * r52);
    r52 = r0 * r55;
    r52 = r52 * r61;
    r44 = fma(r58, r52, r44);
    r12 = r35 * r60;
    r12 = r12 * r28;
    r44 = fma(r26, r12, r44);
    r63 = r4 * r7;
    r44 = fma(r36, r63, r44);
    r44 = fma(r25, r19, r44);
    r63 = r0 * r43;
    r12 = r27 * r28;
    r52 = r0 * r49;
    r52 = r52 * r11;
    r52 = r52 * r7;
    r52 = fma(r30, r52, r34 * r12);
    r12 = r0 * r59;
    r34 = r51 * r20;
    r34 = r34 * r9;
    r34 = fma(r17, r34, r11 * r46);
    r46 = r0 * r47;
    r46 = r46 * r27;
    r46 = r46 * r28;
    r34 = fma(r9, r46, r34);
    r12 = r12 * r34;
    r52 = fma(r58, r12, r52);
    r46 = r45 * r20;
    r46 = r46 * r30;
    r52 = fma(r17, r46, r52);
    r63 = r63 * r52;
    r63 = fma(r14, r63, r56);
    r56 = r41 * r20;
    r56 = r56 * r17;
    r63 = fma(r19, r56, r63);
    r46 = r0 * r55;
    r46 = r46 * r34;
    r63 = fma(r58, r46, r63);
    r12 = r35 * r27;
    r12 = r12 * r28;
    r63 = fma(r26, r12, r63);
    r6 = r11 * r7;
    r63 = fma(r36, r6, r63);
    r63 = 1.0 / r63;
    r63 = r44 * r63;
    r31 = r31 * r7;
    r44 = r57 * r7;
    r6 = r0 * r50;
    r12 = r0 * r52;
    r12 = fma(r63, r12, r5);
    r12 = r14 * r12;
    r14 = r0 * r34;
    r14 = fma(r63, r14, r12 * r6);
    r14 = r14 + r61;
    r14 = r58 * r14;
    r58 = r0 * r14;
    r44 = fma(r58, r44, r63 * r31);
    r31 = r0 * r32;
    r31 = r31 * r7;
    r44 = fma(r12, r31, r44);
    r44 = fma(r4, r7, r44);
    r31 = r54 * r28;
    r61 = r0 * r27;
    r61 = r61 * r28;
    r61 = fma(r63, r61, r58 * r31);
    r31 = r0 * r33;
    r61 = fma(r12, r31, r61);
    r61 = fma(r60, r28, r61);
    WriteIdx2<32, double, double, double2>(solu2km1, 0 * solu2km1_num_alloc,
                                           global_thread_idx * solu2km1_stride + solu2km1_offset,
                                           r44, r61);
    r61 = r40 * r17;
    r61 = fma(r14, r61, r22 * r24);
    r24 = r20 * r17;
    r61 = fma(r63, r24, r61);
    r61 = fma(r38, r12, r61);
    WriteIdx1<32, double, double, double>(solu2km1, 2 * solu2km1_num_alloc,
                                          global_thread_idx * solu2km1_stride + solu2km1_offset,
                                          r61);
    WriteIdx2<32, double, double, double2>(
        solu2k, 0 * solu2k_num_alloc, global_thread_idx * solu2k_stride + solu2k_offset, r14, r12);
    WriteIdx1<32, double, double, double>(solu2k, 2 * solu2k_num_alloc,
                                          global_thread_idx * solu2k_stride + solu2k_offset, r63);
  };
}

void ChainBCyclicReductionSolveTwo(double* const eq, unsigned int eq_num_alloc, int eq_offset,
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
      ChainBCyclicReductionSolveTwoKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 512);
  if (attribute_error != cudaSuccess) {
    throw std::runtime_error(std::string("ChainBCyclicReductionSolveTwoKernel: ") +
                             cudaGetErrorString(attribute_error));
  }
  ChainBCyclicReductionSolveTwoKernel<<<n_blocks, 32, 512>>>(
      eq, eq_num_alloc, eq_offset, eq_stride, solu2km1, solu2km1_num_alloc, solu2km1_offset,
      solu2km1_stride, solu2k, solu2k_num_alloc, solu2k_offset, solu2k_stride, problem_size);
  // A launch that fails (e.g. from exceeding shared memory or register limits)
  // otherwise leaves its outputs untouched without any error.
  const cudaError_t launch_error = cudaGetLastError();
  if (launch_error != cudaSuccess) {
    throw std::runtime_error(std::string("ChainBCyclicReductionSolveTwoKernel launch failed: ") +
                             cudaGetErrorString(launch_error));
  }
}

}  // namespace caspar