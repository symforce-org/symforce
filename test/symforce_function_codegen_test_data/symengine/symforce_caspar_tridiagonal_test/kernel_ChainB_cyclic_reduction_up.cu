#include <stdexcept>
#include <string>

#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include "kernel_ChainB_cyclic_reduction_up.h"
#include "memops.cuh"

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(32, 1)
    ChainBCyclicReductionUpKernel(double* eqk, unsigned int eqk_num_alloc, int eqk_offset,
                                  int eqk_stride, double* const x, unsigned int x_num_alloc,
                                  int x_offset, int x_stride, double* soluk,
                                  unsigned int soluk_num_alloc, int soluk_offset, int soluk_stride,
                                  size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  // Scratch data, then the shared indices of each shared accessor; dynamic, so
  // that it may exceed the 48 KB static shared memory limit (see the launch
  // below).
  extern __shared__ __align__(16) uint8_t inout_shared[];

  double r0 = 0, r1 = 0, r2 = 0, r3 = 0, r4 = 0, r5 = 0, r6 = 0, r7 = 0, r8 = 0, r9 = 0, r10 = 0,
         r11 = 0, r12 = 0, r13 = 0, r14 = 0, r15 = 0, r16 = 0, r17 = 0, r18 = 0, r19 = 0, r20 = 0,
         r21 = 0, r22 = 0;
  ReadAndShuffleWithDefault2<32, double, double, double2>(
      (double*)inout_shared, x, 0 * x_num_alloc, global_thread_idx * x_stride + x_offset,
      threadIdx.x == 31, global_thread_idx <= problem_size, x_num_alloc, x_stride, r0, r1, r2, r3,
      0.000000f, 0.000000f);
  if (global_thread_idx < problem_size) {
    ReadIdx2<32, double, double, double2>(eqk, 28 * eqk_num_alloc,
                                          global_thread_idx * eqk_stride + eqk_offset, r4, r5);
    r6 = -1.00000000000000000e+00;
    ReadIdx2<32, double, double, double2>(eqk, 22 * eqk_num_alloc,
                                          global_thread_idx * eqk_stride + eqk_offset, r7, r8);
    ReadIdx2<32, double, double, double2>(eqk, 18 * eqk_num_alloc,
                                          global_thread_idx * eqk_stride + eqk_offset, r9, r10);
    r10 = fma(r10, r2, r7 * r3);
    ReadIdx2<32, double, double, double2>(eqk, 24 * eqk_num_alloc,
                                          global_thread_idx * eqk_stride + eqk_offset, r7, r11);
  };
  ReadAndShuffleWithDefault1<32, double, double, double>(
      (double*)inout_shared, x, 2 * x_num_alloc, global_thread_idx * x_stride + x_offset,
      threadIdx.x == 31, global_thread_idx <= problem_size, x_num_alloc, x_stride, r12, r13,
      0.000000f);
  if (global_thread_idx < problem_size) {
    r10 = fma(r11, r13, r10);
    r10 = fma(r6, r10, r4);
    ReadIdx2<32, double, double, double2>(eqk, 0 * eqk_num_alloc,
                                          global_thread_idx * eqk_stride + eqk_offset, r4, r11);
    ReadIdx2<32, double, double, double2>(eqk, 4 * eqk_num_alloc,
                                          global_thread_idx * eqk_stride + eqk_offset, r14, r15);
    r14 = fma(r14, r1, r11 * r0);
    ReadIdx2<32, double, double, double2>(eqk, 6 * eqk_num_alloc,
                                          global_thread_idx * eqk_stride + eqk_offset, r11, r16);
    r14 = fma(r16, r12, r14);
    ReadIdx2<32, double, double, double2>(eqk, 30 * eqk_num_alloc,
                                          global_thread_idx * eqk_stride + eqk_offset, r16, r17);
    r18 = r17 * r6;
    ReadIdx2<32, double, double, double2>(eqk, 26 * eqk_num_alloc,
                                          global_thread_idx * eqk_stride + eqk_offset, r19, r20);
    ReadIdx2<32, double, double, double2>(eqk, 20 * eqk_num_alloc,
                                          global_thread_idx * eqk_stride + eqk_offset, r21, r22);
    r7 = fma(r7, r13, r22 * r3);
    r7 = fma(r9, r2, r7);
    r7 = fma(r6, r7, r20);
    ReadIdx2<32, double, double, double2>(eqk, 2 * eqk_num_alloc,
                                          global_thread_idx * eqk_stride + eqk_offset, r20, r9);
    r4 = fma(r4, r0, r9 * r1);
    r4 = fma(r11, r12, r4);
    r7 = fma(r6, r4, r7);
    r16 = 1.0 / r16;
    r16 = r7 * r16;
    r10 = fma(r16, r18, r10);
    r10 = fma(r6, r14, r10);
    ReadIdx2<32, double, double, double2>(eqk, 32 * eqk_num_alloc,
                                          global_thread_idx * eqk_stride + eqk_offset, r18, r14);
    r18 = 1.0 / r18;
    r18 = r10 * r18;
    ReadIdx2<32, double, double, double2>(eqk, 40 * eqk_num_alloc,
                                          global_thread_idx * eqk_stride + eqk_offset, r10, r7);
    r2 = fma(r21, r2, r19 * r13);
    r2 = fma(r8, r3, r2);
    r2 = fma(r6, r2, r5);
    ReadIdx2<32, double, double, double2>(eqk, 8 * eqk_num_alloc,
                                          global_thread_idx * eqk_stride + eqk_offset, r5, r3);
    r12 = fma(r5, r12, r20 * r0);
    r12 = fma(r15, r1, r12);
    ReadIdx2<32, double, double, double2>(eqk, 34 * eqk_num_alloc,
                                          global_thread_idx * eqk_stride + eqk_offset, r1, r15);
    r5 = r1 * r6;
    r2 = fma(r18, r5, r2);
    r0 = r14 * r6;
    r2 = fma(r16, r0, r2);
    r2 = fma(r6, r12, r2);
    r15 = 1.0 / r15;
    r15 = r2 * r15;
    r7 = 1.0 / r7;
    r15 = r15 * r7;
    r7 = r6 * r15;
    r10 = fma(r10, r7, r18);
    ReadIdx2<32, double, double, double2>(eqk, 38 * eqk_num_alloc,
                                          global_thread_idx * eqk_stride + eqk_offset, r18, r2);
    r2 = 1.0 / r2;
    r2 = r10 * r2;
    r7 = fma(r18, r7, r16);
    ReadIdx2<32, double, double, double2>(eqk, 36 * eqk_num_alloc,
                                          global_thread_idx * eqk_stride + eqk_offset, r18, r16);
    r10 = r16 * r6;
    r7 = fma(r2, r10, r7);
    r18 = 1.0 / r18;
    r18 = r7 * r18;
    WriteIdx2<32, double, double, double2>(
        soluk, 0 * soluk_num_alloc, global_thread_idx * soluk_stride + soluk_offset, r18, r2);
    WriteIdx1<32, double, double, double>(soluk, 2 * soluk_num_alloc,
                                          global_thread_idx * soluk_stride + soluk_offset, r15);
  };
}

void ChainBCyclicReductionUp(double* eqk, unsigned int eqk_num_alloc, int eqk_offset,
                             int eqk_stride, double* const x, unsigned int x_num_alloc,
                             int x_offset, int x_stride, double* soluk,
                             unsigned int soluk_num_alloc, int soluk_offset, int soluk_stride,
                             size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 32 - 1) / 32;
  // Kernels using more than 48 KB of shared memory per block must opt in.
  static const cudaError_t attribute_error = cudaFuncSetAttribute(
      ChainBCyclicReductionUpKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 512);
  if (attribute_error != cudaSuccess) {
    throw std::runtime_error(std::string("ChainBCyclicReductionUpKernel: ") +
                             cudaGetErrorString(attribute_error));
  }
  ChainBCyclicReductionUpKernel<<<n_blocks, 32, 512>>>(
      eqk, eqk_num_alloc, eqk_offset, eqk_stride, x, x_num_alloc, x_offset, x_stride, soluk,
      soluk_num_alloc, soluk_offset, soluk_stride, problem_size);
  // A launch that fails (e.g. from exceeding shared memory or register limits)
  // otherwise leaves its outputs untouched without any error.
  const cudaError_t launch_error = cudaGetLastError();
  if (launch_error != cudaSuccess) {
    throw std::runtime_error(std::string("ChainBCyclicReductionUpKernel launch failed: ") +
                             cudaGetErrorString(launch_error));
  }
}

}  // namespace caspar