#include <stdexcept>
#include <string>

#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include "kernel_ChainA_cyclic_reduction_up.h"
#include "memops.cuh"

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(32, 1)
    ChainACyclicReductionUpKernel(double* eqk, unsigned int eqk_num_alloc, int eqk_offset,
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
         r11 = 0, r12 = 0, r13 = 0, r14 = 0;
  ReadAndShuffleWithDefault2<32, double, double, double2>(
      (double*)inout_shared, x, 0 * x_num_alloc, global_thread_idx * x_stride + x_offset,
      threadIdx.x == 31, global_thread_idx <= problem_size, x_num_alloc, x_stride, r0, r1, r2, r3,
      0.000000f, 0.000000f);
  if (global_thread_idx < problem_size) {
    ReadIdx2<32, double, double, double2>(eqk, 12 * eqk_num_alloc,
                                          global_thread_idx * eqk_stride + eqk_offset, r4, r5);
    r6 = -1.00000000000000000e+00;
    ReadIdx2<32, double, double, double2>(eqk, 0 * eqk_num_alloc,
                                          global_thread_idx * eqk_stride + eqk_offset, r7, r8);
    ReadIdx2<32, double, double, double2>(eqk, 2 * eqk_num_alloc,
                                          global_thread_idx * eqk_stride + eqk_offset, r9, r10);
    r10 = fma(r10, r1, r8 * r0);
    r10 = fma(r6, r10, r5);
    ReadIdx2<32, double, double, double2>(eqk, 10 * eqk_num_alloc,
                                          global_thread_idx * eqk_stride + eqk_offset, r5, r8);
    ReadIdx2<32, double, double, double2>(eqk, 8 * eqk_num_alloc,
                                          global_thread_idx * eqk_stride + eqk_offset, r11, r12);
    r12 = fma(r12, r2, r8 * r3);
    ReadIdx2<32, double, double, double2>(eqk, 14 * eqk_num_alloc,
                                          global_thread_idx * eqk_stride + eqk_offset, r8, r13);
    r14 = r13 * r6;
    r0 = fma(r7, r0, r9 * r1);
    r0 = fma(r6, r0, r4);
    r2 = fma(r11, r2, r5 * r3);
    r0 = fma(r6, r2, r0);
    r8 = 1.0 / r8;
    r8 = r0 * r8;
    r10 = fma(r8, r14, r10);
    r10 = fma(r6, r12, r10);
    ReadIdx2<32, double, double, double2>(eqk, 16 * eqk_num_alloc,
                                          global_thread_idx * eqk_stride + eqk_offset, r14, r12);
    r14 = 1.0 / r14;
    r14 = r10 * r14;
    ReadIdx2<32, double, double, double2>(eqk, 18 * eqk_num_alloc,
                                          global_thread_idx * eqk_stride + eqk_offset, r10, r0);
    r0 = 1.0 / r0;
    r14 = r14 * r0;
    r0 = r10 * r6;
    r0 = fma(r14, r0, r8);
    r12 = 1.0 / r12;
    r12 = r0 * r12;
    WriteIdx2<32, double, double, double2>(
        soluk, 0 * soluk_num_alloc, global_thread_idx * soluk_stride + soluk_offset, r12, r14);
  };
}

void ChainACyclicReductionUp(double* eqk, unsigned int eqk_num_alloc, int eqk_offset,
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
      ChainACyclicReductionUpKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 512);
  if (attribute_error != cudaSuccess) {
    throw std::runtime_error(std::string("ChainACyclicReductionUpKernel: ") +
                             cudaGetErrorString(attribute_error));
  }
  ChainACyclicReductionUpKernel<<<n_blocks, 32, 512>>>(
      eqk, eqk_num_alloc, eqk_offset, eqk_stride, x, x_num_alloc, x_offset, x_stride, soluk,
      soluk_num_alloc, soluk_offset, soluk_stride, problem_size);
  // A launch that fails (e.g. from exceeding shared memory or register limits)
  // otherwise leaves its outputs untouched without any error.
  const cudaError_t launch_error = cudaGetLastError();
  if (launch_error != cudaSuccess) {
    throw std::runtime_error(std::string("ChainACyclicReductionUpKernel launch failed: ") +
                             cudaGetErrorString(launch_error));
  }
}

}  // namespace caspar