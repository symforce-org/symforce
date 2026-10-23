#include <stdexcept>
#include <string>

#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include "kernel_ChainB_update_p.h"
#include "memops.cuh"

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    ChainBUpdatePKernel(double* ChainB_z, unsigned int ChainB_z_num_alloc, double* ChainB_p_k,
                        unsigned int ChainB_p_k_num_alloc, const double* const beta,
                        double* out_ChainB_p_kp1, unsigned int out_ChainB_p_kp1_num_alloc,
                        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  // Scratch data, then the shared indices of each shared accessor; dynamic, so
  // that it may exceed the 48 KB static shared memory limit (see the launch
  // below).
  extern __shared__ __align__(16) uint8_t inout_shared[];

  double r0 = 0, r1 = 0, r2 = 0, r3 = 0, r4 = 0;

  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(ChainB_p_k, 0 * ChainB_p_k_num_alloc, global_thread_idx,
                                            r0, r1);
    ReadIdx2<1024, double, double, double2>(ChainB_z, 0 * ChainB_z_num_alloc, global_thread_idx, r2,
                                            r3);
  };
  LoadUnique<1, double, double>(beta, 0, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>((double*)inout_shared, 0, r4);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r0 = fma(r0, r4, r2);
    r1 = fma(r1, r4, r3);
    WriteIdx2<1024, double, double, double2>(out_ChainB_p_kp1, 0 * out_ChainB_p_kp1_num_alloc,
                                             global_thread_idx, r0, r1);
    ReadIdx1<1024, double, double, double>(ChainB_p_k, 2 * ChainB_p_k_num_alloc, global_thread_idx,
                                           r1);
    ReadIdx1<1024, double, double, double>(ChainB_z, 2 * ChainB_z_num_alloc, global_thread_idx, r0);
    r4 = fma(r1, r4, r0);
    WriteIdx1<1024, double, double, double>(out_ChainB_p_kp1, 2 * out_ChainB_p_kp1_num_alloc,
                                            global_thread_idx, r4);
  };
}

void ChainBUpdateP(double* ChainB_z, unsigned int ChainB_z_num_alloc, double* ChainB_p_k,
                   unsigned int ChainB_p_k_num_alloc, const double* const beta,
                   double* out_ChainB_p_kp1, unsigned int out_ChainB_p_kp1_num_alloc,
                   size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  // Kernels using more than 48 KB of shared memory per block must opt in.
  static const cudaError_t attribute_error =
      cudaFuncSetAttribute(ChainBUpdatePKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 8192);
  if (attribute_error != cudaSuccess) {
    throw std::runtime_error(std::string("ChainBUpdatePKernel: ") +
                             cudaGetErrorString(attribute_error));
  }
  ChainBUpdatePKernel<<<n_blocks, 1024, 8192>>>(ChainB_z, ChainB_z_num_alloc, ChainB_p_k,
                                                ChainB_p_k_num_alloc, beta, out_ChainB_p_kp1,
                                                out_ChainB_p_kp1_num_alloc, problem_size);
  // A launch that fails (e.g. from exceeding shared memory or register limits)
  // otherwise leaves its outputs untouched without any error.
  const cudaError_t launch_error = cudaGetLastError();
  if (launch_error != cudaSuccess) {
    throw std::runtime_error(std::string("ChainBUpdatePKernel launch failed: ") +
                             cudaGetErrorString(launch_error));
  }
}

}  // namespace caspar