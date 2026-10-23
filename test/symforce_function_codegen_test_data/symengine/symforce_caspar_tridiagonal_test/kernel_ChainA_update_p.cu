#include <stdexcept>
#include <string>

#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include "kernel_ChainA_update_p.h"
#include "memops.cuh"

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    ChainAUpdatePKernel(double* ChainA_z, unsigned int ChainA_z_num_alloc, double* ChainA_p_k,
                        unsigned int ChainA_p_k_num_alloc, const double* const beta,
                        double* out_ChainA_p_kp1, unsigned int out_ChainA_p_kp1_num_alloc,
                        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  // Scratch data, then the shared indices of each shared accessor; dynamic, so
  // that it may exceed the 48 KB static shared memory limit (see the launch
  // below).
  extern __shared__ __align__(16) uint8_t inout_shared[];

  double r0 = 0, r1 = 0, r2 = 0, r3 = 0, r4 = 0;

  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(ChainA_p_k, 0 * ChainA_p_k_num_alloc, global_thread_idx,
                                            r0, r1);
    ReadIdx2<1024, double, double, double2>(ChainA_z, 0 * ChainA_z_num_alloc, global_thread_idx, r2,
                                            r3);
  };
  LoadUnique<1, double, double>(beta, 0, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>((double*)inout_shared, 0, r4);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r0 = fma(r0, r4, r2);
    r4 = fma(r1, r4, r3);
    WriteIdx2<1024, double, double, double2>(out_ChainA_p_kp1, 0 * out_ChainA_p_kp1_num_alloc,
                                             global_thread_idx, r0, r4);
  };
}

void ChainAUpdateP(double* ChainA_z, unsigned int ChainA_z_num_alloc, double* ChainA_p_k,
                   unsigned int ChainA_p_k_num_alloc, const double* const beta,
                   double* out_ChainA_p_kp1, unsigned int out_ChainA_p_kp1_num_alloc,
                   size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  // Kernels using more than 48 KB of shared memory per block must opt in.
  static const cudaError_t attribute_error =
      cudaFuncSetAttribute(ChainAUpdatePKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 8192);
  if (attribute_error != cudaSuccess) {
    throw std::runtime_error(std::string("ChainAUpdatePKernel: ") +
                             cudaGetErrorString(attribute_error));
  }
  ChainAUpdatePKernel<<<n_blocks, 1024, 8192>>>(ChainA_z, ChainA_z_num_alloc, ChainA_p_k,
                                                ChainA_p_k_num_alloc, beta, out_ChainA_p_kp1,
                                                out_ChainA_p_kp1_num_alloc, problem_size);
  // A launch that fails (e.g. from exceeding shared memory or register limits)
  // otherwise leaves its outputs untouched without any error.
  const cudaError_t launch_error = cudaGetLastError();
  if (launch_error != cudaSuccess) {
    throw std::runtime_error(std::string("ChainAUpdatePKernel launch failed: ") +
                             cudaGetErrorString(launch_error));
  }
}

}  // namespace caspar