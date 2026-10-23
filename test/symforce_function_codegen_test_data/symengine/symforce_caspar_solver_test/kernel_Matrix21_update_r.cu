#include <stdexcept>
#include <string>

#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include "kernel_Matrix21_update_r.h"
#include "memops.cuh"

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    Matrix21UpdateRKernel(double* Matrix21_r_k, unsigned int Matrix21_r_k_num_alloc,
                          double* Matrix21_w, unsigned int Matrix21_w_num_alloc,
                          const double* const negalpha, double* out_Matrix21_r_kp1,
                          unsigned int out_Matrix21_r_kp1_num_alloc,
                          double* const out_Matrix21_r_kp1_norm2_tot, size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  // Scratch data, then the shared indices of each shared accessor; dynamic, so
  // that it may exceed the 48 KB static shared memory limit (see the launch
  // below).
  extern __shared__ __align__(16) uint8_t inout_shared[];

  __shared__ double out_Matrix21_r_kp1_norm2_tot_local[1];

  double r0 = 0, r1 = 0, r2 = 0, r3 = 0, r4 = 0;

  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(Matrix21_r_k, 0 * Matrix21_r_k_num_alloc,
                                            global_thread_idx, r0, r1);
    ReadIdx2<1024, double, double, double2>(Matrix21_w, 0 * Matrix21_w_num_alloc, global_thread_idx,
                                            r2, r3);
  };
  LoadUnique<1, double, double>(negalpha, 0, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>((double*)inout_shared, 0, r4);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r2 = fma(r2, r4, r0);
    r4 = fma(r3, r4, r1);
    WriteIdx2<1024, double, double, double2>(out_Matrix21_r_kp1, 0 * out_Matrix21_r_kp1_num_alloc,
                                             global_thread_idx, r2, r4);
    r2 = fma(r2, r2, r4 * r4);
  };
  SumStore<double>(out_Matrix21_r_kp1_norm2_tot_local, (double*)inout_shared, 0,
                   global_thread_idx < problem_size, r2);
  SumFlushFinal<double>(out_Matrix21_r_kp1_norm2_tot_local, out_Matrix21_r_kp1_norm2_tot, 1);
}

void Matrix21UpdateR(double* Matrix21_r_k, unsigned int Matrix21_r_k_num_alloc, double* Matrix21_w,
                     unsigned int Matrix21_w_num_alloc, const double* const negalpha,
                     double* out_Matrix21_r_kp1, unsigned int out_Matrix21_r_kp1_num_alloc,
                     double* const out_Matrix21_r_kp1_norm2_tot, size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  // Kernels using more than 48 KB of shared memory per block must opt in.
  static const cudaError_t attribute_error = cudaFuncSetAttribute(
      Matrix21UpdateRKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 8192);
  if (attribute_error != cudaSuccess) {
    throw std::runtime_error(std::string("Matrix21UpdateRKernel: ") +
                             cudaGetErrorString(attribute_error));
  }
  Matrix21UpdateRKernel<<<n_blocks, 1024, 8192>>>(
      Matrix21_r_k, Matrix21_r_k_num_alloc, Matrix21_w, Matrix21_w_num_alloc, negalpha,
      out_Matrix21_r_kp1, out_Matrix21_r_kp1_num_alloc, out_Matrix21_r_kp1_norm2_tot, problem_size);
  // A launch that fails (e.g. from exceeding shared memory or register limits)
  // otherwise leaves its outputs untouched without any error.
  const cudaError_t launch_error = cudaGetLastError();
  if (launch_error != cudaSuccess) {
    throw std::runtime_error(std::string("Matrix21UpdateRKernel launch failed: ") +
                             cudaGetErrorString(launch_error));
  }
}

}  // namespace caspar