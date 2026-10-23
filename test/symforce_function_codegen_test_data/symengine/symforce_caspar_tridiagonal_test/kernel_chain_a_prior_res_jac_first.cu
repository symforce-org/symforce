#include <stdexcept>
#include <string>

#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include "kernel_chain_a_prior_res_jac_first.h"
#include "memops.cuh"

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    ChainAPriorResJacFirstKernel(double* a, unsigned int a_num_alloc, SharedIndex* a_indices,
                                 double* target, unsigned int target_num_alloc, double* out_res,
                                 unsigned int out_res_num_alloc, double* const out_rTr,
                                 double* const out_a_njtr, unsigned int out_a_njtr_num_alloc,
                                 double* const out_a_precond_diag,
                                 unsigned int out_a_precond_diag_num_alloc,
                                 double* const out_a_precond_tril,
                                 unsigned int out_a_precond_tril_num_alloc, size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  // Scratch data, then the shared indices of each shared accessor; dynamic, so
  // that it may exceed the 48 KB static shared memory limit (see the launch
  // below).
  extern __shared__ __align__(16) uint8_t inout_shared[];
  constexpr unsigned int a_indices_shared_offset = 16384;

  SharedIndex* const a_indices_loc =
      reinterpret_cast<SharedIndex*>(inout_shared + a_indices_shared_offset);
  a_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size ? a_indices[global_thread_idx]
                                        : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ double out_rTr_local[1];

  double r0 = 0, r1 = 0, r2 = 0, r3 = 0, r4 = 0;
  LoadShared<2, double, double>(a, 0 * a_num_alloc, a_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared, a_indices_loc[threadIdx.x].target, r0, r1);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(target, 0 * target_num_alloc, global_thread_idx, r2,
                                            r3);
    r4 = -1.00000000000000000e+00;
    r2 = fma(r2, r4, r0);
    r3 = fma(r3, r4, r1);
    WriteIdx2<1024, double, double, double2>(out_res, 0 * out_res_num_alloc, global_thread_idx, r2,
                                             r3);
    r1 = fma(r3, r3, r2 * r2);
  };
  SumStore<double>(out_rTr_local, (double*)inout_shared, 0, global_thread_idx < problem_size, r1);
  if (global_thread_idx < problem_size) {
    r2 = r4 * r2;
    r3 = r4 * r3;
    WriteSum2<double, double>((double*)inout_shared, r2, r3);
  };
  FlushSumShared<2, double>(out_a_njtr, 0 * out_a_njtr_num_alloc, a_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r3 = 1.00000000000000000e+00;
    WriteSum2<double, double>((double*)inout_shared, r3, r3);
  };
  FlushSumShared<2, double>(out_a_precond_diag, 0 * out_a_precond_diag_num_alloc, a_indices_loc,
                            (double*)inout_shared);
  SumFlushFinal<double>(out_rTr_local, out_rTr, 1);
}

void ChainAPriorResJacFirst(double* a, unsigned int a_num_alloc, SharedIndex* a_indices,
                            double* target, unsigned int target_num_alloc, double* out_res,
                            unsigned int out_res_num_alloc, double* const out_rTr,
                            double* const out_a_njtr, unsigned int out_a_njtr_num_alloc,
                            double* const out_a_precond_diag,
                            unsigned int out_a_precond_diag_num_alloc,
                            double* const out_a_precond_tril,
                            unsigned int out_a_precond_tril_num_alloc, size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  // Kernels using more than 48 KB of shared memory per block must opt in.
  static const cudaError_t attribute_error = cudaFuncSetAttribute(
      ChainAPriorResJacFirstKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 24576);
  if (attribute_error != cudaSuccess) {
    throw std::runtime_error(std::string("ChainAPriorResJacFirstKernel: ") +
                             cudaGetErrorString(attribute_error));
  }
  ChainAPriorResJacFirstKernel<<<n_blocks, 1024, 24576>>>(
      a, a_num_alloc, a_indices, target, target_num_alloc, out_res, out_res_num_alloc, out_rTr,
      out_a_njtr, out_a_njtr_num_alloc, out_a_precond_diag, out_a_precond_diag_num_alloc,
      out_a_precond_tril, out_a_precond_tril_num_alloc, problem_size);
  // A launch that fails (e.g. from exceeding shared memory or register limits)
  // otherwise leaves its outputs untouched without any error.
  const cudaError_t launch_error = cudaGetLastError();
  if (launch_error != cudaSuccess) {
    throw std::runtime_error(std::string("ChainAPriorResJacFirstKernel launch failed: ") +
                             cudaGetErrorString(launch_error));
  }
}

}  // namespace caspar