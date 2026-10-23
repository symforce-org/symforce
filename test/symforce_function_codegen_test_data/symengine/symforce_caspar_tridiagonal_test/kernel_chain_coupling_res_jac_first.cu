#include <stdexcept>
#include <string>

#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include "kernel_chain_coupling_res_jac_first.h"
#include "memops.cuh"

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1) ChainCouplingResJacFirstKernel(
    double* a, unsigned int a_num_alloc, SharedIndex* a_indices, double* b,
    unsigned int b_num_alloc, SharedIndex* b_indices, double* weight, unsigned int weight_num_alloc,
    double* offset, unsigned int offset_num_alloc, double* out_res, unsigned int out_res_num_alloc,
    double* const out_rTr, double* out_a_jac, unsigned int out_a_jac_num_alloc,
    double* const out_a_njtr, unsigned int out_a_njtr_num_alloc, double* const out_a_precond_diag,
    unsigned int out_a_precond_diag_num_alloc, double* const out_a_precond_tril,
    unsigned int out_a_precond_tril_num_alloc, double* out_b_jac, unsigned int out_b_jac_num_alloc,
    double* const out_b_njtr, unsigned int out_b_njtr_num_alloc, double* const out_b_precond_diag,
    unsigned int out_b_precond_diag_num_alloc, double* const out_b_precond_tril,
    unsigned int out_b_precond_tril_num_alloc, size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  // Scratch data, then the shared indices of each shared accessor; dynamic, so
  // that it may exceed the 48 KB static shared memory limit (see the launch
  // below).
  extern __shared__ __align__(16) uint8_t inout_shared[];
  constexpr unsigned int a_indices_shared_offset = 16384;
  constexpr unsigned int b_indices_shared_offset = 24576;

  SharedIndex* const a_indices_loc =
      reinterpret_cast<SharedIndex*>(inout_shared + a_indices_shared_offset);
  a_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size ? a_indices[global_thread_idx]
                                        : SharedIndex{0xffffffff, 0xffff, 0xffff});
  SharedIndex* const b_indices_loc =
      reinterpret_cast<SharedIndex*>(inout_shared + b_indices_shared_offset);
  b_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size ? b_indices[global_thread_idx]
                                        : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ double out_rTr_local[1];

  double r0 = 0, r1 = 0, r2 = 0, r3 = 0, r4 = 0, r5 = 0, r6 = 0, r7 = 0, r8 = 0, r9 = 0, r10 = 0,
         r11 = 0, r12 = 0, r13 = 0;
  LoadShared<2, double, double>(a, 0 * a_num_alloc, a_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared, a_indices_loc[threadIdx.x].target, r0, r1);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(offset, 0 * offset_num_alloc, global_thread_idx, r2,
                                            r3);
    r4 = -1.00000000000000000e+00;
    r2 = fma(r2, r4, r0);
  };
  LoadShared<1, double, double>(b, 2 * b_num_alloc, b_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>((double*)inout_shared, b_indices_loc[threadIdx.x].target, r0);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(weight, 4 * weight_num_alloc, global_thread_idx, r5,
                                            r6);
  };
  LoadShared<2, double, double>(b, 0 * b_num_alloc, b_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared, b_indices_loc[threadIdx.x].target, r7, r8);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(weight, 2 * weight_num_alloc, global_thread_idx, r9,
                                            r10);
    r11 = fma(r8, r9, r0 * r5);
    ReadIdx2<1024, double, double, double2>(weight, 0 * weight_num_alloc, global_thread_idx, r12,
                                            r13);
    r11 = fma(r7, r12, r11);
    r2 = fma(r4, r11, r2);
    r3 = fma(r3, r4, r1);
    r8 = fma(r8, r10, r0 * r6);
    r8 = fma(r7, r13, r8);
    r3 = fma(r4, r8, r3);
    WriteIdx2<1024, double, double, double2>(out_res, 0 * out_res_num_alloc, global_thread_idx, r2,
                                             r3);
    r8 = fma(r2, r2, r3 * r3);
  };
  SumStore<double>(out_rTr_local, (double*)inout_shared, 0, global_thread_idx < problem_size, r8);
  if (global_thread_idx < problem_size) {
    r8 = r4 * r2;
    r7 = r4 * r3;
    WriteSum2<double, double>((double*)inout_shared, r8, r7);
  };
  FlushSumShared<2, double>(out_a_njtr, 0 * out_a_njtr_num_alloc, a_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r7 = 1.00000000000000000e+00;
    WriteSum2<double, double>((double*)inout_shared, r7, r7);
  };
  FlushSumShared<2, double>(out_a_precond_diag, 0 * out_a_precond_diag_num_alloc, a_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r7 = r12 * r4;
    r8 = r13 * r4;
    WriteIdx2<1024, double, double, double2>(out_b_jac, 0 * out_b_jac_num_alloc, global_thread_idx,
                                             r7, r8);
    r8 = r9 * r4;
    r7 = r10 * r4;
    WriteIdx2<1024, double, double, double2>(out_b_jac, 2 * out_b_jac_num_alloc, global_thread_idx,
                                             r8, r7);
    r7 = r5 * r4;
    r4 = r6 * r4;
    WriteIdx2<1024, double, double, double2>(out_b_jac, 4 * out_b_jac_num_alloc, global_thread_idx,
                                             r7, r4);
    r4 = fma(r12, r2, r13 * r3);
    r7 = fma(r9, r2, r10 * r3);
    WriteSum2<double, double>((double*)inout_shared, r4, r7);
  };
  FlushSumShared<2, double>(out_b_njtr, 0 * out_b_njtr_num_alloc, b_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r2 = fma(r5, r2, r6 * r3);
    WriteSum1<double, double>((double*)inout_shared, r2);
  };
  FlushSumShared<1, double>(out_b_njtr, 2 * out_b_njtr_num_alloc, b_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r2 = fma(r12, r12, r13 * r13);
    r3 = fma(r9, r9, r10 * r10);
    WriteSum2<double, double>((double*)inout_shared, r2, r3);
  };
  FlushSumShared<2, double>(out_b_precond_diag, 0 * out_b_precond_diag_num_alloc, b_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r3 = fma(r5, r5, r6 * r6);
    WriteSum1<double, double>((double*)inout_shared, r3);
  };
  FlushSumShared<1, double>(out_b_precond_diag, 2 * out_b_precond_diag_num_alloc, b_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r3 = fma(r13, r10, r12 * r9);
    r13 = fma(r13, r6, r12 * r5);
    WriteSum2<double, double>((double*)inout_shared, r3, r13);
  };
  FlushSumShared<2, double>(out_b_precond_tril, 0 * out_b_precond_tril_num_alloc, b_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r6 = fma(r10, r6, r9 * r5);
    WriteSum1<double, double>((double*)inout_shared, r6);
  };
  FlushSumShared<1, double>(out_b_precond_tril, 2 * out_b_precond_tril_num_alloc, b_indices_loc,
                            (double*)inout_shared);
  SumFlushFinal<double>(out_rTr_local, out_rTr, 1);
}

void ChainCouplingResJacFirst(
    double* a, unsigned int a_num_alloc, SharedIndex* a_indices, double* b,
    unsigned int b_num_alloc, SharedIndex* b_indices, double* weight, unsigned int weight_num_alloc,
    double* offset, unsigned int offset_num_alloc, double* out_res, unsigned int out_res_num_alloc,
    double* const out_rTr, double* out_a_jac, unsigned int out_a_jac_num_alloc,
    double* const out_a_njtr, unsigned int out_a_njtr_num_alloc, double* const out_a_precond_diag,
    unsigned int out_a_precond_diag_num_alloc, double* const out_a_precond_tril,
    unsigned int out_a_precond_tril_num_alloc, double* out_b_jac, unsigned int out_b_jac_num_alloc,
    double* const out_b_njtr, unsigned int out_b_njtr_num_alloc, double* const out_b_precond_diag,
    unsigned int out_b_precond_diag_num_alloc, double* const out_b_precond_tril,
    unsigned int out_b_precond_tril_num_alloc, size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  // Kernels using more than 48 KB of shared memory per block must opt in.
  static const cudaError_t attribute_error = cudaFuncSetAttribute(
      ChainCouplingResJacFirstKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 32768);
  if (attribute_error != cudaSuccess) {
    throw std::runtime_error(std::string("ChainCouplingResJacFirstKernel: ") +
                             cudaGetErrorString(attribute_error));
  }
  ChainCouplingResJacFirstKernel<<<n_blocks, 1024, 32768>>>(
      a, a_num_alloc, a_indices, b, b_num_alloc, b_indices, weight, weight_num_alloc, offset,
      offset_num_alloc, out_res, out_res_num_alloc, out_rTr, out_a_jac, out_a_jac_num_alloc,
      out_a_njtr, out_a_njtr_num_alloc, out_a_precond_diag, out_a_precond_diag_num_alloc,
      out_a_precond_tril, out_a_precond_tril_num_alloc, out_b_jac, out_b_jac_num_alloc, out_b_njtr,
      out_b_njtr_num_alloc, out_b_precond_diag, out_b_precond_diag_num_alloc, out_b_precond_tril,
      out_b_precond_tril_num_alloc, problem_size);
  // A launch that fails (e.g. from exceeding shared memory or register limits)
  // otherwise leaves its outputs untouched without any error.
  const cudaError_t launch_error = cudaGetLastError();
  if (launch_error != cudaSuccess) {
    throw std::runtime_error(std::string("ChainCouplingResJacFirstKernel launch failed: ") +
                             cudaGetErrorString(launch_error));
  }
}

}  // namespace caspar