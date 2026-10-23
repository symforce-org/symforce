#include <stdexcept>
#include <string>

#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include "kernel_chain_a_link_res_jac_first.h"
#include "memops.cuh"

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1) ChainALinkResJacFirstKernel(
    double* const a, unsigned int a_num_alloc, double* weight, unsigned int weight_num_alloc,
    double* offset, unsigned int offset_num_alloc, double* out_res, unsigned int out_res_num_alloc,
    double* const out_rTr, double* out_a_tridig_U, unsigned int out_a_tridig_U_num_alloc,
    double* const out_a_njtr, unsigned int out_a_njtr_num_alloc, double* const out_a_precond_diag,
    unsigned int out_a_precond_diag_num_alloc, double* const out_a_precond_tril,
    unsigned int out_a_precond_tril_num_alloc, size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  // Scratch data, then the shared indices of each shared accessor; dynamic, so
  // that it may exceed the 48 KB static shared memory limit (see the launch
  // below).
  extern __shared__ __align__(16) uint8_t inout_shared[];

  __shared__ double out_rTr_local[1];

  double r0 = 0, r1 = 0, r2 = 0, r3 = 0, r4 = 0, r5 = 0, r6 = 0, r7 = 0, r8 = 0, r9 = 0, r10 = 0,
         r11 = 0, r12 = 0, r13 = 0, r14 = 0;

  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(offset, 0 * offset_num_alloc, global_thread_idx, r0,
                                            r1);
    r2 = -1.00000000000000000e+00;
    ReadIdx2<1024, double, double, double2>(weight, 0 * weight_num_alloc, global_thread_idx, r3,
                                            r4);
  };
  ReadAndShuffle2<1024, double, double, double2>((double*)inout_shared, a, 0 * a_num_alloc,
                                                 global_thread_idx, threadIdx.x == 1023,
                                                 global_thread_idx <= problem_size, r5, r6, r7, r8);
  if (global_thread_idx < problem_size) {
    r5 = fma(r5, r2, r7);
    r0 = fma(r3, r5, r0 * r2);
    ReadIdx2<1024, double, double, double2>(weight, 2 * weight_num_alloc, global_thread_idx, r7,
                                            r9);
    r6 = fma(r6, r2, r8);
    r0 = fma(r7, r6, r0);
    r5 = fma(r4, r5, r1 * r2);
    r5 = fma(r9, r6, r5);
    WriteIdx2<1024, double, double, double2>(out_res, 0 * out_res_num_alloc, global_thread_idx, r0,
                                             r5);
    r6 = fma(r0, r0, r5 * r5);
  };
  SumStore<double>(out_rTr_local, (double*)inout_shared, 0, global_thread_idx < problem_size, r6);
  if (global_thread_idx < problem_size) {
    r6 = r4 * r2;
    r1 = r3 * r3;
    r8 = fma(r2, r1, r4 * r6);
    r10 = r3 * r7;
    r10 = fma(r9, r6, r2 * r10);
    WriteIdx2<1024, double, double, double2>(out_a_tridig_U, 0 * out_a_tridig_U_num_alloc,
                                             global_thread_idx, r8, r10);
    r8 = r9 * r9;
    r11 = r7 * r7;
    r12 = fma(r2, r11, r2 * r8);
    WriteIdx2<1024, double, double, double2>(out_a_tridig_U, 2 * out_a_tridig_U_num_alloc,
                                             global_thread_idx, r10, r12);
    r12 = fma(r3, r0, r4 * r5);
    r10 = fma(r7, r0, r9 * r5);
    r13 = r9 * r2;
    r14 = r7 * r2;
    r14 = fma(r0, r14, r5 * r13);
    r13 = r3 * r2;
    r6 = fma(r5, r6, r0 * r13);
  };
  ShuffleAndWrite2<true, 1024, double, double, double2>((double*)inout_shared, out_a_njtr,
                                                        0 * out_a_njtr_num_alloc, global_thread_idx,
                                                        problem_size, r12, r10, r6, r14);
  if (global_thread_idx < problem_size) {
    r1 = fma(r4, r4, r1);
    r11 = r8 + r11;
  };
  ShuffleAndWrite2<true, 1024, double, double, double2>(
      (double*)inout_shared, out_a_precond_diag, 0 * out_a_precond_diag_num_alloc,
      global_thread_idx, problem_size, r1, r11, r1, r11);
  if (global_thread_idx < problem_size) {
    r4 = fma(r4, r9, r3 * r7);
  };
  ShuffleAndWrite1<true, 1024, double, double, double>((double*)inout_shared, out_a_precond_tril,
                                                       0 * out_a_precond_tril_num_alloc,
                                                       global_thread_idx, problem_size, r4, r4);
  SumFlushFinal<double>(out_rTr_local, out_rTr, 1);
}

void ChainALinkResJacFirst(
    double* const a, unsigned int a_num_alloc, double* weight, unsigned int weight_num_alloc,
    double* offset, unsigned int offset_num_alloc, double* out_res, unsigned int out_res_num_alloc,
    double* const out_rTr, double* out_a_tridig_U, unsigned int out_a_tridig_U_num_alloc,
    double* const out_a_njtr, unsigned int out_a_njtr_num_alloc, double* const out_a_precond_diag,
    unsigned int out_a_precond_diag_num_alloc, double* const out_a_precond_tril,
    unsigned int out_a_precond_tril_num_alloc, size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  // Kernels using more than 48 KB of shared memory per block must opt in.
  static const cudaError_t attribute_error = cudaFuncSetAttribute(
      ChainALinkResJacFirstKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 16400);
  if (attribute_error != cudaSuccess) {
    throw std::runtime_error(std::string("ChainALinkResJacFirstKernel: ") +
                             cudaGetErrorString(attribute_error));
  }
  ChainALinkResJacFirstKernel<<<n_blocks, 1024, 16400>>>(
      a, a_num_alloc, weight, weight_num_alloc, offset, offset_num_alloc, out_res,
      out_res_num_alloc, out_rTr, out_a_tridig_U, out_a_tridig_U_num_alloc, out_a_njtr,
      out_a_njtr_num_alloc, out_a_precond_diag, out_a_precond_diag_num_alloc, out_a_precond_tril,
      out_a_precond_tril_num_alloc, problem_size);
  // A launch that fails (e.g. from exceeding shared memory or register limits)
  // otherwise leaves its outputs untouched without any error.
  const cudaError_t launch_error = cudaGetLastError();
  if (launch_error != cudaSuccess) {
    throw std::runtime_error(std::string("ChainALinkResJacFirstKernel launch failed: ") +
                             cudaGetErrorString(launch_error));
  }
}

}  // namespace caspar