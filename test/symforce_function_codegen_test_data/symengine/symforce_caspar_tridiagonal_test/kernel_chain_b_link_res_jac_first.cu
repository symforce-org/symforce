#include <stdexcept>
#include <string>

#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include "kernel_chain_b_link_res_jac_first.h"
#include "memops.cuh"

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1) ChainBLinkResJacFirstKernel(
    double* const b, unsigned int b_num_alloc, double* weight, unsigned int weight_num_alloc,
    double* offset, unsigned int offset_num_alloc, double* out_res, unsigned int out_res_num_alloc,
    double* const out_rTr, double* out_b_tridig_U, unsigned int out_b_tridig_U_num_alloc,
    double* const out_b_njtr, unsigned int out_b_njtr_num_alloc, double* const out_b_precond_diag,
    unsigned int out_b_precond_diag_num_alloc, double* const out_b_precond_tril,
    unsigned int out_b_precond_tril_num_alloc, size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  // Scratch data, then the shared indices of each shared accessor; dynamic, so
  // that it may exceed the 48 KB static shared memory limit (see the launch
  // below).
  extern __shared__ __align__(16) uint8_t inout_shared[];

  __shared__ double out_rTr_local[1];

  double r0 = 0, r1 = 0, r2 = 0, r3 = 0, r4 = 0, r5 = 0, r6 = 0, r7 = 0, r8 = 0, r9 = 0, r10 = 0,
         r11 = 0, r12 = 0, r13 = 0, r14 = 0, r15 = 0, r16 = 0, r17 = 0, r18 = 0, r19 = 0, r20 = 0,
         r21 = 0, r22 = 0, r23 = 0, r24 = 0, r25 = 0, r26 = 0;

  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(offset, 0 * offset_num_alloc, global_thread_idx, r0,
                                            r1);
    r2 = -1.00000000000000000e+00;
    ReadIdx2<1024, double, double, double2>(weight, 6 * weight_num_alloc, global_thread_idx, r3,
                                            r4);
  };
  ReadAndShuffle1<1024, double, double, double>((double*)inout_shared, b, 2 * b_num_alloc,
                                                global_thread_idx, threadIdx.x == 1023,
                                                global_thread_idx <= problem_size, r5, r6);
  if (global_thread_idx < problem_size) {
    r5 = fma(r5, r2, r6);
    r0 = fma(r3, r5, r0 * r2);
    ReadIdx2<1024, double, double, double2>(weight, 0 * weight_num_alloc, global_thread_idx, r6,
                                            r7);
  };
  ReadAndShuffle2<1024, double, double, double2>(
      (double*)inout_shared, b, 0 * b_num_alloc, global_thread_idx, threadIdx.x == 1023,
      global_thread_idx <= problem_size, r8, r9, r10, r11);
  if (global_thread_idx < problem_size) {
    r8 = fma(r8, r2, r10);
    ReadIdx2<1024, double, double, double2>(weight, 2 * weight_num_alloc, global_thread_idx, r10,
                                            r12);
    r9 = fma(r9, r2, r11);
    r0 = fma(r6, r8, r0);
    r0 = fma(r12, r9, r0);
    r1 = fma(r4, r5, r1 * r2);
    ReadIdx2<1024, double, double, double2>(weight, 4 * weight_num_alloc, global_thread_idx, r11,
                                            r13);
    r1 = fma(r7, r8, r1);
    r1 = fma(r11, r9, r1);
    WriteIdx2<1024, double, double, double2>(out_res, 0 * out_res_num_alloc, global_thread_idx, r0,
                                             r1);
    ReadIdx1<1024, double, double, double>(offset, 2 * offset_num_alloc, global_thread_idx, r14);
    ReadIdx1<1024, double, double, double>(weight, 8 * weight_num_alloc, global_thread_idx, r15);
    r5 = fma(r15, r5, r14 * r2);
    r5 = fma(r10, r8, r5);
    r5 = fma(r13, r9, r5);
    WriteIdx1<1024, double, double, double>(out_res, 2 * out_res_num_alloc, global_thread_idx, r5);
    r9 = fma(r0, r0, r5 * r5);
    r9 = fma(r1, r1, r9);
  };
  SumStore<double>(out_rTr_local, (double*)inout_shared, 0, global_thread_idx < problem_size, r9);
  if (global_thread_idx < problem_size) {
    r9 = r7 * r2;
    r8 = r10 * r10;
    r14 = fma(r2, r8, r7 * r9);
    r16 = r6 * r6;
    r14 = fma(r2, r16, r14);
    r17 = r6 * r12;
    r18 = r10 * r13;
    r18 = fma(r2, r18, r2 * r17);
    r18 = fma(r11, r9, r18);
    WriteIdx2<1024, double, double, double2>(out_b_tridig_U, 0 * out_b_tridig_U_num_alloc,
                                             global_thread_idx, r14, r18);
    r14 = r6 * r3;
    r17 = r10 * r15;
    r17 = fma(r2, r17, r2 * r14);
    r17 = fma(r4, r9, r17);
    WriteIdx2<1024, double, double, double2>(out_b_tridig_U, 2 * out_b_tridig_U_num_alloc,
                                             global_thread_idx, r17, r18);
    r18 = r12 * r3;
    r14 = r13 * r15;
    r14 = fma(r2, r14, r2 * r18);
    r18 = r11 * r4;
    r14 = fma(r2, r18, r14);
    r18 = r11 * r11;
    r19 = r13 * r13;
    r20 = fma(r2, r19, r2 * r18);
    r21 = r12 * r12;
    r20 = fma(r2, r21, r20);
    WriteIdx2<1024, double, double, double2>(out_b_tridig_U, 4 * out_b_tridig_U_num_alloc,
                                             global_thread_idx, r20, r14);
    WriteIdx2<1024, double, double, double2>(out_b_tridig_U, 6 * out_b_tridig_U_num_alloc,
                                             global_thread_idx, r17, r14);
    r14 = r4 * r4;
    r17 = r15 * r15;
    r20 = fma(r2, r17, r2 * r14);
    r22 = r3 * r3;
    r20 = fma(r2, r22, r20);
    WriteIdx1<1024, double, double, double>(out_b_tridig_U, 8 * out_b_tridig_U_num_alloc,
                                            global_thread_idx, r20);
    r20 = fma(r6, r0, r10 * r5);
    r20 = fma(r7, r1, r20);
    r23 = fma(r12, r0, r13 * r5);
    r23 = fma(r11, r1, r23);
    r24 = r13 * r2;
    r25 = r12 * r2;
    r25 = fma(r0, r25, r5 * r24);
    r24 = r11 * r2;
    r25 = fma(r1, r24, r25);
    r24 = r10 * r2;
    r26 = r6 * r2;
    r26 = fma(r0, r26, r5 * r24);
    r26 = fma(r1, r9, r26);
  };
  ShuffleAndWrite2<true, 1024, double, double, double2>((double*)inout_shared, out_b_njtr,
                                                        0 * out_b_njtr_num_alloc, global_thread_idx,
                                                        problem_size, r20, r23, r26, r25);
  if (global_thread_idx < problem_size) {
    r25 = fma(r3, r0, r15 * r5);
    r25 = fma(r4, r1, r25);
    r26 = r15 * r2;
    r23 = r3 * r2;
    r23 = fma(r0, r23, r5 * r26);
    r26 = r4 * r2;
    r23 = fma(r1, r26, r23);
  };
  ShuffleAndWrite1<true, 1024, double, double, double>((double*)inout_shared, out_b_njtr,
                                                       2 * out_b_njtr_num_alloc, global_thread_idx,
                                                       problem_size, r25, r23);
  if (global_thread_idx < problem_size) {
    r8 = fma(r7, r7, r8);
    r8 = r8 + r16;
    r19 = r18 + r19;
    r19 = r19 + r21;
  };
  ShuffleAndWrite2<true, 1024, double, double, double2>(
      (double*)inout_shared, out_b_precond_diag, 0 * out_b_precond_diag_num_alloc,
      global_thread_idx, problem_size, r8, r19, r8, r19);
  if (global_thread_idx < problem_size) {
    r17 = r14 + r17;
    r17 = r17 + r22;
  };
  ShuffleAndWrite1<true, 1024, double, double, double>((double*)inout_shared, out_b_precond_diag,
                                                       2 * out_b_precond_diag_num_alloc,
                                                       global_thread_idx, problem_size, r17, r17);
  if (global_thread_idx < problem_size) {
    r17 = fma(r10, r13, r6 * r12);
    r17 = fma(r7, r11, r17);
    r22 = fma(r10, r15, r6 * r3);
    r22 = fma(r7, r4, r22);
  };
  ShuffleAndWrite2<true, 1024, double, double, double2>(
      (double*)inout_shared, out_b_precond_tril, 0 * out_b_precond_tril_num_alloc,
      global_thread_idx, problem_size, r17, r22, r17, r22);
  if (global_thread_idx < problem_size) {
    r22 = fma(r13, r15, r12 * r3);
    r22 = fma(r11, r4, r22);
  };
  ShuffleAndWrite1<true, 1024, double, double, double>((double*)inout_shared, out_b_precond_tril,
                                                       2 * out_b_precond_tril_num_alloc,
                                                       global_thread_idx, problem_size, r22, r22);
  SumFlushFinal<double>(out_rTr_local, out_rTr, 1);
}

void ChainBLinkResJacFirst(
    double* const b, unsigned int b_num_alloc, double* weight, unsigned int weight_num_alloc,
    double* offset, unsigned int offset_num_alloc, double* out_res, unsigned int out_res_num_alloc,
    double* const out_rTr, double* out_b_tridig_U, unsigned int out_b_tridig_U_num_alloc,
    double* const out_b_njtr, unsigned int out_b_njtr_num_alloc, double* const out_b_precond_diag,
    unsigned int out_b_precond_diag_num_alloc, double* const out_b_precond_tril,
    unsigned int out_b_precond_tril_num_alloc, size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  // Kernels using more than 48 KB of shared memory per block must opt in.
  static const cudaError_t attribute_error = cudaFuncSetAttribute(
      ChainBLinkResJacFirstKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 16400);
  if (attribute_error != cudaSuccess) {
    throw std::runtime_error(std::string("ChainBLinkResJacFirstKernel: ") +
                             cudaGetErrorString(attribute_error));
  }
  ChainBLinkResJacFirstKernel<<<n_blocks, 1024, 16400>>>(
      b, b_num_alloc, weight, weight_num_alloc, offset, offset_num_alloc, out_res,
      out_res_num_alloc, out_rTr, out_b_tridig_U, out_b_tridig_U_num_alloc, out_b_njtr,
      out_b_njtr_num_alloc, out_b_precond_diag, out_b_precond_diag_num_alloc, out_b_precond_tril,
      out_b_precond_tril_num_alloc, problem_size);
  // A launch that fails (e.g. from exceeding shared memory or register limits)
  // otherwise leaves its outputs untouched without any error.
  const cudaError_t launch_error = cudaGetLastError();
  if (launch_error != cudaSuccess) {
    throw std::runtime_error(std::string("ChainBLinkResJacFirstKernel launch failed: ") +
                             cudaGetErrorString(launch_error));
  }
}

}  // namespace caspar