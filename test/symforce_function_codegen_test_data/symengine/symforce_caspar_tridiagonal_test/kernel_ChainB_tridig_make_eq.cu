#include <stdexcept>
#include <string>

#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include "kernel_ChainB_tridig_make_eq.h"
#include "memops.cuh"

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    ChainBTridigMakeEqKernel(double* precond_diag, unsigned int precond_diag_num_alloc,
                             double* precond_tril, unsigned int precond_tril_num_alloc,
                             double* const tridig_U, unsigned int tridig_U_num_alloc,
                             int tridig_U_offset, int tridig_U_stride, double* njtr,
                             unsigned int njtr_num_alloc, const double* const diag, double* eq,
                             unsigned int eq_num_alloc, size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  // Scratch data, then the shared indices of each shared accessor; dynamic, so
  // that it may exceed the 48 KB static shared memory limit (see the launch
  // below).
  extern __shared__ __align__(16) uint8_t inout_shared[];

  double r0 = 0, r1 = 0, r2 = 0, r3 = 0, r4 = 0, r5 = 0, r6 = 0, r7 = 0, r8 = 0, r9 = 0, r10 = 0,
         r11 = 0, r12 = 0, r13 = 0, r14 = 0, r15 = 0, r16 = 0, r17 = 0;
  ReadAndShuffleWithDefault2<1024, double, double, double2>(
      (double*)inout_shared, tridig_U, 0 * tridig_U_num_alloc,
      global_thread_idx * tridig_U_stride + tridig_U_offset, threadIdx.x == 1023,
      global_thread_idx <= problem_size, tridig_U_num_alloc, tridig_U_stride, r0, r1, r2, r3,
      0.000000f, 0.000000f);
  ReadAndShuffleWithDefault2<1024, double, double, double2>(
      (double*)inout_shared, tridig_U, 2 * tridig_U_num_alloc,
      global_thread_idx * tridig_U_stride + tridig_U_offset, threadIdx.x == 1023,
      global_thread_idx <= problem_size, tridig_U_num_alloc, tridig_U_stride, r4, r5, r6, r7,
      0.000000f, 0.000000f);
  if (global_thread_idx < problem_size) {
    WriteIdx2<1024, double, double, double2>(eq, 0 * eq_num_alloc, global_thread_idx, r0, r5);
  };
  ReadAndShuffleWithDefault2<1024, double, double, double2>(
      (double*)inout_shared, tridig_U, 6 * tridig_U_num_alloc,
      global_thread_idx * tridig_U_stride + tridig_U_offset, threadIdx.x == 1023,
      global_thread_idx <= problem_size, tridig_U_num_alloc, tridig_U_stride, r5, r0, r8, r9,
      0.000000f, 0.000000f);
  if (global_thread_idx < problem_size) {
    WriteIdx2<1024, double, double, double2>(eq, 2 * eq_num_alloc, global_thread_idx, r5, r1);
  };
  ReadAndShuffleWithDefault2<1024, double, double, double2>(
      (double*)inout_shared, tridig_U, 4 * tridig_U_num_alloc,
      global_thread_idx * tridig_U_stride + tridig_U_offset, threadIdx.x == 1023,
      global_thread_idx <= problem_size, tridig_U_num_alloc, tridig_U_stride, r1, r5, r10, r11,
      0.000000f, 0.000000f);
  if (global_thread_idx < problem_size) {
    WriteIdx2<1024, double, double, double2>(eq, 4 * eq_num_alloc, global_thread_idx, r1, r0);
    WriteIdx2<1024, double, double, double2>(eq, 6 * eq_num_alloc, global_thread_idx, r4, r5);
  };
  ReadAndShuffleWithDefault1<1024, double, double, double>(
      (double*)inout_shared, tridig_U, 8 * tridig_U_num_alloc,
      global_thread_idx * tridig_U_stride + tridig_U_offset, threadIdx.x == 1023,
      global_thread_idx <= problem_size, tridig_U_num_alloc, tridig_U_stride, r5, r4, 0.000000f);
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(precond_diag, 0 * precond_diag_num_alloc,
                                            global_thread_idx, r0, r1);
  };
  LoadUnique<1, double, double>(diag, 0, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>((double*)inout_shared, 0, r12);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r13 = 1.00000000000000000e+00;
    r14 = r12 + r13;
    r15 = 2.22044604925031308e-15;
    r15 = r12 * r15;
    r0 = fma(r0, r14, r15);
    WriteIdx2<1024, double, double, double2>(eq, 8 * eq_num_alloc, global_thread_idx, r5, r0);
    ReadIdx2<1024, double, double, double2>(precond_tril, 0 * precond_tril_num_alloc,
                                            global_thread_idx, r5, r12);
    WriteIdx2<1024, double, double, double2>(eq, 10 * eq_num_alloc, global_thread_idx, r5, r12);
    r1 = fma(r1, r14, r15);
    WriteIdx2<1024, double, double, double2>(eq, 12 * eq_num_alloc, global_thread_idx, r5, r1);
    ReadIdx1<1024, double, double, double>(precond_tril, 2 * precond_tril_num_alloc,
                                           global_thread_idx, r16);
    WriteIdx2<1024, double, double, double2>(eq, 14 * eq_num_alloc, global_thread_idx, r16, r12);
    ReadIdx1<1024, double, double, double>(precond_diag, 2 * precond_diag_num_alloc,
                                           global_thread_idx, r17);
    r14 = r17 * r14;
    r17 = r15 + r14;
    WriteIdx2<1024, double, double, double2>(eq, 16 * eq_num_alloc, global_thread_idx, r16, r17);
    WriteIdx2<1024, double, double, double2>(eq, 18 * eq_num_alloc, global_thread_idx, r2, r3);
    WriteIdx2<1024, double, double, double2>(eq, 20 * eq_num_alloc, global_thread_idx, r6, r7);
    WriteIdx2<1024, double, double, double2>(eq, 22 * eq_num_alloc, global_thread_idx, r10, r11);
    WriteIdx2<1024, double, double, double2>(eq, 24 * eq_num_alloc, global_thread_idx, r8, r9);
    ReadIdx2<1024, double, double, double2>(njtr, 0 * njtr_num_alloc, global_thread_idx, r9, r8);
    WriteIdx2<1024, double, double, double2>(eq, 26 * eq_num_alloc, global_thread_idx, r4, r9);
    ReadIdx1<1024, double, double, double>(njtr, 2 * njtr_num_alloc, global_thread_idx, r9);
    WriteIdx2<1024, double, double, double2>(eq, 28 * eq_num_alloc, global_thread_idx, r8, r9);
    r9 = 1.0 / r0;
    r8 = r5 * r9;
    WriteIdx2<1024, double, double, double2>(eq, 30 * eq_num_alloc, global_thread_idx, r13, r8);
    r4 = r12 * r9;
    WriteIdx2<1024, double, double, double2>(eq, 32 * eq_num_alloc, global_thread_idx, r13, r4);
    r4 = -1.00000000000000000e+00;
    r11 = r5 * r4;
    r11 = fma(r8, r11, r1);
    r1 = 1.0 / r11;
    r10 = r12 * r4;
    r8 = fma(r8, r10, r16);
    r1 = r1 * r8;
    WriteIdx2<1024, double, double, double2>(eq, 34 * eq_num_alloc, global_thread_idx, r1, r13);
    WriteIdx2<1024, double, double, double2>(eq, 36 * eq_num_alloc, global_thread_idx, r0, r5);
    WriteIdx2<1024, double, double, double2>(eq, 38 * eq_num_alloc, global_thread_idx, r12, r11);
    r14 = r15 + r14;
    r15 = r4 * r8;
    r14 = fma(r1, r15, r14);
    r1 = r12 * r9;
    r14 = fma(r10, r1, r14);
    WriteIdx2<1024, double, double, double2>(eq, 40 * eq_num_alloc, global_thread_idx, r8, r14);
  };
}

void ChainBTridigMakeEq(double* precond_diag, unsigned int precond_diag_num_alloc,
                        double* precond_tril, unsigned int precond_tril_num_alloc,
                        double* const tridig_U, unsigned int tridig_U_num_alloc,
                        int tridig_U_offset, int tridig_U_stride, double* njtr,
                        unsigned int njtr_num_alloc, const double* const diag, double* eq,
                        unsigned int eq_num_alloc, size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  // Kernels using more than 48 KB of shared memory per block must opt in.
  static const cudaError_t attribute_error = cudaFuncSetAttribute(
      ChainBTridigMakeEqKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 16384);
  if (attribute_error != cudaSuccess) {
    throw std::runtime_error(std::string("ChainBTridigMakeEqKernel: ") +
                             cudaGetErrorString(attribute_error));
  }
  ChainBTridigMakeEqKernel<<<n_blocks, 1024, 16384>>>(
      precond_diag, precond_diag_num_alloc, precond_tril, precond_tril_num_alloc, tridig_U,
      tridig_U_num_alloc, tridig_U_offset, tridig_U_stride, njtr, njtr_num_alloc, diag, eq,
      eq_num_alloc, problem_size);
  // A launch that fails (e.g. from exceeding shared memory or register limits)
  // otherwise leaves its outputs untouched without any error.
  const cudaError_t launch_error = cudaGetLastError();
  if (launch_error != cudaSuccess) {
    throw std::runtime_error(std::string("ChainBTridigMakeEqKernel launch failed: ") +
                             cudaGetErrorString(launch_error));
  }
}

}  // namespace caspar