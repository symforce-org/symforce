#include <stdexcept>
#include <string>

#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include "kernel_backend_test_function.h"
#include "memops.cuh"

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    BackendTestFunctionKernel(double* x, unsigned int x_num_alloc, double* y,
                              unsigned int y_num_alloc, double* out, unsigned int out_num_alloc,
                              size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

  double r0 = 0, r1 = 0, r2 = 0, r3 = 0, r4 = 0, r5 = 0, r6 = 0, r7 = 0;

  if (global_thread_idx < problem_size) {
    r0 = 0.00000000000000000e+00;
    r1 = 1.00000000000000000e+00;
    WriteIdx2<1024, double, double, double2>(out, 0 * out_num_alloc, global_thread_idx, r0, r1);
    r0 = 5.00000000000000000e-01;
    r2 = 1.60000000000000009e+00;
    WriteIdx2<1024, double, double, double2>(out, 2 * out_num_alloc, global_thread_idx, r0, r2);
    r2 = 2.11538461538461536e-01;
    r3 = 2.71828182845904509e+00;
    WriteIdx2<1024, double, double, double2>(out, 4 * out_num_alloc, global_thread_idx, r2, r3);
    r3 = 2.00000000000000000e+00;
    r2 = log(r3);
    r4 = 1.0 / r2;
    WriteIdx2<1024, double, double, double2>(out, 6 * out_num_alloc, global_thread_idx, r4, r2);
    r2 = 1.00000000000000000e+01;
    r2 = log(r2);
    r4 = 3.14159265358979312e+00;
    WriteIdx2<1024, double, double, double2>(out, 8 * out_num_alloc, global_thread_idx, r2, r4);
    r2 = r0 * r4;
    r5 = 2.50000000000000000e-01;
    r5 = r4 * r5;
    WriteIdx2<1024, double, double, double2>(out, 10 * out_num_alloc, global_thread_idx, r2, r5);
    r5 = 1.0 / r4;
    r2 = r3 * r5;
    WriteIdx2<1024, double, double, double2>(out, 12 * out_num_alloc, global_thread_idx, r5, r2);
    r2 = sqrt(r3);
    r4 = rsqrt(r4);
    r4 = r3 * r4;
    WriteIdx2<1024, double, double, double2>(out, 14 * out_num_alloc, global_thread_idx, r4, r2);
    ReadIdx1<1024, double, double, double>(x, 0 * x_num_alloc, global_thread_idx, r4);
    r5 = fabs(r4);
    r2 = r0 * r2;
    WriteIdx2<1024, double, double, double2>(out, 16 * out_num_alloc, global_thread_idx, r2, r5);
    r5 = sin(r4);
    r2 = cos(r4);
    WriteIdx2<1024, double, double, double2>(out, 18 * out_num_alloc, global_thread_idx, r5, r2);
    r2 = tan(r4);
    r5 = asin(r4);
    WriteIdx2<1024, double, double, double2>(out, 20 * out_num_alloc, global_thread_idx, r2, r5);
    r5 = acos(r4);
    r2 = atan(r4);
    WriteIdx2<1024, double, double, double2>(out, 22 * out_num_alloc, global_thread_idx, r5, r2);
    r2 = exp(r4);
    r5 = log(r4);
    WriteIdx2<1024, double, double, double2>(out, 24 * out_num_alloc, global_thread_idx, r2, r5);
    r5 = sinh(r4);
    r2 = cosh(r4);
    WriteIdx2<1024, double, double, double2>(out, 26 * out_num_alloc, global_thread_idx, r5, r2);
    r2 = tanh(r4);
    r5 = floor(r4);
    WriteIdx2<1024, double, double, double2>(out, 28 * out_num_alloc, global_thread_idx, r2, r5);
    r5 = ceil(r4);
    r2 = sqrt(r4);
    WriteIdx2<1024, double, double, double2>(out, 30 * out_num_alloc, global_thread_idx, r5, r2);
    r5 = asinh(r4);
    r6 = acosh(r4);
    WriteIdx2<1024, double, double, double2>(out, 32 * out_num_alloc, global_thread_idx, r5, r6);
    r6 = atanh(r4);
    r5 = -5.50000000000000000e+00;
    r7 = 1.81818181818181823e-01;
    r7 = r4 * r7;
    r7 = floor(r7);
    r7 = fma(r5, r7, r4);
    WriteIdx2<1024, double, double, double2>(out, 34 * out_num_alloc, global_thread_idx, r6, r7);
    r7 = r4 + r1;
    r3 = r4 * r3;
    WriteIdx2<1024, double, double, double2>(out, 36 * out_num_alloc, global_thread_idx, r7, r3);
    r3 = r4 * r4;
    r7 = r4 * r3;
    WriteIdx2<1024, double, double, double2>(out, 38 * out_num_alloc, global_thread_idx, r3, r7);
    r3 = r3 * r3;
    r7 = r4 * r3;
    WriteIdx2<1024, double, double, double2>(out, 40 * out_num_alloc, global_thread_idx, r3, r7);
    r7 = r4 * r2;
    WriteIdx2<1024, double, double, double2>(out, 42 * out_num_alloc, global_thread_idx, r2, r7);
    r7 = copysign(1.0, r4);
    r1 = r1 + r7;
    r1 = r0 * r1;
    WriteIdx2<1024, double, double, double2>(out, 44 * out_num_alloc, global_thread_idx, r1, r7);
    r7 = erfc(r4);
    r1 = lgamma(r4);
    WriteIdx2<1024, double, double, double2>(out, 46 * out_num_alloc, global_thread_idx, r7, r1);
    r1 = erf(r4);
    r7 = tgamma(r4);
    WriteIdx2<1024, double, double, double2>(out, 48 * out_num_alloc, global_thread_idx, r1, r7);
    ReadIdx1<1024, double, double, double>(y, 0 * y_num_alloc, global_thread_idx, r7);
    r1 = atan2(r4, r7);
    r0 = fmax(r4, r7);
    WriteIdx2<1024, double, double, double2>(out, 50 * out_num_alloc, global_thread_idx, r1, r0);
    r0 = fmin(r4, r7);
    r1 = -1.00000000000000000e+00;
    r2 = r7 * r1;
    r3 = 1.0 / r7;
    r3 = r4 * r3;
    r3 = floor(r3);
    r2 = fma(r3, r2, r4);
    WriteIdx2<1024, double, double, double2>(out, 52 * out_num_alloc, global_thread_idx, r0, r2);
    r2 = r4 + r7;
    r0 = r4 * r7;
    WriteIdx2<1024, double, double, double2>(out, 54 * out_num_alloc, global_thread_idx, r2, r0);
    r0 = pow(r4, r7);
    r3 = r2 * r2;
    WriteIdx2<1024, double, double, double2>(out, 56 * out_num_alloc, global_thread_idx, r0, r3);
    r4 = copysign(r4, r7);
    r3 = r2 * r3;
    WriteIdx2<1024, double, double, double2>(out, 58 * out_num_alloc, global_thread_idx, r3, r4);
  };
}

void BackendTestFunction(double* x, unsigned int x_num_alloc, double* y, unsigned int y_num_alloc,
                         double* out, unsigned int out_num_alloc, size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  BackendTestFunctionKernel<<<n_blocks, 1024, 0>>>(x, x_num_alloc, y, y_num_alloc, out,
                                                   out_num_alloc, problem_size);
  // A launch that fails (e.g. from exceeding shared memory or register limits)
  // otherwise leaves its outputs untouched without any error.
  const cudaError_t launch_error = cudaGetLastError();
  if (launch_error != cudaSuccess) {
    throw std::runtime_error(std::string("BackendTestFunctionKernel launch failed: ") +
                             cudaGetErrorString(launch_error));
  }
}

}  // namespace caspar