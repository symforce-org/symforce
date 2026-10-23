#include <stdexcept>
#include <string>

#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include "kernel_ChainB_cyclic_reduction_down.h"
#include "memops.cuh"

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(64, 1)
    ChainBCyclicReductionDownKernel(double* eqk, unsigned int eqk_num_alloc, int eqk_offset,
                                    int eqk_stride, double* const eq, unsigned int eq_num_alloc,
                                    int eq_offset, int eq_stride, double* eq_marked,
                                    unsigned int eq_marked_num_alloc, int eq_marked_offset,
                                    int eq_marked_stride, size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  // Scratch data, then the shared indices of each shared accessor; dynamic, so
  // that it may exceed the 48 KB static shared memory limit (see the launch
  // below).
  extern __shared__ __align__(16) uint8_t inout_shared[];

  double r0 = 0, r1 = 0, r2 = 0, r3 = 0, r4 = 0, r5 = 0, r6 = 0, r7 = 0, r8 = 0, r9 = 0, r10 = 0,
         r11 = 0, r12 = 0, r13 = 0, r14 = 0, r15 = 0, r16 = 0, r17 = 0, r18 = 0, r19 = 0, r20 = 0,
         r21 = 0, r22 = 0, r23 = 0, r24 = 0, r25 = 0, r26 = 0, r27 = 0, r28 = 0, r29 = 0, r30 = 0,
         r31 = 0, r32 = 0, r33 = 0, r34 = 0, r35 = 0, r36 = 0, r37 = 0, r38 = 0, r39 = 0, r40 = 0,
         r41 = 0, r42 = 0, r43 = 0, r44 = 0, r45 = 0, r46 = 0, r47 = 0, r48 = 0, r49 = 0, r50 = 0,
         r51 = 0, r52 = 0, r53 = 0, r54 = 0, r55 = 0, r56 = 0, r57 = 0, r58 = 0, r59 = 0, r60 = 0,
         r61 = 0, r62 = 0, r63 = 0, r64 = 0, r65 = 0, r66 = 0, r67 = 0, r68 = 0, r69 = 0, r70 = 0,
         r71 = 0, r72 = 0, r73 = 0, r74 = 0, r75 = 0, r76 = 0, r77 = 0, r78 = 0, r79 = 0, r80 = 0,
         r81 = 0, r82 = 0, r83 = 0, r84 = 0, r85 = 0, r86 = 0, r87 = 0, r88 = 0, r89 = 0, r90 = 0;

  if (global_thread_idx < problem_size) {
    ReadIdx2<64, double, double, double2>(eqk, 0 * eqk_num_alloc,
                                          global_thread_idx * eqk_stride + eqk_offset, r0, r1);
  };
  ReadAndShuffleWithDefault2<64, double, double, double2>(
      (double*)inout_shared, eq, 2 * eq_num_alloc, global_thread_idx * eq_stride + eq_offset,
      threadIdx.x == 63, global_thread_idx <= problem_size, eq_num_alloc, eq_stride, r2, r3, r4, r5,
      0.000000f, 0.000000f);
  if (global_thread_idx < problem_size) {
    ReadIdx2<64, double, double, double2>(eqk, 6 * eqk_num_alloc,
                                          global_thread_idx * eqk_stride + eqk_offset, r6, r7);
  };
  ReadAndShuffleWithDefault2<64, double, double, double2>(
      (double*)inout_shared, eq, 38 * eq_num_alloc, global_thread_idx * eq_stride + eq_offset,
      threadIdx.x == 63, global_thread_idx <= problem_size, eq_num_alloc, eq_stride, r8, r9, r10,
      r11, 0.000000f, 1.000000f);
  if (global_thread_idx < problem_size) {
    r12 = -1.00000000000000000e+00;
    r8 = r8 * r12;
  };
  ReadAndShuffleWithDefault2<64, double, double, double2>(
      (double*)inout_shared, eq, 36 * eq_num_alloc, global_thread_idx * eq_stride + eq_offset,
      threadIdx.x == 63, global_thread_idx <= problem_size, eq_num_alloc, eq_stride, r13, r14, r15,
      r16, 1.000000f, 0.000000f);
  if (global_thread_idx < problem_size) {
    r13 = 1.0 / r13;
    r8 = r8 * r13;
    r6 = fma(r0, r8, r6);
  };
  ReadAndShuffleWithDefault2<64, double, double, double2>(
      (double*)inout_shared, eq, 40 * eq_num_alloc, global_thread_idx * eq_stride + eq_offset,
      threadIdx.x == 63, global_thread_idx <= problem_size, eq_num_alloc, eq_stride, r17, r18, r19,
      r20, 0.000000f, 1.000000f);
  if (global_thread_idx < problem_size) {
    r17 = r17 * r12;
    r9 = 1.0 / r9;
    r17 = r17 * r9;
    ReadIdx2<64, double, double, double2>(eqk, 2 * eqk_num_alloc,
                                          global_thread_idx * eqk_stride + eqk_offset, r21, r22);
    r23 = r0 * r14;
    r23 = r23 * r12;
    r23 = fma(r13, r23, r22);
    r6 = fma(r23, r17, r6);
    r18 = 1.0 / r18;
  };
  ReadAndShuffleWithDefault2<64, double, double, double2>(
      (double*)inout_shared, eq, 34 * eq_num_alloc, global_thread_idx * eq_stride + eq_offset,
      threadIdx.x == 63, global_thread_idx <= problem_size, eq_num_alloc, eq_stride, r22, r24, r25,
      r26, 0.000000f, 1.000000f);
  if (global_thread_idx < problem_size) {
    r24 = 1.0 / r24;
    r24 = r18 * r24;
    r18 = r12 * r24;
    r27 = r6 * r18;
  };
  ReadAndShuffleWithDefault2<64, double, double, double2>(
      (double*)inout_shared, eq, 0 * eq_num_alloc, global_thread_idx * eq_stride + eq_offset,
      threadIdx.x == 63, global_thread_idx <= problem_size, eq_num_alloc, eq_stride, r28, r29, r30,
      r31, 0.000000f, 0.000000f);
  if (global_thread_idx < problem_size) {
    r32 = r29 * r12;
    r23 = fma(r23, r9, r22 * r27);
  };
  ReadAndShuffleWithDefault2<64, double, double, double2>(
      (double*)inout_shared, eq, 32 * eq_num_alloc, global_thread_idx * eq_stride + eq_offset,
      threadIdx.x == 63, global_thread_idx <= problem_size, eq_num_alloc, eq_stride, r33, r34, r35,
      r36, 1.000000f, 0.000000f);
  if (global_thread_idx < problem_size) {
    r33 = 1.0 / r33;
    r23 = r23 * r33;
    r32 = fma(r23, r32, r2 * r27);
    r37 = r28 * r12;
  };
  ReadAndShuffleWithDefault2<64, double, double, double2>(
      (double*)inout_shared, eq, 30 * eq_num_alloc, global_thread_idx * eq_stride + eq_offset,
      threadIdx.x == 63, global_thread_idx <= problem_size, eq_num_alloc, eq_stride, r38, r39, r40,
      r41, 1.000000f, 0.000000f);
  if (global_thread_idx < problem_size) {
    r42 = r39 * r12;
    r42 = fma(r0, r13, r23 * r42);
    r42 = fma(r34, r27, r42);
    r38 = 1.0 / r38;
    r42 = r42 * r38;
    r32 = fma(r42, r37, r32);
    r37 = r28 * r12;
    r43 = r39 * r12;
    ReadIdx2<64, double, double, double2>(eqk, 4 * eqk_num_alloc,
                                          global_thread_idx * eqk_stride + eqk_offset, r44, r45);
    r46 = r1 * r14;
    r46 = r46 * r12;
    r46 = fma(r13, r46, r44);
    r7 = fma(r46, r17, r7);
    r7 = fma(r1, r8, r7);
    r44 = r22 * r7;
    r44 = fma(r18, r44, r46 * r9);
    r43 = r43 * r44;
    r43 = fma(r33, r43, r1 * r13);
    r46 = r34 * r7;
    r43 = fma(r18, r46, r43);
    r37 = r37 * r43;
    r46 = r2 * r7;
    r46 = fma(r18, r46, r38 * r37);
    r37 = r29 * r12;
    r37 = r37 * r44;
    r46 = fma(r33, r37, r46);
    WriteIdx2<64, double, double, double2>(eq_marked, 0 * eq_marked_num_alloc,
                                           global_thread_idx * eq_marked_stride + eq_marked_offset,
                                           r32, r46);
    r46 = r28 * r12;
    r32 = r39 * r12;
    ReadIdx2<64, double, double, double2>(eqk, 8 * eqk_num_alloc,
                                          global_thread_idx * eqk_stride + eqk_offset, r37, r47);
    r8 = fma(r21, r8, r37);
    r37 = r21 * r14;
    r37 = r37 * r12;
    r37 = fma(r13, r37, r45);
    r8 = fma(r37, r17, r8);
    r17 = r22 * r8;
    r9 = fma(r37, r9, r18 * r17);
    r32 = r32 * r9;
    r13 = fma(r21, r13, r33 * r32);
    r32 = r34 * r8;
    r13 = fma(r18, r32, r13);
    r46 = r46 * r13;
    r32 = r2 * r8;
    r32 = fma(r18, r32, r38 * r46);
    r46 = r29 * r12;
    r46 = r46 * r9;
    r32 = fma(r33, r46, r32);
  };
  ReadAndShuffleWithDefault2<64, double, double, double2>(
      (double*)inout_shared, eq, 4 * eq_num_alloc, global_thread_idx * eq_stride + eq_offset,
      threadIdx.x == 63, global_thread_idx <= problem_size, eq_num_alloc, eq_stride, r46, r37, r17,
      r45, 0.000000f, 0.000000f);
  if (global_thread_idx < problem_size) {
    r48 = r46 * r12;
    r48 = fma(r23, r48, r37 * r27);
    r49 = r3 * r12;
    r48 = fma(r42, r49, r48);
    WriteIdx2<64, double, double, double2>(eq_marked, 2 * eq_marked_num_alloc,
                                           global_thread_idx * eq_marked_stride + eq_marked_offset,
                                           r32, r48);
    r48 = r3 * r12;
    r48 = r48 * r43;
    r32 = r37 * r7;
    r32 = fma(r18, r32, r38 * r48);
    r48 = r46 * r12;
    r48 = r48 * r44;
    r32 = fma(r33, r48, r32);
    r48 = r3 * r12;
    r48 = r48 * r13;
    r49 = r37 * r8;
    r49 = fma(r18, r49, r38 * r48);
    r48 = r46 * r12;
    r48 = r48 * r9;
    r49 = fma(r33, r48, r49);
    WriteIdx2<64, double, double, double2>(eq_marked, 4 * eq_marked_num_alloc,
                                           global_thread_idx * eq_marked_stride + eq_marked_offset,
                                           r32, r49);
  };
  ReadAndShuffleWithDefault2<64, double, double, double2>(
      (double*)inout_shared, eq, 8 * eq_num_alloc, global_thread_idx * eq_stride + eq_offset,
      threadIdx.x == 63, global_thread_idx <= problem_size, eq_num_alloc, eq_stride, r49, r32, r48,
      r50, 0.000000f, 1.000000f);
  ReadAndShuffleWithDefault2<64, double, double, double2>(
      (double*)inout_shared, eq, 6 * eq_num_alloc, global_thread_idx * eq_stride + eq_offset,
      threadIdx.x == 63, global_thread_idx <= problem_size, eq_num_alloc, eq_stride, r50, r32, r51,
      r52, 0.000000f, 0.000000f);
  if (global_thread_idx < problem_size) {
    r53 = r32 * r12;
    r53 = fma(r23, r53, r49 * r27);
    r27 = r50 * r12;
    r53 = fma(r42, r27, r53);
    r27 = r50 * r12;
    r27 = r27 * r43;
    r54 = r49 * r7;
    r54 = fma(r18, r54, r38 * r27);
    r27 = r32 * r12;
    r27 = r27 * r44;
    r54 = fma(r33, r27, r54);
    WriteIdx2<64, double, double, double2>(eq_marked, 6 * eq_marked_num_alloc,
                                           global_thread_idx * eq_marked_stride + eq_marked_offset,
                                           r53, r54);
  };
  ReadAndShuffleWithDefault2<64, double, double, double2>(
      (double*)inout_shared, eq, 20 * eq_num_alloc, global_thread_idx * eq_stride + eq_offset,
      threadIdx.x == 63, global_thread_idx <= problem_size, eq_num_alloc, eq_stride, r54, r53, r27,
      r55, 0.000000f, 0.000000f);
  if (global_thread_idx < problem_size) {
    r56 = r54 * r6;
  };
  ReadAndShuffleWithDefault2<64, double, double, double2>(
      (double*)inout_shared, eq, 18 * eq_num_alloc, global_thread_idx * eq_stride + eq_offset,
      threadIdx.x == 63, global_thread_idx <= problem_size, eq_num_alloc, eq_stride, r57, r58, r59,
      r60, 0.000000f, 0.000000f);
  if (global_thread_idx < problem_size) {
    r56 = fma(r58, r23, r24 * r56);
    r56 = fma(r57, r42, r56);
    r56 = fma(r12, r56, r47);
    r11 = 1.0 / r11;
    ReadIdx2<64, double, double, double2>(eqk, 20 * eqk_num_alloc,
                                          global_thread_idx * eqk_stride + eqk_offset, r47, r61);
    ReadIdx2<64, double, double, double2>(eqk, 18 * eqk_num_alloc,
                                          global_thread_idx * eqk_stride + eqk_offset, r62, r63);
    r16 = r16 * r12;
    r15 = 1.0 / r15;
    r16 = r16 * r15;
    r61 = fma(r62, r16, r61);
    ReadIdx2<64, double, double, double2>(eqk, 24 * eqk_num_alloc,
                                          global_thread_idx * eqk_stride + eqk_offset, r64, r65);
    r19 = r19 * r12;
    r19 = r19 * r11;
    r64 = fma(r61, r19, r64);
    r66 = r62 * r10;
    r66 = r66 * r12;
    r64 = fma(r15, r66, r64);
    r20 = 1.0 / r20;
    r26 = 1.0 / r26;
    r26 = r20 * r26;
    r64 = r64 * r26;
    r20 = r12 * r64;
    r61 = fma(r25, r20, r61 * r11);
    r35 = 1.0 / r35;
    r61 = r61 * r35;
    r66 = r41 * r12;
    r66 = fma(r61, r66, r62 * r15);
    r66 = fma(r36, r20, r66);
    r40 = 1.0 / r40;
    r66 = r66 * r40;
    r67 = fma(r30, r66, r31 * r61);
    r67 = fma(r4, r64, r67);
    r56 = fma(r12, r67, r56);
    r67 = r50 * r12;
    r67 = r67 * r13;
    r68 = r49 * r8;
    r68 = fma(r18, r68, r38 * r67);
    r67 = r32 * r12;
    r67 = r67 * r9;
    r68 = fma(r33, r67, r68);
    WriteIdx2<64, double, double, double2>(eq_marked, 8 * eq_marked_num_alloc,
                                           global_thread_idx * eq_marked_stride + eq_marked_offset,
                                           r68, r56);
    ReadIdx2<64, double, double, double2>(eqk, 10 * eqk_num_alloc,
                                          global_thread_idx * eqk_stride + eqk_offset, r68, r67);
    r18 = r41 * r12;
    r69 = r25 * r12;
    ReadIdx2<64, double, double, double2>(eqk, 22 * eqk_num_alloc,
                                          global_thread_idx * eqk_stride + eqk_offset, r70, r71);
    r70 = fma(r63, r16, r70);
    r65 = fma(r70, r19, r65);
    r72 = r63 * r10;
    r72 = r72 * r12;
    r65 = fma(r15, r72, r65);
    r69 = r69 * r65;
    r70 = fma(r70, r11, r26 * r69);
    r18 = r18 * r70;
    r18 = fma(r35, r18, r63 * r15);
    r69 = r36 * r12;
    r69 = r69 * r65;
    r18 = fma(r26, r69, r18);
    r69 = r30 * r18;
    r72 = r4 * r65;
    r72 = fma(r26, r72, r40 * r69);
    r69 = r31 * r70;
    r72 = fma(r35, r69, r72);
    r72 = fma(r12, r72, r68);
    r68 = r57 * r43;
    r69 = r58 * r44;
    r69 = fma(r33, r69, r38 * r68);
    r68 = r54 * r7;
    r69 = fma(r24, r68, r69);
    r72 = fma(r12, r69, r72);
    r69 = r25 * r12;
    ReadIdx2<64, double, double, double2>(eqk, 26 * eqk_num_alloc,
                                          global_thread_idx * eqk_stride + eqk_offset, r68, r73);
    r16 = fma(r47, r16, r71);
    r19 = fma(r16, r19, r68);
    r68 = r47 * r10;
    r68 = r68 * r12;
    r19 = fma(r15, r68, r19);
    r69 = r69 * r19;
    r11 = fma(r16, r11, r26 * r69);
    r16 = r31 * r11;
    r69 = r4 * r19;
    r69 = fma(r26, r69, r35 * r16);
    r16 = r36 * r12;
    r16 = r16 * r19;
    r16 = fma(r26, r16, r47 * r15);
    r15 = r41 * r12;
    r15 = r15 * r11;
    r16 = fma(r35, r15, r16);
    r15 = r30 * r16;
    r69 = fma(r40, r15, r69);
    r69 = fma(r12, r69, r67);
    r67 = r57 * r13;
    r15 = r54 * r8;
    r15 = fma(r24, r15, r38 * r67);
    r67 = r58 * r9;
    r15 = fma(r33, r67, r15);
    r69 = fma(r12, r15, r69);
    WriteIdx2<64, double, double, double2>(eq_marked, 10 * eq_marked_num_alloc,
                                           global_thread_idx * eq_marked_stride + eq_marked_offset,
                                           r72, r69);
    ReadIdx2<64, double, double, double2>(eqk, 12 * eqk_num_alloc,
                                          global_thread_idx * eqk_stride + eqk_offset, r15, r67);
  };
  ReadAndShuffleWithDefault2<64, double, double, double2>(
      (double*)inout_shared, eq, 22 * eq_num_alloc, global_thread_idx * eq_stride + eq_offset,
      threadIdx.x == 63, global_thread_idx <= problem_size, eq_num_alloc, eq_stride, r68, r71, r74,
      r75, 0.000000f, 0.000000f);
  if (global_thread_idx < problem_size) {
    r76 = r71 * r6;
    r76 = fma(r68, r23, r24 * r76);
    r76 = fma(r53, r42, r76);
    r76 = fma(r12, r76, r15);
    r15 = fma(r5, r66, r17 * r61);
    r15 = fma(r45, r64, r15);
    r76 = fma(r12, r15, r76);
    r15 = r5 * r18;
    r77 = r45 * r65;
    r77 = fma(r26, r77, r40 * r15);
    r15 = r17 * r70;
    r77 = fma(r35, r15, r77);
    r77 = fma(r12, r77, r67);
    r67 = r53 * r43;
    r15 = r68 * r44;
    r15 = fma(r33, r15, r38 * r67);
    r67 = r71 * r7;
    r15 = fma(r24, r67, r15);
    r77 = fma(r12, r15, r77);
    WriteIdx2<64, double, double, double2>(eq_marked, 12 * eq_marked_num_alloc,
                                           global_thread_idx * eq_marked_stride + eq_marked_offset,
                                           r76, r77);
    ReadIdx2<64, double, double, double2>(eqk, 14 * eqk_num_alloc,
                                          global_thread_idx * eqk_stride + eqk_offset, r15, r67);
    r78 = r17 * r11;
    r79 = r45 * r19;
    r79 = fma(r26, r79, r35 * r78);
    r78 = r5 * r16;
    r79 = fma(r40, r78, r79);
    r79 = fma(r12, r79, r15);
    r15 = r53 * r13;
    r78 = r71 * r8;
    r78 = fma(r24, r78, r38 * r15);
    r15 = r68 * r9;
    r78 = fma(r33, r15, r78);
    r79 = fma(r12, r78, r79);
  };
  ReadAndShuffleWithDefault2<64, double, double, double2>(
      (double*)inout_shared, eq, 26 * eq_num_alloc, global_thread_idx * eq_stride + eq_offset,
      threadIdx.x == 63, global_thread_idx <= problem_size, eq_num_alloc, eq_stride, r78, r15, r80,
      r81, 0.000000f, 0.000000f);
  if (global_thread_idx < problem_size) {
    r82 = r78 * r6;
  };
  ReadAndShuffleWithDefault2<64, double, double, double2>(
      (double*)inout_shared, eq, 24 * eq_num_alloc, global_thread_idx * eq_stride + eq_offset,
      threadIdx.x == 63, global_thread_idx <= problem_size, eq_num_alloc, eq_stride, r83, r84, r85,
      r86, 0.000000f, 0.000000f);
  if (global_thread_idx < problem_size) {
    r82 = fma(r84, r23, r24 * r82);
    r82 = fma(r83, r42, r82);
    r82 = fma(r12, r82, r67);
    r67 = fma(r51, r66, r52 * r61);
    r67 = fma(r48, r64, r67);
    r82 = fma(r12, r67, r82);
    WriteIdx2<64, double, double, double2>(eq_marked, 14 * eq_marked_num_alloc,
                                           global_thread_idx * eq_marked_stride + eq_marked_offset,
                                           r79, r82);
    ReadIdx2<64, double, double, double2>(eqk, 16 * eqk_num_alloc,
                                          global_thread_idx * eqk_stride + eqk_offset, r67, r87);
    r88 = r51 * r18;
    r89 = r48 * r65;
    r89 = fma(r26, r89, r40 * r88);
    r88 = r52 * r70;
    r89 = fma(r35, r88, r89);
    r89 = fma(r12, r89, r67);
    r67 = r83 * r43;
    r88 = r84 * r44;
    r88 = fma(r33, r88, r38 * r67);
    r67 = r78 * r7;
    r88 = fma(r24, r67, r88);
    r89 = fma(r12, r88, r89);
    r88 = r52 * r11;
    r67 = r48 * r19;
    r67 = fma(r26, r67, r35 * r88);
    r88 = r51 * r16;
    r67 = fma(r40, r88, r67);
    r67 = fma(r12, r67, r87);
    r87 = r83 * r13;
    r88 = r78 * r8;
    r88 = fma(r24, r88, r38 * r87);
    r87 = r84 * r9;
    r88 = fma(r33, r87, r88);
    r67 = fma(r12, r88, r67);
    WriteIdx2<64, double, double, double2>(eq_marked, 16 * eq_marked_num_alloc,
                                           global_thread_idx * eq_marked_stride + eq_marked_offset,
                                           r89, r67);
    r88 = r60 * r12;
    r87 = r59 * r12;
    r87 = fma(r66, r87, r61 * r88);
    r87 = fma(r27, r20, r87);
    r88 = r59 * r12;
    r88 = r88 * r18;
    r90 = r27 * r12;
    r90 = r90 * r65;
    r90 = fma(r26, r90, r40 * r88);
    r88 = r60 * r12;
    r88 = r88 * r70;
    r90 = fma(r35, r88, r90);
    WriteIdx2<64, double, double, double2>(eq_marked, 18 * eq_marked_num_alloc,
                                           global_thread_idx * eq_marked_stride + eq_marked_offset,
                                           r87, r90);
    r90 = r60 * r12;
    r90 = r90 * r11;
    r87 = r27 * r12;
    r87 = r87 * r19;
    r87 = fma(r26, r87, r35 * r90);
    r90 = r59 * r12;
    r90 = r90 * r16;
    r87 = fma(r40, r90, r87);
    r90 = r74 * r12;
    r88 = r55 * r12;
    r88 = fma(r66, r88, r61 * r90);
    r88 = fma(r75, r20, r88);
    WriteIdx2<64, double, double, double2>(eq_marked, 20 * eq_marked_num_alloc,
                                           global_thread_idx * eq_marked_stride + eq_marked_offset,
                                           r87, r88);
    r88 = r55 * r12;
    r88 = r88 * r18;
    r87 = r75 * r12;
    r87 = r87 * r65;
    r87 = fma(r26, r87, r40 * r88);
    r88 = r74 * r12;
    r88 = r88 * r70;
    r87 = fma(r35, r88, r87);
    r88 = r74 * r12;
    r88 = r88 * r11;
    r90 = r75 * r12;
    r90 = r90 * r19;
    r90 = fma(r26, r90, r35 * r88);
    r88 = r55 * r12;
    r88 = r88 * r16;
    r90 = fma(r40, r88, r90);
    WriteIdx2<64, double, double, double2>(eq_marked, 22 * eq_marked_num_alloc,
                                           global_thread_idx * eq_marked_stride + eq_marked_offset,
                                           r87, r90);
    r90 = r86 * r12;
    r87 = r85 * r12;
    r87 = fma(r66, r87, r61 * r90);
    r87 = fma(r80, r20, r87);
    r20 = r85 * r12;
    r20 = r20 * r18;
    r90 = r80 * r12;
    r90 = r90 * r65;
    r90 = fma(r26, r90, r40 * r20);
    r20 = r86 * r12;
    r20 = r20 * r70;
    r90 = fma(r35, r20, r90);
    WriteIdx2<64, double, double, double2>(eq_marked, 24 * eq_marked_num_alloc,
                                           global_thread_idx * eq_marked_stride + eq_marked_offset,
                                           r87, r90);
  };
  ReadAndShuffleWithDefault2<64, double, double, double2>(
      (double*)inout_shared, eq, 28 * eq_num_alloc, global_thread_idx * eq_stride + eq_offset,
      threadIdx.x == 63, global_thread_idx <= problem_size, eq_num_alloc, eq_stride, r90, r87, r20,
      r88, 0.000000f, 0.000000f);
  if (global_thread_idx < problem_size) {
    r64 = fma(r88, r64, r20 * r61);
    r64 = fma(r81, r66, r64);
    r64 = fma(r12, r64, r73);
    r23 = fma(r90, r23, r15 * r42);
    r42 = r87 * r6;
    r23 = fma(r24, r42, r23);
    r64 = fma(r12, r23, r64);
    r23 = r86 * r12;
    r23 = r23 * r11;
    r42 = r80 * r12;
    r42 = r42 * r19;
    r42 = fma(r26, r42, r35 * r23);
    r23 = r85 * r12;
    r23 = r23 * r16;
    r42 = fma(r40, r23, r42);
    WriteIdx2<64, double, double, double2>(eq_marked, 26 * eq_marked_num_alloc,
                                           global_thread_idx * eq_marked_stride + eq_marked_offset,
                                           r42, r64);
    ReadIdx2<64, double, double, double2>(eqk, 28 * eqk_num_alloc,
                                          global_thread_idx * eqk_stride + eqk_offset, r64, r42);
    r23 = r20 * r70;
    r73 = r81 * r18;
    r73 = fma(r40, r73, r35 * r23);
    r23 = r88 * r65;
    r73 = fma(r26, r23, r73);
    r73 = fma(r12, r73, r64);
    r64 = r15 * r43;
    r23 = r90 * r44;
    r23 = fma(r33, r23, r38 * r64);
    r64 = r87 * r7;
    r23 = fma(r24, r64, r23);
    r73 = fma(r12, r23, r73);
    r23 = r90 * r9;
    r64 = r87 * r8;
    r64 = fma(r24, r64, r33 * r23);
    r23 = r15 * r13;
    r64 = fma(r38, r23, r64);
    r64 = fma(r12, r64, r42);
    r42 = r88 * r19;
    r23 = r81 * r16;
    r23 = fma(r40, r23, r26 * r42);
    r42 = r20 * r11;
    r23 = fma(r35, r42, r23);
    r64 = fma(r12, r23, r64);
    WriteIdx2<64, double, double, double2>(eq_marked, 28 * eq_marked_num_alloc,
                                           global_thread_idx * eq_marked_stride + eq_marked_offset,
                                           r73, r64);
    ReadIdx2<64, double, double, double2>(eqk, 30 * eqk_num_alloc,
                                          global_thread_idx * eqk_stride + eqk_offset, r64, r73);
    r73 = 1.00000000000000000e+00;
    r64 = 1.0 / r56;
    r72 = r72 * r64;
    WriteIdx2<64, double, double, double2>(eq_marked, 30 * eq_marked_num_alloc,
                                           global_thread_idx * eq_marked_stride + eq_marked_offset,
                                           r73, r72);
    ReadIdx2<64, double, double, double2>(eqk, 32 * eqk_num_alloc,
                                          global_thread_idx * eqk_stride + eqk_offset, r23, r42);
    r42 = r69 * r64;
    WriteIdx2<64, double, double, double2>(eq_marked, 32 * eq_marked_num_alloc,
                                           global_thread_idx * eq_marked_stride + eq_marked_offset,
                                           r73, r42);
    ReadIdx2<64, double, double, double2>(eqk, 34 * eqk_num_alloc,
                                          global_thread_idx * eqk_stride + eqk_offset, r42, r23);
    r72 = r12 * r72;
    r77 = fma(r76, r72, r77);
    r23 = 1.0 / r77;
    r42 = r12 * r69;
    r42 = r42 * r76;
    r42 = fma(r64, r42, r79);
    r42 = r23 * r42;
    WriteIdx2<64, double, double, double2>(eq_marked, 34 * eq_marked_num_alloc,
                                           global_thread_idx * eq_marked_stride + eq_marked_offset,
                                           r42, r73);
    ReadIdx2<64, double, double, double2>(eqk, 36 * eqk_num_alloc,
                                          global_thread_idx * eqk_stride + eqk_offset, r73, r23);
    WriteIdx2<64, double, double, double2>(eq_marked, 36 * eq_marked_num_alloc,
                                           global_thread_idx * eq_marked_stride + eq_marked_offset,
                                           r56, r76);
    ReadIdx2<64, double, double, double2>(eqk, 38 * eqk_num_alloc,
                                          global_thread_idx * eqk_stride + eqk_offset, r76, r56);
    WriteIdx2<64, double, double, double2>(eq_marked, 38 * eq_marked_num_alloc,
                                           global_thread_idx * eq_marked_stride + eq_marked_offset,
                                           r82, r77);
    ReadIdx2<64, double, double, double2>(eqk, 40 * eqk_num_alloc,
                                          global_thread_idx * eqk_stride + eqk_offset, r77, r56);
    r72 = fma(r82, r72, r89);
    r89 = r12 * r72;
    r56 = r12 * r69;
    r56 = r56 * r82;
    r56 = fma(r64, r56, r42 * r89);
    r56 = r56 + r67;
    WriteIdx2<64, double, double, double2>(eq_marked, 40 * eq_marked_num_alloc,
                                           global_thread_idx * eq_marked_stride + eq_marked_offset,
                                           r72, r56);
  };
}

void ChainBCyclicReductionDown(double* eqk, unsigned int eqk_num_alloc, int eqk_offset,
                               int eqk_stride, double* const eq, unsigned int eq_num_alloc,
                               int eq_offset, int eq_stride, double* eq_marked,
                               unsigned int eq_marked_num_alloc, int eq_marked_offset,
                               int eq_marked_stride, size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 64 - 1) / 64;
  // Kernels using more than 48 KB of shared memory per block must opt in.
  static const cudaError_t attribute_error = cudaFuncSetAttribute(
      ChainBCyclicReductionDownKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 1024);
  if (attribute_error != cudaSuccess) {
    throw std::runtime_error(std::string("ChainBCyclicReductionDownKernel: ") +
                             cudaGetErrorString(attribute_error));
  }
  ChainBCyclicReductionDownKernel<<<n_blocks, 64, 1024>>>(
      eqk, eqk_num_alloc, eqk_offset, eqk_stride, eq, eq_num_alloc, eq_offset, eq_stride, eq_marked,
      eq_marked_num_alloc, eq_marked_offset, eq_marked_stride, problem_size);
  // A launch that fails (e.g. from exceeding shared memory or register limits)
  // otherwise leaves its outputs untouched without any error.
  const cudaError_t launch_error = cudaGetLastError();
  if (launch_error != cudaSuccess) {
    throw std::runtime_error(std::string("ChainBCyclicReductionDownKernel launch failed: ") +
                             cudaGetErrorString(launch_error));
  }
}

}  // namespace caspar