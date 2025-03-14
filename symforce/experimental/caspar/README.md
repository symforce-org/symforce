# Caspar

Caspar is an experimental extension of SymForce.
It enables the generation of optimized CUDA kernels from symbolic functions and includes a custom pipeline for generating and solving factor graphs using CUDA.

## Usage

### Kernels
Kernels can easily be defined in Python.
The `examples/kernel_example` folder contains the following kernel example and how to run it.

```python
caslib = CasparLibrary()

@caslib.add_kernel
def example_kernel(
    arg0: T.Annotated[sf.V3, mem.ReadShared],
    arg1: T.Annotated[sf.Symbol, mem.ReadUnique],
) -> T.Tuple[
    T.Annotated[sf.V2, mem.AddSharedSum],
    T.Annotated[sf.Symbol, mem.WriteIndexed],
]:
    sincos = sf.V2(sf.sin(arg0[0]), sf.cos(arg0[0]))
    product = arg0[2] * arg1
    return sincos, product
```

The Python code above will automatically generate the following optimized kernel. Caspar will find an expression ordering that minimizes register usage and utilizes CUDA intrinsics such as `fma`, `sincos`, and vector operations for performance.

```c++
__global__ void __launch_bounds__(1024, 1)
    example_kernel_kernel(float* arg0, unsigned int arg0_num_alloc, SharedIndex* arg0_indices,
                          float* arg1, float* out_0, unsigned int out_0_num_alloc,
                          SharedIndex* out_0_indices, float* out_1, unsigned int out_1_num_alloc,
                          unsigned int* out_1_indices, size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float inout_shared[4096];
  __shared__ SharedIndex arg0_indices_loc[1024];
  arg0_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size ? arg0_indices[global_thread_idx]
                                        : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ SharedIndex out_0_indices_loc[1024];
  out_0_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size ? out_0_indices[global_thread_idx]
                                        : SharedIndex{0xffffffff, 0xffff, 0xffff});
  unsigned int out_1_idx = out_1_indices[global_thread_idx];

  float r0, r1, r2, r3, r4;
  load_unique<2>(arg1, 4, inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_2(inout_shared, 0, r0, r1);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r1 = r1 + r0;
  };
  load_unique<4>(arg1, 0, inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_4(inout_shared, 0, r0, r2, r3, r4);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r1 = r1 + r4;
    r1 = r1 + r3;
    r1 = r1 + r2;
    r1 = r1 + r0;
  };
  load_shared<3>(arg0, 0 * arg0_num_alloc, arg0_indices_loc, inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_3(inout_shared, arg0_indices_loc[threadIdx.x].target, r0, r2, r2);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r2 = fma(r1, r2, 1.00000000e+00f);
    write_idx_1(out_1, 0 * out_1_num_alloc, out_1_idx, r2);
    sincosf(r0, &r0, &r2);
    r2 = ldexpf(r2, 1);
    r0 = ldexpf(r0, 1);
    write_sum_2(inout_shared, r0, r2);
  };
  flush_sum<2>(out_0, 0 * out_0_num_alloc, out_0_indices_loc, inout_shared);
}
```

### Factors
TODO

### Memory Accessors
Correct memory management is key to generating efficient CUDA kernels.
Caspar supports multiple ways for kernels to access memory.
The different access patterns generate code leveraging shared memory and warp intrinsics to achieve high throughput.

The following access patterns are currently supported:
- `ReadSequential`: Each thread reads the element at its global thread index.
- `WriteSequential`: Each thread writes to the element at its global thread index.
- `ReadIndexed`: Each thread reads an element specified by an index array.
- `WriteIndexed`: Each thread writes to an element specified by an index array.
- `ReadUnique`: Every thread reads the same element.
- `AddSequential`: Each thread adds to the element at its global thread index.
- `AddIndexed`: Each thread adds to the value specified by an index array.
- `ReadShared`: Same as `ReadIndexed` but uses shared memory to avoid read collisions.
- `AddSharedSum`: Same as `AddIndexed` but uses shared memory to avoid write collisions.
- `WriteBlockSum`: Each CUDA block writes to a single element. Useful for reductions.
- `Tunable`: Used in factor graphs to designate an input as tunable (such as input poses).
- `Constant`: Used in factor graphs to designate an input as constant (such as measurement values).




### Etymology
Caspar, an acronym for **C**UDA **A**ccelerator for **S**ymbolic **P**rogramming with **A**daptive **R**eordering, is named after the Danish–Norwegian mathematician [Caspar Wessel](https://en.wikipedia.org/wiki/Caspar_Wessel). Wessel was the first to describe the geometrical interpretation of complex numbers as points in the complex plane and as vectors. However, since his thesis was written in Danish, it initially received little recognition. When his work was rediscovered later, the mathematician Sophus Lie, known for his discovery of Lie algebra, wrote the following in the newspaper:

> "It is now Norway's cause to do what should be done, that this strange Man's Memory may be brought forth by Oblivion and his Name find its proper place before the History of Mathematics."

### Acknowledgements
Caspar began as a research project at the Norwegian University of Science and Technology (NTNU) as part of the SFI Autoship Centre. The current version of the library was developed during an internship at [Skydio](https://www.skydio.com/).
