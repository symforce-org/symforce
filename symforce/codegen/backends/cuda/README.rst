***THIS MODULE IS EXPERIMENTAL***

Backend for CUDA.  This generates CUDA ``__host__ __device__`` functions that are intended to be called from
kernels - e.g. to use one of these, you should write a kernel with your memory, threads, etc laid
out how you'd like, and call the generated function from one thread / for one element of your input
or output.

This currently only supports vector inputs and outputs, we do not have geo or cam types for CUDA yet.
