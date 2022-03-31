Inverse Compose Jacobian Benchmark
---


This directory contains a benchmark that computes the inverse of a pose multiplied by a point, and the derivatives of that function with respect to the pose.  In some libraries, this can be optimized or fused in various ways, while other libraries compute jacobians for each of these things individually and multiply the jacobians together.  SymForce, for example, can generate a flat runtime function that computes the result and jacobian, leveraging sparsity and common subexpression elimination.
