Fixed Size Bundle Adjustment
============================

[Source on GitHub](https://github.com/symforce-org/symforce/tree/main/symforce/examples/bundle_adjustment_fixed_size)

This example demonstrates bundle adjustment of camera extrinsics and landmarks using factors built into SymForce.  This particular example is set up so that the number of cameras and landmarks is set at code generation time; in contrast, the `Bundle Adjustment` example shows how to make them configurable at runtime.  Fixing the size of the problem at generation time can produce significantly more efficient linearization functions, because common subexpression elimination can be applied across multiple factors.  For instance, multiple factors that reproject different features into the same cameras will often share computation.

We randomly generate a set of camera poses and feature correspondences with noise and some outliers, and perform bundle adjustment with feature reprojection residuals and pose priors.

## Files:

### `build_example_state.*`:

Utilities for building up the problem, by randomly sampling camera poses and feature correspondences, and perturbing initial guesses.  This is all returned in the form of a `Values` produced by the `BuildValues` function.

### `build_values.py`:

Builds a symbolic Python `Values` with all of the variables in the problem.  This is used to build up the symbolic problem in `generate_fixed_problem.py`.

### `generate_fixed_problem.py`:

This actually defines the fixed-size problem, taking the `Values` built by `build_values.py` and constructing all of the residuals.  We can then generate the entire problem into C++, with common subexpression elimination running across the entire problem together.  The `FixedBundleAdjustmentProblem.generate` method is called by `test/symforce_examples_bundle_adjustment_fixed_size_codegen_test.py` to actually generate the linearization function in `gen`.

### `run_bundle_adjustment.cc`

This is the C++ file that actually runs the optimization.  It builds up the `Values` for the problem and builds a factor graph.  In this example, the C++ optimization consists of one `sym::Factor`, with a single generated linearization function that contains all of the symbolic residuals.
