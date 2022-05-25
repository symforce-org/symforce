Bundle-Adjustment-in-the-Large
======================================

[Source on GitHub](https://github.com/symforce-org/symforce/tree/main/symforce/examples/bundle_adjustment_in_the_large)

This example demonstrates bundle adjustment of camera extrinsics and intrinsics, as well as 3D landmark positions, for a Structure-from-Motion problem.  The example isn't particularly optimized for performance, but demonstrates the simplest way to set this up with SymForce.

We use the Bundle-Adjustment-in-the-Large dataset, as described here: https://grail.cs.washington.edu/projects/bal/

Feature correspondences have already been selected, and we're given initial guesses for all of the variables; our only task is to perform bundle adjustment.

The camera model is a simple polynomial model, and each image is assumed to be captured by a different camera with its own intrinsics.

Ceres and GTSAM also have reference implementations for this dataset, see [here](https://github.com/ceres-solver/ceres-solver/blob/master/examples/simple_bundle_adjuster.cc) for Ceres and [here](https://github.com/devbharat/gtsam/blob/master/examples/SFMExample_bal.cpp) for GTSAM.

## Files:

### `download_dataset.py`:

Script to download the dataset files into the `./data` folder, run this first if you'd like to run the example

### `bundle_adjustment_in_the_large.py`

Defines the symbolic residual function for the reprojection error factor, and a function to generate the symbolic factor into C++.  The `generate` function is called by `symforce/test/symforce_examples_bundle_adjustment_in_the_large_codegen_test.py` to generate everything in the `gen` directory.

### `bundle_adjustment_in_the_large.cc`

This is the C++ file that actually runs the optimization.  It loads a dataset, builds a factor graph,
and performs bundle adjustment.  See the comments there for more information.
