Custom Factor Generation
========================

[Source on GitHub](https://github.com/symforce-org/symforce/tree/main/symforce/examples/custom_factor_generation)

Here we show how to define your own residual function and generate it into a factor to use from C++.  You can define factors by importing existing symbolic factors that we've defined and modifying them, or by writing your own symbolic functions from scratch using our symbolic toolkit.

## Files:

### `factor_residuals.py`:

Defines a single function `custom_between_factor_residual`, which takes a set of symbolic arguments and returns a symbolic vector (the residual).  The inputs to the function represent all of the variables the residual depends on, both optimized variables and constants (plus, you can generate the same residual function with different sets of optimized variables).

### `generate_factors.py`:

Contains a `generate` function which generates the residual in `factor_residuals.py` into an output directory.  The `generate` function here is called by `test/symforce_examples_custom_factor_generation_codegen_test.py` to generate the code in `gen` under this directory.

### `gen/*`:

This directory contains the generated code for the C++ linearization function.
