Robot 3D Localization
=====================

[Source on GitHub](https://github.com/symforce-org/symforce/tree/main/symforce/examples/robot_3d_localization)

Demonstrates solving a 3D localization problem with SymForce. A robot moving
in 3D performs scan matching and gets relative translation constraints to landmarks
in the environment. It also has odometry constraints between its poses. The goal is
to estimate the trajectory of the robot given known landmarks and noisy measurements.

## Files:

### `robot_3d_localization.py`:

The main entry point for the symbolic problem.  In this file, we:

1. Define the symbolic residual functions we'll need
2. Build a symbolic factor graph for the full problem, and turn that into a single combined residual vector
3. Build a Python `Values` with numerical inputs to the problem - this includes the sampled measurements as well as initial guesses for the optimized variables (the poses)
4. Run the optimization from Python, creating an `Optimizer` and calling it with the created factors and `Values`.
5. Generate a linearization function to solve the same problem, but in C++

### `run_dynamic_size.cc`:

Constructs and optimizes a factor graph in C++, with the linearization functions we've generated.  The size of the problem (the number of poses, landmarks, and observations) can be changed at runtime.

### `run_fixed_size.cc`:

Runs the optimization from C++, using a fixed-size linearization function for the entire problem.  This is significantly faster than the dynamic size version, but requires that the number of poses and landmarks is known at codegen time.

### `common.h`:

Contains helper functions to build up the C++ `sym::Values` for the problem and create default params for the `Optimizer`

### `plotting.py`:

Contains helper functions for visualizing the optimization problem
