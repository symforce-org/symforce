Robot 2D Localization
=====================

[Source on GitHub](https://github.com/symforce-org/symforce/tree/main/symforce/examples/robot_2d_localization)

Demonstrates solving a 2D localization problem with SymForce. The goal is for a robot
in a 2D plane to compute its trajectory given distance measurements from wheel odometry
and relative bearing angle measurements to known landmarks in the environment.

## Files:

### `robot_2d_localization.py`:

Sets up and solves the optimization problem step-by-step.  See the [tutorial](https://symforce.org#tutorial) on the SymForce homepage for a detailed walkthrough.

### `plotting.py`:

Contains helper functions for visualizing the optimization problem
