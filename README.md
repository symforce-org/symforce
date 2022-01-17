![SymForce Logo](https://symforce.org/img/symforce_horizontal.png)

[![Documentation](https://img.shields.io/badge/api-reference-blue)](https://symforce-6d87c842-22de-4727-863b-e556dcc9093b.vercel.app/docs/index.html)
[![Source Code](https://img.shields.io/badge/source-code-blue)](https://github.com/symforce-org/symforce)
[![Issues](https://img.shields.io/badge/issue-tracker-blue)](https://github.com/symforce-org/symforce/issues)
![Python 3.8](https://img.shields.io/badge/python-3.8-blue)
![C++ 14](https://img.shields.io/badge/c++-14-blue)

---
<span style="color:red">**WARNING: SymForce must be kept confidential until its public release.**</code>

SymForce is a Python and C++ library for symbolic computation and code generation. It contains three independently useful systems:

+ **Symbolic Toolkit** - builds on the SymPy API to provide rigorous geometric and camera types, lie group calculus, singularity handling, and tools to model large problems

+ **Code Generator** - transforms symbolic expressions into blazing-fast, branchless code with clean APIs and minimal dependencies, with a template system to target any language

+ **Optimization Library** - performs on-manifold factor graph optimization with a highly optimized implementation for real-time robotics applications

SymForce accelerates robotics, vision, and applied science tasks like visual odometry, bundle adjustment, calibration, sparse nonlinear MPC, and embedded motor control. It is battle-hardened in production robotics.

Highlights:

+ Rapidly prototype and analyze complex problems with symbolic math
+ Compute fast and correct tangent-space jacobians for any expression
+ Reduce duplication and minimize bugs by generating native runtime code in multiple languages from a canonical symbolic representation
+ Generate embedded-friendly C++ functions that depend only on Eigen, are templated on the scalar type, and require no dynamic allocation
+ Outperform automatic differentiation techniques, sometimes by 10x
+ Leverage high quality and performant APIs at all levels

# Note to early access participants

Thank you for helping us develop SymForce! We value the time you're taking to provide feedback during this private beta program.

In conjunction with this program, you will be invited to a private Slack channel with the authors. Please file specific issues on Github, and then use Slack for discussion. Do not share any links or code without express permission from the authors.

Things to try:

* Review our tutorials and documentation
* Create symbolic expressions in a notebook
* Use the symbolic geometry and camera modules
* Compute tangent-space jacobians using `symforce/jacobian_helpers.py`
* Use the Python Factor and Optimizer to solve a problem
* Generate runtime C++ code with the Codegen class
* Use the generated C++ geometry module
* Use the C++ Factor and Optimizer to solve a problem
* Read and understand the SymForce codebase

In the future we will share a survey with specific questions, but more generally we are interested in all feedback you can provide about the value of the library, comparisons to alternatives, and any guidance to us.

# Building from source

<span style="color:blue">TODO: Create wheels for <code style="color:blue"><b>pip install symforce</b></code></span>

SymForce requires Python 3.8 or later. We suggest using conda or virtualenv:

```
conda create --name symforce "python>=3.8"
conda activate symforce
```

Install packages:
```
# Linux
apt install doxygen libgmp-dev pandoc

# Mac
brew install doxygen gmp pandoc

# Conda
conda install -c conda-forge doxygen gmp pandoc
```

Install python requirements:
```
pip install -r requirements.txt
```

Install CMake if you don't already have a recent version:
```
pip install "cmake>=3.17"
```

Build SymForce (requires C++14 or later):
```
mkdir build
cd build
cmake ..
make -j 7
```
If you have build errors, try updating CMake.

Install built Python packages:
```
cd ..
pip install -e build/lcmtypes/python2.7
pip install -e gen/python
pip install -e .
```

Verify the installation in Python:
```python
>>> from symforce import geo
>>> geo.Rot3()
```
