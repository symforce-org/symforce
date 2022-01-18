<img alt="SymForce" src="https://symforce-6d87c842-22de-4727-863b-e556dcc9093b.vercel.app/images/symforce_horizontal_white.png" width="600px"/>

[![Documentation](https://img.shields.io/badge/api-docs-blue)](https://symforce-6d87c842-22de-4727-863b-e556dcc9093b.vercel.app/docs/index.html)
[![Source Code](https://img.shields.io/badge/source-code-blue)](https://github.com/symforce-org/symforce)
[![Issues](https://img.shields.io/badge/issue-tracker-blue)](https://github.com/symforce-org/symforce/issues)
![Python 3.8](https://img.shields.io/badge/python-3.8-blue)
![C++ 14](https://img.shields.io/badge/c++-14-blue)

---
<span style="color:red"><b>WARNING: SymForce is strictly confidential until its public release.</b></span>

SymForce is a Python and C++ library for symbolic computation and code generation. It contains three independently useful systems:

+ **Symbolic Toolkit** - builds on the SymPy API to provide rigorous geometric and camera types, lie group calculus, singularity handling, and tools to model large problems

+ **Code Generator** - transforms symbolic expressions into blazing-fast, branchless code with clean APIs and minimal dependencies, with a template system to target any language

+ **Optimization Library** - performs on-manifold factor graph optimization with a highly optimized implementation for real-time robotics applications

SymForce accelerates robotics and computer vision tasks like visual odometry, bundle adjustment, calibration, sparse nonlinear MPC, and embedded motor control.

<br/>

<img alt="SymForce" src="https://symforce-6d87c842-22de-4727-863b-e556dcc9093b.vercel.app/images/symforce_diagram.png" width="700px"/>

<br/>

Features:

+ Rapidly prototype and analyze complex problems with symbolic math
+ Compute fast and correct tangent-space jacobians for any expression
+ Reduce duplication and minimize bugs by generating native runtime code in multiple languages from a canonical symbolic representation
+ Generate embedded-friendly C++ functions that depend only on Eigen, are templated on the scalar type, and require no dynamic allocation
+ Outperform automatic differentiation techniques, sometimes by 10x
+ Leverage high quality, battle-hardened, and performant APIs at all levels

# Note to early access participants

Thank you for helping us develop SymForce! We value the time you're taking to provide feedback during this private beta program.

You will be invited to a private Slack channel with the authors. Please file specific issues on Github, and then use Slack for discussion. Do not share any links or code without express permission from the authors.

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

[**Read the docs!**](https://symforce-6d87c842-22de-4727-863b-e556dcc9093b.vercel.app/docs/index.html)

In the future we will share a survey with specific questions, but more generally we are interested in all feedback you can provide about the value of the library, comparisons to alternatives, and any guidance to us.

# Build from source

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

# Tutorial

Let's walk through a simple example of modeling and optimizing a problem with SymForce. In this example a robot moves through a 2D plane and the goal is to estimate its pose at multiple time steps given noisy measurements.

The robot measures:

 * the distance it traveled from an odometry sensor
 * relative bearing angles to known landmarks in the scene

The robot's heading angle is defined counter-clockwise from the x-axis, and its relative bearing measurements are defined from the robot's forward direction:

<img alt="Robot 2D Triangulation Figure" src="https://symforce-6d87c842-22de-4727-863b-e556dcc9093b.vercel.app/images/robot_2d_triangulation/robot_2d_triangulation_figure.png" width="350px"/>

## Explore the math

Import the augmented SymPy API and geometry module:
```python
from symforce import sympy as sm
from symforce import geo
```

Create a symbolic 2D pose and landmark location. Using symbolic variables lets us explore and build up the math in a pure form.
```python
pose = geo.Pose2(
    t=geo.V2.symbolic("t"),
    R=geo.Rot2.symbolic("R")
)
landmark = geo.V2.symbolic("L")
```

Let's transform the landmark into the local frame of the robot:
```python
landmark_body = pose.inverse() * landmark
```
<img src="https://latex.codecogs.com/png.image?\dpi{200}&space;\begin{bmatrix}&space;&space;R_{re}&space;L_0&space;&plus;&space;R_{im}&space;L_1&space;-&space;R_{im}&space;t_1&space;-&space;R_{re}&space;t_0&space;\\&space;&space;-R_{im}&space;L_0&space;&plus;&space;R_{re}&space;L_1&space;&plus;&space;R_{im}&space;t_0&space;&plus;&space;R_{re}&space;t_1\end{bmatrix}" width="250px" />

<!-- $
\begin{bmatrix}
  R_{re} L_0 + R_{im} L_1 - R_{im} t_1 - R_{re} t_0 \\
  -R_{im} L_0 + R_{re} L_1 + R_{im} t_0 + R_{re} t_1
\end{bmatrix}
$ -->

You can see that `geo.Rot2` is represented internally by a complex number (𝑅𝑟𝑒, 𝑅𝑖𝑚) and we can study how it rotates the landmark 𝐿.

For exploration purposes, let's take the jacobian of the body-frame landmark with respect to the tangent space of the `Pose2`, parameterized as (𝑥, 𝑦, 𝜃):

<!-- TODO: flip this when we make Pose2 (theta, x, y) -->

```python
landmark_body.jacobian(pose)
```
<img src="https://latex.codecogs.com/png.image?\dpi{200}&space;\begin{bmatrix}&space;&space;-R_{re}&space;,&space;&&space;-R_{im},&space;&space;&&space;-R_{im}&space;L_0&space;&plus;&space;R_{re}&space;L_1&space;&plus;&space;R_{im}&space;t_0&space;-&space;R_{re}&space;t_1&space;\\&space;&space;R_{im}&space;,&space;&&space;-R_{re}&space;,&space;&&space;-R_{re}&space;L_0&space;-&space;R_{im}&space;L_1&space;&plus;&space;R_{re}&space;t_0&space;&plus;&space;R_{im}&space;t_1\end{bmatrix}" width="350px" />

<!-- $
\begin{bmatrix}
  -R_{re} , & -R_{im},  & -R_{im} L_0 + R_{re} L_1 + R_{im} t_0 - R_{re} t_1 \\
  R_{im} , & -R_{re} , & -R_{re} L_0 - R_{im} L_1 + R_{re} t_0 + R_{im} t_1
\end{bmatrix}
$ -->

Note that even though the orientation is stored as a complex number, the tangent space is a scalar angle and SymForce understands that.

Now compute the relative bearing angle:

```python
sm.atan2(landmark_body[1], landmark_body[0])
```
<img src="https://latex.codecogs.com/png.image?\dpi{200}&space;atan_2(-R_{im}&space;L_0&space;&plus;&space;R_{re}&space;L_1&space;&plus;&space;R_{im}&space;t_0&space;&plus;&space;R_{re}&space;t_1,&space;R_{re}&space;L_0&space;&space;&plus;&space;R_{im}&space;L_1&space;-&space;R_{im}&space;t_1&space;-&space;R_{re}&space;t_0)" width="500px" />

<!-- $
atan_2(-R_{im} L_0 + R_{re} L_1 + R_{im} t_0 + R_{re} t_1, R_{re} L_0  + R_{im} L_1 - R_{im} t_1 - R_{re} t_0)
$ -->

One important note is that `atan2` is singular at (0, 0). In SymForce we handle this by placing a symbol ϵ (epsilon) that preserves the value of an expression in the limit of ϵ → 0, but allows evaluating at runtime with a very small nonzero value. Functions with singularities accept an `epsilon` argument:

```python
geo.V3.symbolic("x").norm(epsilon=sm.epsilon)
```
<img src="https://latex.codecogs.com/png.image?\dpi{200}&space;\sqrt{x_0^2&space;&plus;&space;x_1^2&space;&plus;&space;x_2^2&space;&plus;&space;\epsilon}" width="135px" />

<!-- $\sqrt{x_0^2 + x_1^2 + x_2^2 + \epsilon}$ -->

<span style="color:blue">TODO: Link to a detailed epsilon tutorial once created.</span>

## Build an optimization problem

We will model this problem as a factor graph and solve it with nonlinear least-squares.

The residual function comprises of two terms - one for the bearing measurements and one for the odometry measurements. Let's formalize the math we just defined for the bearing measurements into a symbolic residual function:

```python
from symforce import typing as T

def bearing_residual(
    pose: geo.Pose2, landmark: geo.V2, angle: T.Scalar, epsilon: T.Scalar
) -> geo.V1:
    t_body = pose.inverse() * landmark
    predicted_angle = sm.atan2(t_body[1], t_body[0], epsilon=epsilon)
    return geo.V1(sm.wrap_angle(predicted_angle - angle))
```

This function takes in a pose and landmark variable and returns the error between the predicted bearing angle and a measured value. Note that we call `sm.wrap_angle` on the angle difference to prevent wraparound effects.

The residual for distance traveled is even simpler:

```python
def odometry_residual(
    pose_a: geo.Pose2, pose_b: geo.Pose2, dist: T.Scalar, epsilon: T.Scalar
) -> geo.V1:
    return geo.V1((pose_b.t - pose_a.t).norm(epsilon=epsilon) - dist)
```

Now we can create [`Factor`](https://symforce-6d87c842-22de-4727-863b-e556dcc9093b.vercel.app/docs/api/symforce.opt.factor.html) objects from the residual functions and a set of keys. The keys are named strings for the function arguments, which will be accessed by name from a [`Values`](https://symforce-6d87c842-22de-4727-863b-e556dcc9093b.vercel.app/docs/api/symforce.values.values.html) class we later instantiate with numerical quantities.

```python
num_poses = 3
num_landmarks = 3

factors = []

# Bearing factors
for i in range(num_poses):
    for j in range(num_landmarks):
        factors.append(Factor(
            residual=bearing_residual,
            keys=[f"poses[{i}]", f"landmarks[{j}]", f"angles[{i}][{j}]", "epsilon"],
        ))

# Odometry factors
for i in range(num_poses - 1):
    factors.append(Factor(
        residual=odometry_residual,
        keys=[f"poses[{i}]", f"poses[{i + 1}]", f"distances[{i}]", "epsilon"],
    ))
```

Here is a visualization of the structure of this factor graph:

<img alt="Robot 2D Triangulation Factor Graph" src="https://symforce-6d87c842-22de-4727-863b-e556dcc9093b.vercel.app/images/robot_2d_triangulation/robot_2d_triangulation_factor_graph.png" width="600px"/>

## Solve the problem

Our goal is to find poses of the robot that minimize the residual of this factor graph. We create an [`Optimizer`](https://symforce-6d87c842-22de-4727-863b-e556dcc9093b.vercel.app/docs/api/symforce.opt.optimizer.html) with these factors and tell it to only optimize the pose keys (the rest are held constant):
```python
optimizer = Optimizer(
    factors=factors,
    optimized_keys=[f"poses[{i}]" for i in range(num_poses)]
)
```

Now we need to instantiate numerical [`Values`](https://symforce-6d87c842-22de-4727-863b-e556dcc9093b.vercel.app/docs/api/symforce.values.values.html) for the problem, including an initial guess for our unknown poses (just set them to identity).

```python
import numpy as np
from symforce.values import Values

initial_values = Values(
    poses=[geo.Pose2.identity()] * num_poses,
    landmarks=[geo.V2(-2, 2), geo.V2(1, -3), geo.V2(5, 2)],
    distances=[1.7, 1.4],
    angles=np.deg2rad([[145, 335, 55], [185, 310, 70], [215, 310, 70]]).tolist(),
    epsilon=sm.default_epsilon,
)
```

Now run the optimization! This returns an [`Optimizer.Result`](https://symforce-6d87c842-22de-4727-863b-e556dcc9093b.vercel.app/docs/api/symforce.opt.optimizer.html#symforce.opt.optimizer.Optimizer.Result) object that contains the optimized values, error statistics, and per-iteration debug stats (if enabled).
```python
result = optimizer.optimize(initial_values)
```

Let's visualize what the optimizer did. The green circle represent the fixed landmarks, the blue circles represent the robot, and the dotted lines represent the bearing measurements.

```python
from symforce.examples.robot_2d_triangulation.plotting import plot_solution
plot_solution(optimizer, result)
```
<img alt="Robot 2D Triangulation Solution" src="https://symforce-6d87c842-22de-4727-863b-e556dcc9093b.vercel.app/images/robot_2d_triangulation/robot_2d_triangulation_iterations.gif" width="600px"/>

With the `verbose=True` param in the optimizer, it will print a table of its progress:
```
[iter 0] lambda: 1.000e+00, error prev/linear/new: 6.396/2.952/2.282, rel reduction: 0.64328
[iter 1] lambda: 2.500e-01, error prev/linear/new: 2.282/0.088/0.074, rel reduction: 0.96768
[iter 2] lambda: 6.250e-02, error prev/linear/new: 0.074/0.007/0.007, rel reduction: 0.91152
[iter 3] lambda: 1.562e-02, error prev/linear/new: 0.007/0.001/0.001, rel reduction: 0.90289
[iter 4] lambda: 3.906e-03, error prev/linear/new: 0.001/0.000/0.000, rel reduction: 0.61885
[iter 5] lambda: 9.766e-04, error prev/linear/new: 0.000/0.000/0.000, rel reduction: 0.08876
[iter 6] lambda: 2.441e-04, error prev/linear/new: 0.000/0.000/0.000, rel reduction: 0.00013
[iter 7] lambda: 6.104e-05, error prev/linear/new: 0.000/0.000/0.000, rel reduction: 0.00000

Final error: 0.000220
```

## Symbolic vs Numerical Types

SymForce provides `sym` packages with runtime code for geometry and camera types that are generated from its symbolic `geo` and `cam` packages. As such, there are multiple versions of a class like `Pose3` and it can be a common source of confusion.

The canonical symbolic class [`geo.Pose3`](https://symforce-6d87c842-22de-4727-863b-e556dcc9093b.vercel.app/docs/api/symforce.geo.pose3.html) lives in the `symforce` package:
```python
from symforce import geo
geo.Pose3.identity()
```

The autogenerated Python runtime class [`sym.Pose3`](https://symforce-6d87c842-22de-4727-863b-e556dcc9093b.vercel.app/docs/api-gen-py/sym.pose3.html) lives in the `sym` package:
```python
import sym
sym.Pose.identity()
```

The autogenerated C++ runtime class [`sym::Pose3`](https://symforce-6d87c842-22de-4727-863b-e556dcc9093b.vercel.app/docs/api-gen-cpp/class/classsym_1_1Pose3.html) lives in the `sym::` namespace:
```C++
sym::Pose3<double>::Identity()
```

The matrix type for symbolic code is [`geo.Matrix`](https://symforce-6d87c842-22de-4727-863b-e556dcc9093b.vercel.app/docs/api/symforce.geo.matrix.html), for generated Python is [`numpy.ndarray`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html), and for C++ is [`Eigen::Matrix`](https://eigen.tuxfamily.org/dox/group__TutorialMatrixClass.html).

The symbolic classes can also handle numerical values, but will be dramatically slower than the generated classes. The symbolic classes must be used when defining functions for codegen and optimization. Generated functions always accept the runtime types.

The `Codegen` or `Factor` objects require symbolic types and functions to do math and generate code. However, once code is generated, numerical types should be used when invoking generated functions and in the initial values when calling the `Optimizer`.

As a convenience, the Python `Optimizer` class can accept symbolic types in its `Values` and convert to numerical types before invoking generated functions. This is done in the tutorial example for simplicity.

## Generate runtime C++ code

Let's look under the hood to understand how that works. For each factor, SymForce introspects the form of the symbolic function, passes through symbolic inputs to build an output expression, automatically computes tangent-space jacobians of those output expressions wrt the optimized variables, and generates fast runtime code for them.

The [`Codegen`](https://symforce-6d87c842-22de-4727-863b-e556dcc9093b.vercel.app/docs/api/symforce.codegen.codegen.html) class is the central tool for generating runtime code from symbolic expressions. In this case, we pass it the bearing residual function and configure it to generate C++ code:
```python
from symforce.codegen import Codegen, CppConfig

codegen = Codegen.function(bearing_residual, config=CppConfig())
```

We can then create another `Codegen` object that computes a Gauss-Newton linearization from this Codegen object. It does this by introspecting and symbolically differentiating the given arguments:
```python
codegen_linearization = codegen.with_linearization(
    which_args=["pose"]
)
```

Generate a C++ function that computes the linearization wrt the pose argument:
```python
metadata = codegen_linearization.generate_function()
print(open(metadata["generated_files"][0]).read())
```

This C++ code depends only on Eigen and computes the results in a single flat function that shares all common sub-expressions:

```c++
#pragma once

#include <Eigen/Dense>

#include <sym/pose2.h>

namespace sym {

/**
 * Residual from a relative bearing mesurement of a 2D pose to a landmark.
 *     jacobian: (1x3) jacobian of res wrt arg pose (3)
 *     hessian: (3x3) Gauss-Newton hessian for arg pose (3)
 *     rhs: (3x1) Gauss-Newton rhs for arg pose (3)
 */
template <typename Scalar>
void BearingFactor(const sym::Pose2<Scalar>& pose, const Eigen::Matrix<Scalar, 2, 1>& landmark,
                   const Scalar angle, const Scalar epsilon,
                   Eigen::Matrix<Scalar, 1, 1>* const res = nullptr,
                   Eigen::Matrix<Scalar, 1, 3>* const jacobian = nullptr,
                   Eigen::Matrix<Scalar, 3, 3>* const hessian = nullptr,
                   Eigen::Matrix<Scalar, 3, 1>* const rhs = nullptr) {
  // Total ops: 79

  // Input arrays
  const Eigen::Matrix<Scalar, 4, 1>& _pose = pose.Data();

  // Intermediate terms (24)
  const Scalar _tmp0 = _pose[1] * _pose[2];
  const Scalar _tmp1 = _pose[0] * _pose[3];
  const Scalar _tmp2 = _pose[0] * landmark(1, 0) - _pose[1] * landmark(0, 0);
  const Scalar _tmp3 = _tmp0 - _tmp1 + _tmp2;
  const Scalar _tmp4 = _pose[0] * _pose[2] + _pose[1] * _pose[3];
  const Scalar _tmp5 = _pose[1] * landmark(1, 0);
  const Scalar _tmp6 = _pose[0] * landmark(0, 0);
  const Scalar _tmp7 = -_tmp4 + _tmp5 + _tmp6;
  const Scalar _tmp8 = _tmp7 + epsilon * ((((_tmp7) > 0) - ((_tmp7) < 0)) + Scalar(0.5));
  const Scalar _tmp9 = -angle + std::atan2(_tmp3, _tmp8);
  const Scalar _tmp10 =
      _tmp9 - 2 * Scalar(M_PI) *
                  std::floor((Scalar(1) / Scalar(2)) * (_tmp9 + Scalar(M_PI)) / Scalar(M_PI));
  const Scalar _tmp11 = std::pow(_tmp8, Scalar(2));
  const Scalar _tmp12 = _tmp3 / _tmp11;
  const Scalar _tmp13 = Scalar(1.0) / (_tmp8);
  const Scalar _tmp14 = _pose[0] * _tmp12 + _pose[1] * _tmp13;
  const Scalar _tmp15 = _tmp11 + std::pow(_tmp3, Scalar(2));
  const Scalar _tmp16 = _tmp11 / _tmp15;
  const Scalar _tmp17 = _tmp14 * _tmp16;
  const Scalar _tmp18 = -_pose[0] * _tmp13 + _pose[1] * _tmp12;
  const Scalar _tmp19 = _tmp16 * _tmp18;
  const Scalar _tmp20 = -_tmp12 * (_tmp0 - _tmp1 + _tmp2) + _tmp13 * (_tmp4 - _tmp5 - _tmp6);
  const Scalar _tmp21 = _tmp16 * _tmp20;
  const Scalar _tmp22 = std::pow(_tmp8, Scalar(4)) / std::pow(_tmp15, Scalar(2));
  const Scalar _tmp23 = _tmp14 * _tmp22;

  // Output terms (4)
  if (res != nullptr) {
    Eigen::Matrix<Scalar, 1, 1>& _res = (*res);

    _res(0, 0) = _tmp10;
  }

  if (jacobian != nullptr) {
    Eigen::Matrix<Scalar, 1, 3>& _jacobian = (*jacobian);

    _jacobian(0, 0) = _tmp17;
    _jacobian(0, 1) = _tmp19;
    _jacobian(0, 2) = _tmp21;
  }

  if (hessian != nullptr) {
    Eigen::Matrix<Scalar, 3, 3>& _hessian = (*hessian);

    _hessian(0, 0) = std::pow(_tmp14, Scalar(2)) * _tmp22;
    _hessian(0, 1) = 0;
    _hessian(0, 2) = 0;
    _hessian(1, 0) = _tmp18 * _tmp23;
    _hessian(1, 1) = std::pow(_tmp18, Scalar(2)) * _tmp22;
    _hessian(1, 2) = 0;
    _hessian(2, 0) = _tmp20 * _tmp23;
    _hessian(2, 1) = _tmp18 * _tmp20 * _tmp22;
    _hessian(2, 2) = std::pow(_tmp20, Scalar(2)) * _tmp22;
  }

  if (rhs != nullptr) {
    Eigen::Matrix<Scalar, 3, 1>& _rhs = (*rhs);

    _rhs(0, 0) = _tmp10 * _tmp17;
    _rhs(1, 0) = _tmp10 * _tmp19;
    _rhs(2, 0) = _tmp10 * _tmp21;
  }
}

}  // namespace sym
```

SymForce can also generate runtime Python code that depends only on `numpy`. The code generation system is written with pluggable `jinja2` templates to minimize the work to add new backend languages.

## Optimize from C++

Now that we can generate C++ functions for each residual function, we can also run the optimization purely from C++ to get Python entirely out of the loop for production use.

```c++
const int num_poses = 3;
const int num_landmarks = 3;

std::vector<sym::Factor<double>> factors;

// Bearing factors
for (int i = 0; i < num_poses; ++i) {
    for (int j = 0; j < num_landmarks; ++j) {
        factors.push_back(sym::Factor<double>::Hessian(
            &sym::BearingFactor,
            {{'P', i}, {'L', j}, {'a', i, j}, {'e'}},  // keys
            {{'P', i}}  // keys to optimize
        ));
    }
}

// Odometry factors
for (int i = 0; i < num_poses - 1; ++i) {
    factors.push_back(sym::Factor<double>::Hessian(
        &sym::OdometryFactor,
        {{'P', i}, {'P', i + 1}, {'d', i}, {'e'}},  // keys
        {{'P', i}, {'P', i + 1}}  // keys to optimize
    ));
}

sym::Optimizer<double> optimizer(
    params,
    factors,
    sym::kDefaultEpsilon<double>
);

sym::Values<double> values;
for (int i = 0; i < num_poses; ++i) {
    values.Set({'P', i}, sym::Pose2::Identity());
}
... (initialize all keys)

# Optimize!
const auto stats = optimizer.Optimize(&values);

std::cout << "Optimized values:" << values << std::endl;
```

This tutorial shows the central workflow in SymForce for creating symbolic expressions, generating code, and optimizing. This approach works well for a wide range of complex problems in robotics, computer vision, and applied science.


<!-- $
<span style="color:blue">TODO: I wanted to show `geo.V1(sm.atan2(landmark_body[1], landmark_body[0])).jacobian(pose.R)`, but you have to call `sm.simplify` to get the expression to -1, otherwise it's more complicated. All this is also showing up extraneously in the generated code. Discuss what to show.</span>

\frac{
    (-\frac{
        (-R_{im} L_0 + R_{re} L_1 + R_{im} t_0 + R_{re} t_1)^2
    }{
        (R_{re} L_0 + R_{im} L_1 - R_{im} t_1 - R_{re} t_0)^2
    } + \frac{

        }{

        })(R_{re} L_0 + R_{im} L_1 - R_{im} t_1 - R_{re} t_0)^2
    }{
        (-R_{im} L_0 + R_{re} L_1 + R_{im} t_0 + R_{re} t_1)^2 +
        (R_{re} L_0 + R_{im} L_1 - R_{im} t_1 - R_{re} t_0)^2
    }
$ -->

# Learn More

You can find more SymForce tutorials [here](https://symforce-6d87c842-22de-4727-863b-e556dcc9093b.vercel.app/docs/index.html#guides).

# License

SymForce is released under the [BSD-3](https://opensource.org/licenses/BSD-3-Clause) license.

See the LICENSE file for more information.
