SymForce is a library for applied symbolic computation.

It provides tools to accelerate real-world robotics through rapid prototyping and analysis
of symbolic math coupled with generation into extremely fast runtime code. It's been battle
hardened by Skydio in production spanning embedded control, clock synchronization,
large sparse nonlinear MPC, and bundle adjustment.

SymForce is built on the `SymPy <https://www.sympy.org/>`_ API with a custom geometry package that
implements rigorous concepts of 2D and 3D transformations. Input/output API specifications of
symbolic quantities are generated into executable code packages in Python or C++.

Benefits:

* One true symbolic implementation
* Consistent but native across multiple languages
* Minimal dependencies + static allocation = good for embedded uses
* Generated C++ supports floats/doubles
* Get correct (and fast) derivatives for everything
* Generated code is typically faster, can exceed 10x fewer flops than autodiff
* Much less prone to bugs
* Uniform and immutable geometric types as first-class citizens
* Flexible to mix generated and handwritten code
