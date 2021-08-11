# SymEngine Python Wrappers

Python wrappers to the C++ library [SymEngine](https://github.com/symengine/symengine),
a fast C++ symbolic manipulation library.

[![Build Status](https://travis-ci.org/symengine/symengine.py.svg)](https://travis-ci.org/symengine/symengine.py) [![Build status](https://ci.appveyor.com/api/projects/status/sl189l9ck3gd8qvk/branch/master?svg=true)](https://ci.appveyor.com/project/symengine/symengine-py/branch/master)

## Installation

### Pip

See License section for information about wheels

    pip install symengine --user

### Conda package manager

    conda install python-symengine -c symengine -c conda-forge

optionally, you may choose to install an early [developer preview](https://github.com/symengine/python-symengine-feedstock):

    conda install python-symengine -c symengine/label/dev -c conda-forge

### Build from source

Install prerequisites.

    CMake       >= 2.8.7
    Python2     >= 2.7      or Python3 >= 3.4
    Cython      >= 0.19.1
    SymEngine   >= 0.4.0

For SymEngine, only a specific commit/tag (see symengine_version.txt) is supported.
Latest git master branch may not work as there may be breaking changes in SymEngine.

Python wrappers can be installed by,

    python setup.py install

Additional options to setup.py are

    python setup.py install build_ext
        --symengine-dir=/path/to/symengine/install/dir          # Path to SymEngine install directory or build directory
        --compiler=mingw32|msvc|cygwin                          # Select the compiler for Windows
        --generator=cmake-generator                             # CMake Generator
        --build-type=Release|Debug                              # Set build-type for multi-configuration generators like MSVC
        --define="var1=value1;var2=value2"                      # Give options to CMake
        --inplace                                               # Build the extension in source tree

Standard options to setup.py like `--user`, `--prefix` can be used to
configure install location. NumPy is used if found by default, if you wish
to make your choice of NumPy use explicit: then add
e.g. ``WITH_NUMPY=False`` to ``--define``.

Use SymEngine from Python as follows:

    >>> from symengine import var
    >>> var("x y z")
    (x, y, z)
    >>> e = (x+y+z)**2
    >>> e.expand()
    2*x*y + 2*x*z + 2*y*z + x**2 + y**2 + z**2

You can read Python tests in `symengine/tests` to see what features are
implemented.


## License

symengine.py is MIT licensed and uses several LGPL, BSD-3 and MIT licensed libraries

Licenses for the dependencies of pip wheels are as follows,

pip wheels on Unix use GMP (LGPL-3.0-or-later), MPFR (LGPL-3.0-or-later),
MPC (LGPL-3.0-or-later), LLVM (Apache-2.0), zlib (Zlib) and symengine (MIT AND BSD-3-Clause).
pip wheels on Windows use MPIR (LGPL-3.0-or-later) instead of GMP above and
pthreads-win32 (LGPL-3.0-or-later) additionally.
NumPy (BSD-3-Clause) and SymPy (BSD-3-Clause) are optional dependencies.
Sources for these binary dependencies can be found on https://github.com/symengine/symengine-wheels/releases

