Development Guide
=================

Guide for how to build, configure, and develop SymForce.

.. contents:: :local:

*************************************************
Organization
*************************************************
SymForce aims to follow Python `standards <https://docs.python-guide.org/writing/structure/>`_. The core ``symforce`` package lives in the equivalently named subdirectory at the top level. Tests, documentation, etc live at the top level outside of the core package.
To import ``symforce`` add the top level to the Python path.

See the :ref:`module reference <api-reference>` for the core package structure.

*************************************************
Build
*************************************************
SymForce is primarily written in Python and C++, and is Python 3.8+ and C++14 compatible.  The build
system is CMake for the C++ components, and optionally pip / setuptools on top for Python packaging.
See the Build section on the [Homepage](/index.html#build-from-source) for build instructions.


*************************************************
Additional useful commands
*************************************************
SymForce also has a top level Makefile which is not used by the build, but provides some high
level commands for development:

+----------------------------------------------+--------------------------+
| Run Python tests                             | ``make test``            |
+----------------------------------------------+--------------------------+
| Run tests which update (most) generated code | ``make test_update``     |
+----------------------------------------------+--------------------------+
| Run tests which update all generated code    | ``make test_update_all`` |
+----------------------------------------------+--------------------------+
| Run tests and open coverage report           | ``make coverage_open``   |
+----------------------------------------------+--------------------------+
| Build docs                                   | ``make docs``            |
+----------------------------------------------+--------------------------+
| Build docs + open in browser                 | ``make docs_open``       |
+----------------------------------------------+--------------------------+
| Run the code formatter (black, clang-format) | ``make format``          |
+----------------------------------------------+--------------------------+
| Check types with mypy                        | ``make check_types``     |
+----------------------------------------------+--------------------------+
| Check formatting and types                   | ``make lint``            |
+----------------------------------------------+--------------------------+

*************************************************
Documentation
*************************************************
This documentation is built with `Sphinx <https://www.sphinx-doc.org/>`_, including automatic parsing of the code to generate a module reference using `sphinx-apidoc <https://www.sphinx-doc.org/en/master/man/sphinx-apidoc.html>`_. The code uses `Google Style <https://www.sphinx-doc.org/en/1.6/ext/example_google.html>`_ docstrings to annotate all modules, classes, and functions. Docs pages are ``.rst`` files in ``docs``, and the Sphinx config file is ``docs/conf.py``.

There are sample `Jupyter <https://jupyter.org/>`_ notebooks in ``notebooks``, some of which are built into these docs using `nbsphinx <https://nbsphinx.readthedocs.io/en/0.5.1/>`_, such as the `tutorial <notebooks/tutorial.html>`_. ``make notebook`` starts a Jupyter server to modify and run them.

*************************************************
Logging
*************************************************
SymForce uses the `logging <https://docs.python.org/2/library/logging.html>`_ module. You can import and use the logger like this:

>>> from symforce import logger
>>> logger.warning('houston, we have a problem')
codegen_test.test_codegen_cpp():126 WARNING -- houston, we have a problem

You can configure the log level using :func:`symforce.set_log_level()` or by setting the ``SYMENGINE_LOGLEVEL`` environment variable. The default is ``logging.INFO``.

*************************************************
Testing and Coverage
*************************************************
SymForce is heavily tested, targeting close to 100% code coverage.
Tests live in ``test`` and use `unittest <https://docs.python.org/2/library/unittest.html>`_. Additionally, `coverage.py <https://coverage.readthedocs.io/en/coverage-5.0.4/>`_ is used to run tests while measuring code coverage. The generated coverage report also provides a great view into what methods need to be tested and what code is potentially unused.

| Run all tests: ``make test``
| Run all tests and open the coverage report: ``make coverage_open``
| Run a specific test: ``python test/symforce_codegen_test.py``
| Run with debug level output: ``SYMFORCE_LOGLEVEL=DEBUG python test/symforce_codegen_test.py``

When debugging a specific test, the use of `ipdb <https://pypi.org/project/ipdb/>`_ is highly recommended, as is reproducing the most minimal example of the issue in a notebook.

*************************************************
Formatting
*************************************************
Symforce uses the `Black <https://github.com/psf/black>`_ formatter for Python code. To quote the authors:

    `Black is the uncompromising Python code formatter. By using it, you agree to cede control over minutiae of hand-formatting. In return, Black gives you speed, determinism, and freedom from nagging about formatting. You will save time and mental energy for more important matters.`

Running ``make format`` will format the entire codebase. It's recommended to develop with `VSCode <https://code.visualstudio.com/>`_ and integrate black.

*************************************************
Templates
*************************************************
Much of the core functionality of SymForce is in generating code using the `Jinja <https://jinja.palletsprojects.com/en/2.11.x/>`_ template language. It's relatively simple and easy to use - you pass it a template file in any language and a python dictionary of data, and it spits out the rendered code.

For example template files, see ``symforce/codegen/backends/cpp/templates``.

*************************************************
Symbolic API
*************************************************
SymForce uses the `SymPy <https://www.sympy.org/en/index.html>`_ API, but supports two implementations of it. The SymPy implementation is pure Python, whereas the `SymEngine <https://github.com/symengine/symengine>`_ implementation is wrapped C++. It can be 100-200 times faster for many operations, but is less fully featured and requires a C++ build.

To set the symbolic API, you can either use :func:`symforce.set_symbolic_api()` before any other imports, or use the ``SYMFORCE_SYMBOLIC_API`` environment variable with the options ``sympy`` or ``symengine``. By default SymEngine will be used if found, otherwise SymPy.

*************************************************
Building wheels
*************************************************

You should be able to build Python wheels of symforce the standard ways.  We recommend using
``build``, i.e. running ``python3 -m build --wheel`` from the ``symforce`` directory.  By default,
this will build a wheel that includes local dependencies on the ``skymarshal`` and ``symforce-sym``
packages (which are separate Python packages from ``symforce`` itself).  For distribution, you'll
typically want to set the environment variable ``SYMFORCE_REWRITE_LOCAL_DEPENDENCIES=True`` when
building, and also run ``python3 -m build --wheel third_party/skymarshal`` and
``python3 -m build --wheel gen/python`` to build wheels for those packages separately.

For SymForce releases, all of this is handled by the ``build_wheels`` GitHub Actions workflow.  This
workflow is currently run manually on a commit, and produces a ``symforce-wheels.zip`` artifact with
wheels (and sdists) for distribution (e.g. on PyPI).  It doesn't upload them to PyPI - to do that
(after verifying that the built wheels work as expected) you should download and unzip the archive,
and upload to PyPI with ``python -m twine upload [--repository testpypi] --verbose *``.
