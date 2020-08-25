Development
===========
Guide for how to build, configure, and develop symforce.

.. contents:: :local:

*************************************************
Organization
*************************************************
Symforce aims to follow Python `standards <https://docs.python-guide.org/writing/structure/>`_. The core symforce package lives in the equivalently named subdirectory at the top level. Tests, documentation, etc live at the top level outside of the core package.
To import symforce add the top level to the Python path.

See the :ref:`module reference <api-reference>` for the core package structure.

*************************************************
Build
*************************************************
Symforce is primarily written in Python, aimed to be 2.7+ compatible. It has a top level Makefile to execute high level commands:

+---------------------------------------+-------------------------+
| Install requirements                  | ``make all_reqs``       |
+---------------------------------------+-------------------------+
| Run tests                             | ``make test``           |
+---------------------------------------+-------------------------+
| Run tests which update generated code | ``make test_update``    |
+---------------------------------------+-------------------------+
| Run tests and open coverage report    | ``make coverage_open``  |
+---------------------------------------+-------------------------+
| Build docs                            | ``make docs``           |
+---------------------------------------+-------------------------+
| Build docs + open in browser          | ``make docs_open``      |
+---------------------------------------+-------------------------+
| Launch Jupyter server                 | ``make notebook``       |
+---------------------------------------+-------------------------+
| Launch Jupyter server + browser       | ``make notebook_open``  |
+---------------------------------------+-------------------------+
| Run the code formatter (black)        | ``make format``         |
+---------------------------------------+-------------------------+
| Check types with mypy                 | ``make check_types``    |
+---------------------------------------+-------------------------+
| Check formatting and types            | ``make lint``           |
+---------------------------------------+-------------------------+
| Clean all build products              | ``make clean``          |
+---------------------------------------+-------------------------+

Note, docs may require a manual `pandoc <https://pandoc.org/>`_ install - ``sudo apt-get pandoc`` or ``brew install pandoc``.

*************************************************
Documentation
*************************************************
This documentation is built with `Sphinx <https://www.sphinx-doc.org/>`_, including automatic parsing of the code to generate a module reference using `sphinx-apidoc <https://www.sphinx-doc.org/en/master/man/sphinx-apidoc.html>`_. The code uses `Google Style <https://www.sphinx-doc.org/en/1.6/ext/example_google.html>`_ docstrings to annotate all modules, classes, and functions. Docs pages are ``.rst`` files in ``docs``, and the Sphinx config file is ``docs/conf.py``.

There are sample `Jupyter <https://jupyter.org/>`_ notebooks in ``notebooks``, some of which are built into these docs using `nbsphinx <https://nbsphinx.readthedocs.io/en/0.5.1/>`_, such as the `tutorial <notebooks/tutorial.html>`_. ``make notebook`` starts a Jupyter server to modify and run them.

*************************************************
Logging
*************************************************
Symforce uses the `logging <https://docs.python.org/2/library/logging.html>`_ module. You can import and use the logger like this:

>>> from symforce import logger
>>> logger.warning('houston, we have a problem')
codegen_test.test_codegen_cpp():126 WARNING -- houston, we have a problem

You can configure the log level using :func:`symforce.set_log_level()` or by setting the ``SYMENGINE_LOGLEVEL`` environment variable. The default is ``logging.INFO``.

*************************************************
Testing and Coverage
*************************************************
Symforce is heavily tested, targeting close to 100% code coverage.
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
Much of the core functionality of symforce is in generating code using the `Jinja <https://jinja.palletsprojects.com/en/2.11.x/>`_ template language. It's relatively simple and easy to use - you pass it a template file in any language and a python dictionary of data, and it spits out the rendered code.

For example template files, see ``symforce/codegen/cpp_templates``.

*************************************************
Symbolic Backends
*************************************************
Symforce uses the `Sympy <https://www.sympy.org/en/index.html>`_ API, but supports two backend implementations of it. The Sympy backend is pure Python, whereas the `SymEngine <https://github.com/symengine/symengine>`_ backend is wrapped C++. It can be 100-200 times faster for many operations, but is less fully featured and requires a C++ build.

To set the backend, you can either use :func:`symforce.set_backend()` before any other imports, or use the ``SYMFORCE_BACKEND`` environment variable with the options ``sympy`` or ``symengine``. By default symengine will be used if found, otherwise sympy.
