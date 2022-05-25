SymForce Home
=============

.. include:: ../README.md
   :parser: myst_parser.sphinx_

Guides
======

:doc:`development`
    How to build, configure, and develop

:doc:`tutorials/sympy_tutorial`
    Basic introduction to SymPy

:doc:`tutorials/geometry_tutorial`
    Introductory guide to doing math and geometry

:doc:`tutorials/ops_tutorial`
    Introductory guide to using Ops in symforce

:doc:`tutorials/cameras_tutorial`
    Introductory guide to using camera models

:doc:`tutorials/values_tutorial`
    How to structure large groups of symbols and expressions

:doc:`tutorials/codegen_tutorial`
    How to generate functions from symbolic expressions

:doc:`tutorials/optimization_tutorial`
    Basic example of using generated code to do optimization

:doc:`tutorials/epsilon_tutorial`
    Guide to how Epsilon is used to prevent singularities

.. toctree::
   :hidden:

   development

.. toctree::
   :caption: Tutorials
   :hidden:

   tutorials/sympy_tutorial
   tutorials/geometry_tutorial
   tutorials/ops_tutorial
   tutorials/cameras_tutorial
   tutorials/values_tutorial
   tutorials/codegen_tutorial
   tutorials/optimization_tutorial
   tutorials/epsilon_tutorial

.. toctree::
   :caption: Examples
   :hidden:
   :glob:
   :titlesonly:

   examples/*/README

.. _api-reference:
.. toctree::
    :hidden:
    :maxdepth: 2
    :titlesonly:
    :caption: symforce Reference

    api/symforce

.. _genpy-api-reference:
.. toctree::
    :maxdepth: 2
    :hidden:
    :titlesonly:
    :caption: sym Python Reference

    api-gen-py/sym

.. _gencpp-api-reference:
.. toctree::
    :maxdepth: 2
    :hidden:
    :titlesonly:
    :caption: sym C++ Reference

    api-gen-cpp/classlist

    api-gen-cpp/filelist

.. _cpp-api-reference:
.. toctree::
    :hidden:
    :maxdepth: 2
    :titlesonly:
    :caption: opt C++ Reference

    api-cpp/classlist

    api-cpp/filelist
