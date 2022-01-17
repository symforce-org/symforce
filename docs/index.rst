SymForce Docs
=============

.. include:: ../README.md
   :parser: myst_parser.sphinx_

Guides
======

.. toctree::
   :caption: Pages
   :hidden:

   self
   development
   notebooks/sympy_tutorial
   notebooks/geometry_tutorial
   notebooks/ops_tutorial
   notebooks/cameras_tutorial
   notebooks/values_tutorial
   notebooks/codegen_tutorial
   notebooks/optimization_tutorial

:doc:`development`
    How to build, configure, and develop

:doc:`notebooks/sympy_tutorial`
    Basic introduction to SymPy

:doc:`notebooks/geometry_tutorial`
    Introductory guide to doing math and geometry

:doc:`notebooks/ops_tutorial`
    Introductory guide to using Concepts in symforce

:doc:`notebooks/cameras_tutorial`
    Introductory guide to using camera models

:doc:`notebooks/values_tutorial`
    How to structure large groups of symbols and expressions

:doc:`notebooks/codegen_tutorial`
    How to generate functions from symbolic expressions

:doc:`notebooks/optimization_tutorial`
    Basic example of using generated code to do optimization

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
