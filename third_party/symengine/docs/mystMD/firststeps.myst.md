---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.10.0
kernelspec:
  display_name: C++17
  language: C++17
  name: xcpp17
---

# First Steps [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/symengine/symengine.github.io/sources?filepath=docs%2Fuse%2Fapi%2Fcpp%2Ffirststeps.myst.md)

This is meant to be a gentle introduction to the `symengine` C++ library.

## Working with Expressions

We will start by inspecting the use of {ref}`Expression <cpp_api:class_sym_engine_1_1_expression>`.

```{code-cell}
#include <symengine/expression.h>
using SymEngine::Expression;
```

```{code-cell}
Expression x("x");
```

```{code-cell}
auto ex = pow(x+sqrt(Expression(2)), 6);
ex
```

```{code-cell}
expand(ex)
```

```{reviewer-meta}
:written-on: "2020-08-27"
:proofread-on: "2021-01-20"
:dust-days-limit: 60
```
