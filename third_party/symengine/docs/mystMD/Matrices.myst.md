---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.10.0
kernelspec:
  display_name: C++11
  language: C++11
  name: xcpp11
---

# Matrices

```{code-cell}
#include <chrono>
#include <xcpp/xdisplay.hpp>

#include <symengine/matrix.h>
#include <symengine/add.h>
#include <symengine/pow.h>
#include <symengine/symengine_exception.h>
#include <symengine/visitor.h>
```

```{code-cell}
SymEngine::vec_basic elems{SymEngine::integer(1),
                           SymEngine::integer(0),
                           SymEngine::integer(-1),
                           SymEngine::integer(-2)};
SymEngine::DenseMatrix A = SymEngine::DenseMatrix(2, 2, elems);
```

```{code-cell}
A.__str__()
```
