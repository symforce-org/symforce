# C++ Style Guide

Please follow these guidelines when submitting patches.

## Whitespace

Use 4 spaces. Format ``if`` as follows:

```cpp
if (d.find(t) == d.end()) {
    d[t] = coef;
} else {
    d[t] = d[t] + coef;
}
```

## Pointers

Never use raw C++ pointers and never use the raw `new` and `delete`. Always use
the smart pointers provided by `symengine_rcp.h` (depending on
WITH_SYMENGINE_RCP, those are either Teuchos::RCP, or our own, faster
implementation), i.e. `Ptr` and `RCP` (and do not use the `.get()` method, only
the `.ptr()` method). In Debug mode, the pointers are 100% safe, i.e. no matter
how you use them the code will not segfault, but instead raise a nice exception
if a pointer becomes dangling or null. In Release mode, the `Ptr` is as fast as
a raw pointer and `RCP` is a lot faster, but no checks are done (so the code
can segfault).

### Declaration

In the `.cpp` files you can declare:

```cpp
using SymEngine::RCP;
using SymEngine::Ptr;
using SymEngine::outArg;
using SymEngine::make_rcp;
using SymEngine::rcp_dynamic_cast;
```
    
and then just use `RCP` or `Ptr`.

In the `.h` header files use the full name like `SymEngine::RCP` or `SymEngine::Ptr`.

### Initialization

Initialize as follows:

```cpp
RCP<Basic> x  = make_rcp<Symbol>("x");
```

Never call the naked `new`, nor use the naked `rcp`. If available, use the
factory functions, e.g. in this case `symbol()` as follows:

```cpp
RCP<Basic> x  = symbol("x");
```

This does the same thing (internally it calls `make_rcp`), but it is easier to
use.

### Freeing

The `RCP` pointer is released automatically. You never call the naked `delete`.

### Passing To/From Functions

Use C++ references for objects that you are **not** passing around. If the object
is *not* modified, use `const A &a`:

```cpp
RCP<const Integer> gcd(const Integer &a, const Integer &b)
{
    integer_class g;
    mp_gcd(g, a.as_integer_class(), b.as_integer_class());
    return integer(std::move(g));
}
```

If it *is* modified, use `A &a` (see the first argument):

```cpp
void Add::dict_add_term(umap_basic_num &d, const RCP<Integer> &coef,
        const RCP<Basic> &t)
{
    if (d.find(t) == d.end()) {
        d[t] = coef;
    } else {
        d[t] = d[t] + coef;
    }
}
```

If the objects **are** passed around, you have to use `RCP`. You also need to
use `RCP` whenever you call some function that uses `RCP`.

Declare functions with two input arguments (and one return value) as follows:

```cpp
RCP<const Basic> multiply(const RCP<const Basic> &a,
        const RCP<const Basic> &b)
{
    ...
    return make_rcp<const Integer>(1);
}
```

Functions with one input and two output arguments are declared:

```cpp
void as_coef_term(const RCP<const Basic> &self,
    const Ptr<RCP<const Integer>> &coef,
    const Ptr<RCP<const Basic>> &term)
{
    ...
    *coef = make_rcp<const Integer>(1);
    *term = self
    ...
}
```

and used as follows:

```cpp
RCP<Integer> coef;
RCP<Basic> t;
as_coef_term(b, outArg(coef), outArg(t));
```

`SymEngine` objects are always immutable, so you always declare them as `const`.
And `RCP` is only used with `SymEngine`'s objects, so you always use `const
RCP<const Integer> &i`. But if the `Integer` was somehow mutable (it's not in
`SymEngine`), you would use `const RCP<Integer> &i`.

For returning objects from functions, simply declare the return type as `RCP<const Basic>` as shown above.

### Casting

You can use dynamic cast as follows:

```cpp
RCP<Basic> tmp;
RCP<Integer> coef;
coef = rcp_dynamic_cast<Integer>(tmp);
```

## Namespaces

Never use "implicit imports": ``using namespace std;``.

In cpp files, either use the full name of the symbol (e.g. ``SymEngine::RCP``),
or use "explicit import" as follows: ``using SymEngine::RCP;``.

In header files, always use the full name (never import symbols there).
