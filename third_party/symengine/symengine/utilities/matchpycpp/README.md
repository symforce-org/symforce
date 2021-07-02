# MatchPyCpp library.

This module is a partial translation into C++ of
[MatchPy](https://github.com/HPAC/matchpy).  It provides tools to generate
decision trees out of MatchPy patterns defined using SymPy expressions.  In
particular, the generated code represents the decision tree of a many-to-one
matcher of MatchPy, which is an object able to perform many
associative-commutative matchings at the same time.

The class `CppCodeGenerator` in `cpp_code_generation.py` is responsible for
generating C++ decision trees, mimicking MatchPy's `CodeGenerator` class.

## Usage example

Clone MatchPy and checkout some past commit (the current wrapper in SymPy is not compatible with the latest MatchPy commit):

```bash
git clone https://github.com/HPAC/matchpy
cd matchpy
git checkout 419c103
```

Make sure that MatchPy is in the `PYTHONPATH`.

Import the MatchPy wrapper from SymPy:

```python
from sympy.integrals.rubi.utility_function import *
from sympy.integrals.rubi.symbol import WC
```

Note: the `utility_function` module needs to be imported as the link between SymPy and MatchPy expressions is defined there.

Import MatchPy and create a many-to-one matcher object:

```python
from matchpy import *

matcher = ManyToOneMatcher()
```

Define SymPy variables and a MatchPy-SymPy wilcard `w`:

```python
w = WC("w")
x, y, z = symbols("x y z")
```

Add some expressions to the many-to-one matcher:

```python
matcher.add(Pattern(x+y))
matcher.add(Pattern(2**w))
```

Use MatchPy to match an expression:

```python
>>> list(matcher.match(2**z))
[(Pattern(2**w), {'w': z})]
```

which means that pattern `2**w` matches the expression `2**z` with substitution `{'w': z}`.

The C++ code performing the same matching can be generated with:

```python
from cpp_code_generation import CppCodeGenerator
cg = CppCodeGenerator(matcher)
part1, part2 = cg.generate_code()
```

Now write `part1` and `part2` to a file:

```python
fout = open("sample_matching_test.cpp", "w")
fout.write(part1)
fout.write(part2)
fout.close()
```
