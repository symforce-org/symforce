#!/usr/bin/env python
import os
from time import clock
import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr
from sympy.parsing.sympy_parser import standard_transformations
import symengine as se
import warnings

src = os.path.join(os.path.dirname(__file__), '6_links_rhs.txt')
serial = open(src).read()
parsed = parse_expr(serial, transformations=standard_transformations)
vec = sp.Matrix(1, 14, parsed)
args = tuple(sorted(vec.free_symbols, key=lambda arg: arg.name))
exprs = vec, vec.jacobian(args[:-14])
inp = np.ones(len(args))
assert inp.size == 26


lmb_sp = sp.lambdify(args, exprs, modules=['math', 'sympy'])
lmb_se = se.Lambdify(args, *exprs)
lmb_se_llvm = se.Lambdify(args, *exprs, backend='llvm')


lmb_sp(*inp)
tim_sympy = clock()
for i in range(500):
    v, m = lmb_sp(*inp)
tim_sympy = clock() - tim_sympy

lmb_se(inp)
tim_se = clock()
for i in range(500):
    v, m = lmb_se(inp)
tim_se = clock() - tim_se


lmb_se_llvm(inp)
tim_se_llvm = clock()
res_se_llvm = np.empty(len(exprs))
for i in range(500):
    v, m = lmb_se_llvm(inp)
tim_se_llvm = clock() - tim_se_llvm


print('SymEngine (lambda double)       speed-up factor (higher is better) vs sympy: %12.5g' %
      (tim_sympy/tim_se))


print('symengine (LLVM)                speed-up factor (higher is better) vs sympy: %12.5g' %
      (tim_sympy/tim_se_llvm))

import itertools
from functools import reduce
from operator import mul

def ManualLLVM(inputs, *outputs):
    outputs_ravel = list(itertools.chain(*outputs))
    cb = se.Lambdify(inputs, outputs_ravel, backend="llvm")
    def func(*args):
        result = []
        n = np.empty(len(outputs_ravel))
        t = cb.unsafe_real(np.concatenate([arg.ravel() for arg in args]), n)
        start = 0
        for output in outputs:
            elems = reduce(mul, output.shape)
            result.append(n[start:start+elems].reshape(output.shape))
            start += elems
        return result
    return func

lmb_se_llvm_manual = ManualLLVM(args, *exprs)
lmb_se_llvm_manual(inp)
tim_se_llvm_manual = clock()
for i in range(500):
    v, m = lmb_se_llvm_manual(inp)
tim_se_llvm_manual = clock() - tim_se_llvm_manual
print('symengine (ManualLLVM)          speed-up factor (higher is better) vs sympy: %12.5g' %
      (tim_sympy/tim_se_llvm_manual))

if tim_se_llvm_manual < tim_se_llvm:
    warnings.warn("Cython code for Lambdify.__call__ is slow.")
