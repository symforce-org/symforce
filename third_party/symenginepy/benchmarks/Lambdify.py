#!/usr/bin/env python
from time import clock
import numpy as np
import sympy as sp
import symengine as se
import warnings

# Real-life example (ion speciation problem in water chemistry)

x = sp.symarray('x', 14)
p = sp.symarray('p', 14)
args = np.concatenate((x, p))
exp = sp.exp
exprs = [x[0] + x[1] - x[4] + 36.252574322669, x[0] - x[2] + x[3] + 21.3219379611249, x[3] + x[5] - x[6] + 9.9011158998744, 2*x[3] + x[5] - x[7] + 18.190422234653, 3*x[3] + x[5] - x[8] + 24.8679190043357, 4*x[3] + x[5] - x[9] + 29.9336062089226, -x[10] + 5*x[3] + x[5] + 28.5520551531262, 2*x[0] + x[11] - 2*x[4] - 2*x[5] + 32.4401680272417, 3*x[1] - x[12] + x[5] + 34.9992934135095, 4*x[1] - x[13] + x[5] + 37.0716199972041, p[0] - p[1] + 2*p[10] + 2*p[11] - p[12] - 2*p[13] + p[2] + 2*p[5] + 2*p[6] + 2*p[7] + 2*p[8] + 2*p[9] - exp(x[0]) + exp(x[1]) - 2*exp(x[10]) - 2*exp(x[11]) + exp(x[12]) + 2*exp(x[13]) - exp(x[2]) - 2*exp(x[5]) - 2*exp(x[6]) - 2*exp(x[7]) - 2*exp(x[8]) - 2*exp(x[9]), -p[0] - p[1] - 15*p[10] - 2*p[11] - 3*p[12] - 4*p[13] - 4*p[2] - 3*p[3] - 2*p[4] - 3*p[6] - 6*p[7] - 9*p[8] - 12*p[9] + exp(x[0]) + exp(x[1]) + 15*exp(x[10]) + 2*exp(x[11]) + 3*exp(x[12]) + 4*exp(x[13]) + 4*exp(x[2]) + 3*exp(x[3]) + 2*exp(x[4]) + 3*exp(x[6]) + 6*exp(x[7]) + 9*exp(x[8]) + 12*exp(x[9]), -5*p[10] - p[2] - p[3] - p[6] - 2*p[7] - 3*p[8] - 4*p[9] + 5*exp(x[10]) + exp(x[2]) + exp(x[3]) + exp(x[6]) + 2*exp(x[7]) + 3*exp(x[8]) + 4*exp(x[9]), -p[1] - 2*p[11] - 3*p[12] - 4*p[13] - p[4] + exp(x[1]) + 2*exp(x[11]) + 3*exp(x[12]) + 4*exp(x[13]) + exp(x[4]), -p[10] - 2*p[11] - p[12] - p[13] - p[5] - p[6] - p[7] - p[8] - p[9] + exp(x[10]) + 2*exp(x[11]) + exp(x[12]) + exp(x[13]) + exp(x[5]) + exp(x[6]) + exp(x[7]) + exp(x[8]) + exp(x[9])]

lmb_sp = sp.lambdify(args, exprs, modules='math')
lmb_se = se.Lambdify(args, exprs)
lmb_se_cse = se.LambdifyCSE(args, exprs)
lmb_se_llvm = se.Lambdify(args, exprs, backend='llvm')

inp = np.ones(28)

lmb_sp(*inp)
tim_sympy = clock()
for i in range(500):
    res_sympy = lmb_sp(*inp)
tim_sympy = clock() - tim_sympy

lmb_se(inp)
tim_se = clock()
res_se = np.empty(len(exprs))
for i in range(500):
    res_se = lmb_se(inp)
tim_se = clock() - tim_se

lmb_se_cse(inp)
tim_se_cse = clock()
res_se_cse = np.empty(len(exprs))
for i in range(500):
    res_se_cse = lmb_se_cse(inp)
tim_se_cse = clock() - tim_se_cse

lmb_se_llvm(inp)
tim_se_llvm = clock()
res_se_llvm = np.empty(len(exprs))
for i in range(500):
    res_se_llvm = lmb_se_llvm(inp)
tim_se_llvm = clock() - tim_se_llvm


print('SymEngine (lambda double)       speed-up factor (higher is better) vs sympy: %12.5g' %
      (tim_sympy/tim_se))

print('symengine (lambda double + CSE) speed-up factor (higher is better) vs sympy: %12.5g' %
      (tim_sympy/tim_se_cse))

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

lmb_se_llvm_manual = ManualLLVM(args, np.array(exprs))
lmb_se_llvm_manual(inp)
tim_se_llvm_manual = clock()
res_se_llvm_manual = np.empty(len(exprs))
for i in range(500):
    res_se_llvm_manual = lmb_se_llvm_manual(inp)
tim_se_llvm_manual = clock() - tim_se_llvm_manual
print('symengine (ManualLLVM)          speed-up factor (higher is better) vs sympy: %12.5g' %
      (tim_sympy/tim_se_llvm_manual))

if tim_se_llvm_manual < tim_se_llvm:
    warnings.warn("Cython code for Lambdify.__call__ is slow.")

import setuptools
import pyximport
pyximport.install()
from Lambdify_reference import _benchmark_reference_for_Lambdify as lmb_ref

lmb_ref(inp)
tim_ref = clock()
for i in range(500):
    res_ref = lmb_ref(inp)
tim_ref = clock() - tim_ref
print('Hard-coded Cython code          speed-up factor (higher is better) vs sympy: %12.5g' %
      (tim_sympy/tim_ref))
