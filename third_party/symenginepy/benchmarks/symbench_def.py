#!/usr/bin/env python

"""
Benchmarks listed at http://wiki.sagemath.org/symbench can be run using this script
To run the benchmark in symengine, run python symbench.py <benchmark_name>
To run the benchmark in sympy, run python symbench.py sympy <benchmark_name>
"""

from timeit import default_timer as clock
from random import random
import sys

if "sympy" in sys.argv:
    is_sympy = True
    from sympy import sqrt, Integer, var, I, sin, cos
else :
    is_sympy = False
    from symengine import sqrt, Integer, var, I, sin, cos

def R1():
    def f(z):
        return sqrt(Integer(1)/3)*z**2 + I/3
    if (is_sympy):
        t1 = clock()
        g = f(f(f(f(f(f(f(f(f(f(I/2)))))))))).as_real_imag()[0]
        t2 = clock()
    else :
        t1 = clock()
        g = f(f(f(f(f(f(f(f(f(f(I/2)))))))))).expand()
        t2 = clock()
    return t2 - t1

def R2():
    def hermite(n,y):
          if n == 1: return 2*y
          if n == 0: return 1
          return (2*y*hermite(n-1,y) - 2*(n-1)*hermite(n-2,y)).expand()
    t1 = clock()
    hermite(15, var('y'))
    t2 = clock()
    return t2 - t1

def R3():
    var('x y z')
    f = x+y+z
    t1 = clock()
    a = [bool(f==f) for _ in range(10)]
    t2 = clock()
    return t2 - t1

def R5():
    def blowup(L,n):
        for i in range(n):
            L.append( (L[i] + L[i+1]) * L[i+2] )
    def uniq(x):
        v = list(set(x))
        return v
    var('x y z')
    L = [x,y,z]
    blowup(L,8)
    t1 = clock()
    L = uniq(L)
    t2 = clock()
    return t2 - t1

def S1():
    var("x y z")
    e = (x+y+z+1)**7
    f = e*(e+1)
    t1 = clock()
    f = f.expand()
    t2 = clock()
    return t2 - t1

def S2():
    var("x y z")
    e = (x**sin(x) + y**cos(y) + z**(x + y))**100
    t1 = clock()
    f = e.expand()
    t2 = clock()
    return t2 - t1

def S3():
    var("x y z")
    e = (x**y + y**z + z**x)**50
    e = e.expand()
    t1 = clock()
    f = e.diff(x)
    t2 = clock()
    return t2 - t1

def S3a():
    var("x y z")
    e = (x**y + y**z + z**x)**500
    e = e.expand()
    t1 = clock()
    f = e.diff(x)
    t2 = clock()
    return t2 - t1

sys.stdout.write("%15.9f" % locals()[sys.argv[-1]]())
