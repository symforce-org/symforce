#!/usr/bin/env sage

"""
Benchmarks listed at http://wiki.sagemath.org/symbench can be run using this script
To run the benchmark in symengine, run python symbench.py <benchmark_name>
To run the benchmark in sympy, run python symbench.py sympy <benchmark_name>
"""

from timeit import default_timer as clock
from sage.all import *

def R1():
    def f(z): return sqrt(1/3)*z**2 + i/3
    t1 = clock()
    a = real(f(f(f(f(f(f(f(f(f(f(I/2)))))))))))
    t2 = clock()
    return t2 - t1

def R2():
    def hermite(n,y):
        if n == 1: return 2*y
        if n == 0: return 1
        return expand(2*y*hermite(n-1,y) - 2*(n-1)*hermite(n-2,y))
    t1 = clock()
    hermite(15,var('y'))
    t2 = clock()
    return t2 - t1

def R3():
    var('x,y,z')
    f = x+y+z
    t1 = clock()
    a = [bool(f==f) for _ in range(10)]
    t2 = clock()
    return t2 - t1

def R5():
    def blowup(L,n):
        for i in range(n):
            L.append( (L[i] + L[i+1]) * L[i+2] )
    (x, y, z)=var('x,y,z')
    L = [x, y, z]
    blowup(L, 8)
    t1 = clock()
    L=uniq(L)
    t2 = clock()
    return t2 - t1

def S1():
    var('x,y,z')
    f = (x+y+z+1)**7
    t1 = clock()
    g = expand(f*(f+1))
    t2 = clock()
    return t2 - t1

def S2():
    var('x,y,z')
    t1 = clock()
    a = expand((x**sin(x) + y**cos(y) - z**(x+y))**100)
    t2 = clock()
    return t2 - t1

def S3():
    var('x,y,z')
    f = expand((x**y + y**z + z**x)**50)
    t1 = clock()
    g = f.diff(x)
    t2 = clock()
    return t2 - t1

def S3a():
    var('x,y,z')
    f = expand((x**y + y**z + z**x)**500)
    t1 = clock()
    g = f.diff(x)
    t2 = clock()
    return t2 - t1

sys.stdout.write("%15.9f" % locals()[sys.argv[-1]]())
