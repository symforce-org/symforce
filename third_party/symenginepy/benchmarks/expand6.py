#!/usr/bin/env python

import sys
sys.path.append("..")
import os
from timeit import default_timer as clock
if os.environ.get("USE_SYMENGINE"):
    from symengine import symbols, expand
else:
    from sympy import symbols, expand

def run_benchmark(n):
    a0 = symbols("a0")
    a1 = symbols("a1")
    e = a0 + a1
    f = 0;
    for i in range(2, n):
        s = symbols("a%s" % i)
        e = e + s
        f = f + s
    f = -f
    t1 = clock()
    e = expand(e**2)
    e = e.xreplace({a0: f})
    e = expand(e)
    t2 = clock()
    print("%s ms" % (1000 * (t2 - t1)))

if __name__ == '__main__':
    if len(sys.argv) > 1:
        n = int(sys.argv[1])
    else:
        n = 100
    run_benchmark(n)
