#!/usr/bin/env python

import sys
sys.path.append("..")
import os
from timeit import default_timer as clock
if os.environ.get("USE_SYMENGINE"):
    from symengine import symbols, sqrt, expand
else:
    from sympy import symbols, sqrt, expand

def run_benchmark(n):
    x, y = symbols("x y")
    e = (1 + sqrt(3) * x + sqrt(5) * y) ** n
    f = e * (e + sqrt(7))
    t1 = clock()
    f = expand(f)
    t2 = clock()
    print("%s ms" % (1000 * (t2 - t1)))

if __name__ == '__main__':
    if len(sys.argv) > 1:
        n = int(sys.argv[1])
    else:
        n = 20
    run_benchmark(n)
