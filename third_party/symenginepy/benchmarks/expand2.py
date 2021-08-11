#!/usr/bin/env python

import sys
sys.path.append("..")
import os
from timeit import default_timer as clock
if os.environ.get("USE_SYMENGINE"):
    from symengine import var
else:
    from sympy import var

def run_benchmark(n):
    var("x y z w")
    e = (x + y + z + w)**n
    f = e * (e + w)
    t1 = clock()
    g = f.expand()
    t2 = clock()
    print("%s ms" % (1000 * (t2 - t1)))

if __name__ == '__main__':
    if len(sys.argv) > 1:
        n = int(sys.argv[1])
    else:
        n = 15
    run_benchmark(n)
