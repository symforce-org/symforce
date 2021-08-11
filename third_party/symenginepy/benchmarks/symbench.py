#!/usr/bin/env python

"""
Benchmarks listed at http://wiki.sagemath.org/symbench can be run using this script
To run all benchmarks using symengine, sympy and sage, python symbench.py sympy sage
"""

import sys
import subprocess

benchmarks = ['R1', 'R2', 'R3', 'R5', 'S1', 'S2', 'S3', 'S3a']
symengine_skip = [False, False, False, False, False, False, False, False]
sympy_skip = [False, False, False, False, False, False, False, True]
sage_skip = [False, False, False, False, False, False, False, False]

args = sys.argv[1:]
sympy = False
sage = False
if "sympy" in sys.argv:
    sympy = True
if "sage" in sys.argv:
    sage = True
ws = "\t\t\t\t"

for i in range(len(benchmarks)):
    benchmark = benchmarks[i]
    sys.stdout.write("Time for " + benchmark)
    a = None
    b = None
    c = None
    if not symengine_skip[i]:
        a = subprocess.check_output(['python', 'symbench_def.py', benchmark])
        a = "\t SymEngine : " + a + " s"
    sys.stdout.write(a or ws)
    if sympy:
        if not sympy_skip[i]:
            b = subprocess.check_output(['python', 'symbench_def.py', 'sympy', benchmark])
            b = "\t SymPy  : " + b + " s"
        sys.stdout.write(b or ws)
    if sage:
        if not sage_skip[i]:
            c = subprocess.check_output(['sage', 'symbench_sage.py', benchmark])
            c = "\t Sage   : " + c + " s"
        sys.stdout.write(c or ws)
    print("")
