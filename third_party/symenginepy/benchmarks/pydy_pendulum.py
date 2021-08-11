import os
import time
import sys
sys.path = ["../sympy", "../pydy", "../symengine.py"] + sys.path

import sympy
import symengine
import pydy
from sympy.physics.mechanics.models import n_link_pendulum_on_cart

print(sympy.__file__)
print(symengine.__file__)
print(pydy.__file__)

if (len(sys.argv) > 1):
    n = int(sys.argv[1])
else:
    n = 4

start = time.time()
sys = n_link_pendulum_on_cart(n, cart_force=False)
end = time.time()

print("%s s" % (end-start))

#print(sys.eom_method.mass_matrix)
