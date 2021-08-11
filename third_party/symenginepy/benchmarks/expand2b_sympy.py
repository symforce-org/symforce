from timeit import default_timer as clock
from sympy import ring, ZZ
R, x, y, z, w = ring("x y z w", ZZ)
e = (x+y+z+w)**15
t1 = clock()
f = e*(e+w)
t2 = clock()
#print f
print("Total time:", t2-t1, "s")
print("number of terms:", len(f))
