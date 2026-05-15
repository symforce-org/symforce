from timeit import default_timer as clock
from sage.all import var
var("x y z")
e = (x+y+z+1)**15
f = e*(e+1)
print(f)
t1 = clock()
g = f.expand()
t2 = clock()
print("Total time:", t2-t1, "s")
