import sys
sys.path.append("..")
from timeit import default_timer as clock
from symengine import var
var("x y z")
f = (x**y + y**z + z**x)**100
print(f)
t1 = clock()
g = f.expand()
t2 = clock()
print("Total time:", t2-t1, "s")
