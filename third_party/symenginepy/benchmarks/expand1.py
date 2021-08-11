import sys
sys.path.append("..")
from timeit import default_timer as clock
from symengine import var
var("x y z w")
e = (x+y+z+w)**60
t1 = clock()
g = e.expand()
t2 = clock()
print("Total time:", t2-t1, "s")
