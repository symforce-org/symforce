print("import...")
from timeit import default_timer as clock
from sage.all import var
var("x")
e = 1
print("constructing expression...")
for i in range(1, 351):
    e *= (i+x)**3
print("running benchmark...")
t1 = clock()
f = e.expand()
t2 = clock()
print("Total time:", t2-t1, "s")
