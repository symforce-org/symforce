print("import...")
from timeit import default_timer as clock
from sage.all import var, Integer
print("    done.")

def fact(n):
    if n in [0, 1]:
        return 1
    else:
        return n*fact(n-1)

def diff(e, x, n):
    for i in range(n):
        e = e.diff(x)
    return e

def legendre(n, x):
    e = Integer(1)/(Integer(2)**n * fact(Integer(n))) * diff((x**2-1)**n, x, n)
    return e.expand()

var("x")
for n in range(10):
    print(n, legendre(n, x))

t1 = clock()
e = legendre(500, x)
t2 = clock()
print("Total time for legendre(500, x):", t2-t1, "s")
