#!/bin/bash maple
# Use `maple -q expand6.mpl -D n=100` to run

e := a0 + a1:
f := 0:

for i from 2 to (n - 1)
do
    f := f + cat(a, convert(i, string)):
    e := e + cat(a, convert(i, string)):
end do:

f := -f:
e := e ^ 2:

st := time[real]():
e := expand(e):
e := subs(a0 = f, e):
e := expand(e):
1000*(time[real]()-st);

print(e);

done
