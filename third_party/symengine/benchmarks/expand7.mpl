#!/bin/bash maple
# Use `maple -q expand7.mpl -D n=20` to run

e := (1 + sqrt(3) * x + sqrt(5) * y) ^ n:
f := e * (e + sqrt(7)):

st := time[real]():
f := expand(f):
1000*(time[real]() - st);

done
