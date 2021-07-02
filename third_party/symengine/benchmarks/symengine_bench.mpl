#!/bin/bash maple
# Use `maple -q symengine_bench.mpl -D n=15` to run

e := sin(cos(x+1)):
st := time[real]():
f := series(e, x=0,n):
1000*(time[real]()-st);

done
