#!/bin/bash maple
# Use `maple -q expand2.mpl -D n=15` to run

e := (x + y + z + w) ^ n:
f := e * (e + w):

st := time[real]():
f := expand(f):
1000*(time[real]() - st);

done
