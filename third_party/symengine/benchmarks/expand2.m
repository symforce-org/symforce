#!/usr/local/bin/WolframScript -script

If[Length[$ScriptCommandLine] == 1, n = 15, n = ToExpression[$ScriptCommandLine[[2]]]];

e = (x + y + z + w) ^ n
f = e * (e + w)

Print[AbsoluteTiming[r = Expand[f]][[1]]*1000]
