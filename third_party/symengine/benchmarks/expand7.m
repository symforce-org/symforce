#!/usr/local/bin/WolframScript -script

If[Length[$ScriptCommandLine] == 1, n = 20, n = ToExpression[$ScriptCommandLine[[2]]]];

e = (1 + Sqrt[3] * x + Sqrt[5] * y) ^ n;
f = e * (e + Sqrt[7])

Print[AbsoluteTiming[r = Expand[f]][[1]] * 1000]

