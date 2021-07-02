#!/usr/local/bin/WolframScript -script

If[Length[$ScriptCommandLine] == 1, n = 15, n = ToExpression[$ScriptCommandLine[[2]]]];

e = Sin[Cos[x+1]];
Print[AbsoluteTiming[Series[e, {x, 0, n}];][[1]] * 1000];

