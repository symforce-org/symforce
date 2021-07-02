#!/usr/local/bin/WolframScript -script

If[Length[$ScriptCommandLine] == 1, n = 100, n = ToExpression[$ScriptCommandLine[[2]]]];

a0 = Symbol["a0"];
a1 = Symbol["a1"];

e = a0 + a1;
f = 0;

Do[f = f + Symbol["a" <> ToString[i]], {i, 2, n - 1}];
Do[e = e + Symbol["a" <> ToString[i]], {i, 2, n - 1}];

f = -f;

g[e, a0, f] := (e = Expand[e ^ 2]; e = e/.a0->f; e = Expand[e]);

Print[AbsoluteTiming[r = g[e, a0, f]][[1]] * 1000];

