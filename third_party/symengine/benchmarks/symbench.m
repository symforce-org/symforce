R1:=(
    Clear[f];
    Module[{f},
        f[z_]:=Sqrt[1/3]*z**2+I/3;     
        AbsoluteTiming[f[f[f[f[f[f[f[f[f[f[I/2]]]]]]]]]]][[1]]
    ]
)


R2:=(
    Clear[y];
    Module[{hermite},
        hermite[n_,y_]:=If[ n==1, 2*y,If[n==0,1,Expand[2*y*hermite[n-1,y]-2*(n-1)*hermite[n-2,y]]]];
        AbsoluteTiming[hermite[15,y]][[1]]
    ]
)


R3:=(
      Module[
          {x,y,z,f},
          f=x+y+z;
          AbsoluteTiming[Table[f==f,{i,10}]][[1]]
      ]
)


R5:=(
    Clear[x,y,z,L];
    Module[{L,x,y,z},
        L={x,y,z};
        Do[AppendTo[L, (L[[i]]+L[[i+1]])*L[[i+2]]], {i, 8}];
        AbsoluteTiming[Union[L]][[1]]
    ]
)

R7:=(
    Clear[x,f];
    Module[{f,x},
        f=x^24+34*x^12+45*x^3+9*x^18+34*x^10+32*x^21;
        AbsoluteTiming[Table[f/.x->0.5,{i,10^4}]][[1]]
    ]
)

R8:=(
    Clear[x];
    Module[{right, c, est, x, Deltax},
        right[f_,a_,b_,n_]:=(Deltax=(b-a)/n;c = a;
        est = 0;
        Do[c+= Deltax; est += f/.x->c, {i,n}]est*Deltax);
        AbsoluteTiming[right[x^2,0,5,10^4]][[1]]
    ]
)

S1:=(
    Clear[x,y,z,f];
    Module[{e,x,y,z},
        e=(x+y+z+1)^7;
        f=e*(e+1);
        AbsoluteTiming[Expand[f]][[1]]
    ]
)

S2:=(
    Clear[x,y,z,e];
    Module[{x,y,z,e},
        e=(x^Sin[x]+y^Cos[y]+z^(x+y))^100;
        AbsoluteTiming[Expand[e]][[1]]
    ]
)

S3:=(
    Clear[x,y,z,e];
    Module[{x,y,z,e},
        e=(x^y+y^z+z^x)^50;
        e = Expand[e];
        AbsoluteTiming[D[e,x]][[1]]
    ]
)

S3a:=(
    Clear[x,y,z,e];
    Module[{x,y,z,e},
        e=(x^y+y^z+z^x)^500;
        e = Expand[e];
        AbsoluteTiming[D[e,x]][[1]]
    ]
)

Print["Time for R1 : \t\t", R1]
Print["Time for R2 : \t\t", R2]
Print["Time for R3 : \t\t", R3]
Print["Time for R5 : \t\t", R5]
Print["Time for R7 : \t\t", R7]
Print["Time for R8 : \t\t", R8]
Print["Time for S1 : \t\t", S1]
Print["Time for S2 : \t\t", S2]
Print["Time for S3 : \t\t", S3]
Print["Time for S3a : \t\t", S3a]