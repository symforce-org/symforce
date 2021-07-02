// ============================================================================
//
//         Author:  Dale Lukas Peterson (dlp), hazelnusse@gmail.com
//
//    Description:  Quick expansion test
//
// ============================================================================
#include <iostream>
#include <chrono>
#include <ginac/ginac.h>
using namespace GiNaC;

// In [1]: from sympy_pyx import Symbol
// In [2]: x = Symbol("x")
// In [3]: y = Symbol("y")
// In [4]: z = Symbol("z")
// In [5]: w = Symbol("w")
// In [6]: e = (x+y+z+w)**15
// In [7]: f = e*(e+w)
// In [8]: f
// Out[8]: (y + x + z + w)^15 * ((y + x + z + w)^15 + w)
//
// In [9]: %time g = f.expand()
// CPU times: user 0.22 s, sys: 0.01 s, total: 0.22 s
// Wall time: 0.22 s
int main(int argc, char *argv[])
{
    int N;
    if (argc == 2) {
        N = std::atoi(argv[1]);
    } else {
        N = 15;
    }

    symbol x("x"), y("y"), z("z"), w("w");
    ex e = pow(x + y + z + w, N);
    ex f = e * (e + w);
    // std::cout << e << std::endl;
    // std::cout << f << std::endl;

    auto t1 = std::chrono::high_resolution_clock::now();
    ex g = f.expand();
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1)
                     .count()
              << "ms" << std::endl;

    return 0;
}
