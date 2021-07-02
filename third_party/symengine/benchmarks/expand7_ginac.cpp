// To compile on a debian system you need to install libginac-dev first
// $ sudo apt-get install libginac-dev
// Then compile with the following command,
// $ g++ -std=c++0x -o expand6_ginac -Wl,--no-as-needed `pkg-config --cflags
// --libs ginac` expand6_ginac.cpp
// See this SO answer: http://stackoverflow.com/a/18696743/1895353

#include <iostream>
#include <chrono>

#include <ginac/ginac.h>
using GiNaC::ex;
using GiNaC::pow;
using GiNaC::add;
using GiNaC::expand;
using GiNaC::exmap;
using GiNaC::symbol;
using GiNaC::sqrt;
using GiNaC::numeric;

int main(int argc, char *argv[])
{
    int N;
    if (argc == 2) {
        N = std::atoi(argv[1]);
    } else {
        N = 20;
    }

    ex e, f, x, y;
    x = symbol("x");
    y = symbol("y");
    e = pow((1 + sqrt(ex(3)) * x + sqrt(ex(5)) * y), N);
    f = e * (e + sqrt(ex(7)));

    auto t1 = std::chrono::high_resolution_clock::now();
    f = expand(f);
    auto t2 = std::chrono::high_resolution_clock::now();

    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1)
                     .count()
              << "ms" << std::endl;
    // std::cout << f << std::endl;

    return 0;
}
