// To compile on a debian system you need to install libginac-dev first
// $ sudo apt-get install libginac-dev
// Then compile with the following command,
// $ g++ -o matrix_add2_ginac -Wl,--no-as-needed `pkg-config --cflags --libs
// ginac` matrix_add2_ginac.cpp
// See this SO answer: http://stackoverflow.com/a/18696743/1895353

#include <iostream>
#include <chrono>

#include <ginac/ginac.h>

using namespace GiNaC;

int main()
{
    matrix A(3, 3), B(3, 3), C(3, 3);

    A = symbol("a"), symbol("b"), symbol("c"), symbol("d"), symbol("e"),
    symbol("f"), symbol("g"), symbol("h"), symbol("i");

    B = symbol("x"), symbol("y"), symbol("z"), symbol("p"), symbol("q"),
    symbol("r"), symbol("u"), symbol("v"), symbol("w");

    unsigned N = 10000;

    auto t1 = std::chrono::high_resolution_clock::now();
    for (unsigned i = 0; i < N; i++)
        C = A.add(B);
    ;
    auto t2 = std::chrono::high_resolution_clock::now();

    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1)
                         .count()
                     / N
              << " microseconds" << std::endl;
}
