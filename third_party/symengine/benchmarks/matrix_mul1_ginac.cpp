// To compile on a debian system you need to install libginac-dev first
// $ sudo apt-get install libginac-dev
// Then compile with the following command,
// $ g++ -o matrix_mul1_ginac -Wl,--no-as-needed `pkg-config --cflags --libs
// ginac` matrix_mul1_ginac.cpp
// See this SO answer: http://stackoverflow.com/a/18696743/1895353

#include <iostream>
#include <chrono>

#include <ginac/ginac.h>

using namespace GiNaC;

int main()
{
    matrix A(4, 4), B(4, 4), C(4, 4);

    A = -23, 67, 3, 4, 54, 61, 7, 8, 32, 15, 12, 13, 100, 17, 15, 178;

    B = 12, 22, 30, 40, 45, 6, 37, 80, 91, 10, 16, 52, 45, 14, 2, 6;

    unsigned N = 10000;

    auto t1 = std::chrono::high_resolution_clock::now();
    for (unsigned i = 0; i < N; i++)
        C = A.mul(B);
    ;
    auto t2 = std::chrono::high_resolution_clock::now();

    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1)
                         .count()
                     / N
              << " microseconds" << std::endl;
}
