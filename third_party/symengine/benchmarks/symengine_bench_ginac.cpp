#include <iostream>
#include <chrono>
#include <ginac/ginac.h>
using namespace GiNaC;

int main(int argc, char *argv[])
{
    int N;
    if (argc == 2) {
        N = std::atoi(argv[1]);
    } else {
        N = 15;
    }

    symbol x("x");
    ex e = sin(cos(x + 1));

    auto t1 = std::chrono::high_resolution_clock::now();
    ex g = e.series(x == 0, N);
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1)
                     .count()
              << "ms" << std::endl;

    return 0;
}
