// To compile on a debian system you need to install libginac-dev first
// $ sudo apt-get install libginac-dev
// Then compile with the following command,
// $ g++ -std=c++0x -o lwbench_ginac -Wl,--no-as-needed `pkg-config --cflags
// --libs ginac` lwbench_ginac.cpp
// See this SO answer: http://stackoverflow.com/a/18696743/1895353

#include <iostream>
#include <chrono>

#include <ginac/ginac.h>
using GiNaC::basic;
using GiNaC::ex;
using GiNaC::ex_is_less;
using GiNaC::sqrt;
using GiNaC::numeric;
using GiNaC::pow;
using GiNaC::I;
using GiNaC::expand;
using GiNaC::real_part;
using GiNaC::symbol;
using GiNaC::factorial;

double A()
{
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i = 1; i <= 100; i++) {
        factorial(1000 + i) / factorial(900 + i);
    }
    auto t2 = std::chrono::high_resolution_clock::now();

    return std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count()
           / 1000000000.0;
}

double B()
{
    numeric s = 0;

    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i = 1; i <= 1000; i++) {
        s = s + numeric(1) / i;
    }
    auto t2 = std::chrono::high_resolution_clock::now();

    return std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count()
           / 1000000000.0;
}

double C()
{
    numeric x = numeric(13 * 17 * 31);
    numeric y = numeric(13 * 19 * 29);

    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i = 1; i <= 200; i++) {
        gcd(pow(x, numeric(300 + i % 181)), pow(y, numeric(200 + i % 183)));
    }
    auto t2 = std::chrono::high_resolution_clock::now();

    return std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count()
           / 1000000000.0;
}

double D()
{
    ex s = numeric(0);
    ex y = symbol("y");
    ex t = symbol("t");

    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i = 1; i <= 10; i++) {
        s = s
            + numeric(i) * y * pow(t, numeric(i))
                  / (pow(y + numeric(i) * t, numeric(i)));
    }
    auto t2 = std::chrono::high_resolution_clock::now();

    return std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count()
           / 1000000000.0;
}

double E()
{
    ex s = numeric(0);
    ex y = symbol("y");
    ex t = symbol("t");

    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i = 1; i <= 10; i++) {
        s = s
            + numeric(i) * y * pow(t, numeric(i))
                  / (pow(y + abs(numeric(5 - i)) * t, numeric(i)));
    }
    auto t2 = std::chrono::high_resolution_clock::now();

    return std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count()
           / 1000000000.0;
}

int main(int argc, char *argv[])
{
    std::cout << "Time for A : \t " << std::setw(15) << std::setprecision(9)
              << std::fixed << A() << std::endl;
    std::cout << "Time for B : \t " << std::setw(15) << std::setprecision(9)
              << std::fixed << B() << std::endl;
    std::cout << "Time for C : \t " << std::setw(15) << std::setprecision(9)
              << std::fixed << C() << std::endl;
    std::cout << "Time for D : \t " << std::setw(15) << std::setprecision(9)
              << std::fixed << D() << std::endl;
    std::cout << "Time for E : \t " << std::setw(15) << std::setprecision(9)
              << std::fixed << E() << std::endl;
    return 0;
}
