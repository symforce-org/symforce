// To compile on a debian system you need to install libginac-dev first
// $ sudo apt-get install libginac-dev
// Then compile with the following command,
// $ g++ -std=c++0x -o symbench_ginac -Wl,--no-as-needed `pkg-config --cflags
// --libs ginac` symbench_ginac.cpp
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

double R1();
double R2();
double R3();
double R5();
double R7();
double R8();
double S1();
double S2();
double S3();
double S3a();

int main(int argc, char *argv[])
{
    std::cout << "Time for R1 : \t " << std::setw(15) << std::setprecision(9)
              << std::fixed << R1() << std::endl;
    std::cout << "Time for R2 : \t " << std::setw(15) << std::setprecision(9)
              << std::fixed << R2() << std::endl;
    std::cout << "Time for R3 : \t " << std::setw(15) << std::setprecision(9)
              << std::fixed << R3() << std::endl;
    std::cout << "Time for R5 : \t " << std::setw(15) << std::setprecision(9)
              << std::fixed << R5() << std::endl;
    std::cout << "Time for R7 : \t " << std::setw(15) << std::setprecision(9)
              << std::fixed << R7() << std::endl;
    std::cout << "Time for R8 : \t " << std::setw(15) << std::setprecision(9)
              << std::fixed << R8() << std::endl;
    std::cout << "Time for S1 : \t " << std::setw(15) << std::setprecision(9)
              << std::fixed << S1() << std::endl;
    std::cout << "Time for S2 : \t " << std::setw(15) << std::setprecision(9)
              << std::fixed << S2() << std::endl;
    std::cout << "Time for S3 : \t " << std::setw(15) << std::setprecision(9)
              << std::fixed << S3() << std::endl;
    std::cout << "Time for S3a : \t " << std::setw(15) << std::setprecision(9)
              << std::fixed << S3a() << std::endl;
    return 0;
}

ex f(ex z)
{
    return sqrt(ex(numeric(1) / 3)) * pow(z, 2) + I / 3;
}

double R1()
{
    ex g;
    ex h = I / 2;
    auto t1 = std::chrono::high_resolution_clock::now();
    g = real_part(f(f(f(f(f(f(f(f(f(f(h)))))))))));
    auto t2 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count()
           / 1000000000.0;
}

ex hermite(numeric n, ex y)
{
    if (n == 1)
        return 2 * y;
    if (n == 0)
        return 1;
    return expand(2 * y * hermite(n - 1, y) - 2 * (n - 1) * hermite(n - 2, y));
}

double R2()
{
    ex g;
    numeric n(15);
    ex y = symbol("y");
    auto t1 = std::chrono::high_resolution_clock::now();
    g = hermite(n, y);
    auto t2 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count()
           / 1000000000.0;
}

double R3()
{
    ex x = symbol("x");
    ex y = symbol("y");
    ex z = symbol("z");
    ex f = x + y + z;
    std::vector<bool> vec(10);
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10; i++) {
        vec.push_back(((bool)f.is_equal(f)));
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count()
           / 1000000000.0;
}

double R5()
{
    ex x = symbol("x");
    ex y = symbol("y");
    ex z = symbol("z");
    ex f = x + y + z;
    std::vector<ex> v;

    v.push_back(x);
    v.push_back(y);
    v.push_back(z);
    for (int i = 0; i < 8; i++) {
        v.push_back(v[i] + v[i + 1] + v[i + 2]);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    std::set<ex, ex_is_less> s(v.begin(), v.end());
    auto t2 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count()
           / 1000000000.0;
}

double R7()
{
    ex x = symbol("x");
    ex f = pow(x, 24) + 34 * pow(x, 12) + 45 * pow(x, 3) + 9 * pow(x, 18)
           + 34 * pow(x, 10) + 32 * pow(x, 21);
    std::vector<ex> v;
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10000; ++i) {
        v.push_back(f.subs(x == 0.5));
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count()
           / 1000000000.0;
}

ex right(ex f, numeric a, numeric b, ex x, int n)
{
    numeric Deltax = (b - a) / n;
    numeric c = a;
    numeric est = 0;
    for (int i = 0; i < n; i++) {
        c += Deltax;
        est += f.subs(x == c);
    }
    return est * Deltax;
}

double R8()
{
    ex x = symbol("x");
    auto t1 = std::chrono::high_resolution_clock::now();
    right(pow(x, 2), 0, 5, x, 10000);
    auto t2 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count()
           / 1000000000.0;
}

double S1()
{
    ex x = symbol("x");
    ex y = symbol("y");
    ex z = symbol("z");
    ex e;
    ex f;

    e = pow(x + y + z + 1, 7);
    f = e * (e + 1);

    auto t1 = std::chrono::high_resolution_clock::now();
    f = expand(f);
    auto t2 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count()
           / 1000000000.0;
}

double S2()
{
    ex x = symbol("x");
    ex y = symbol("y");
    ex z = symbol("z");
    ex e;
    ex f;

    e = pow(pow(x, sin(x)) + pow(y, cos(y)) + pow(z, x + y), 100);

    auto t1 = std::chrono::high_resolution_clock::now();
    f = expand(e);
    auto t2 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count()
           / 1000000000.0;
}

double S3()
{
    symbol x = symbol("x");
    ex y = symbol("y");
    ex z = symbol("z");
    ex e;
    ex f;

    e = pow(pow(x, y) + pow(y, z) + pow(z, x), 50);
    e = expand(e);

    auto t1 = std::chrono::high_resolution_clock::now();
    f = e.diff(x);
    auto t2 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count()
           / 1000000000.0;
}

double S3a()
{
    symbol x = symbol("x");
    ex y = symbol("y");
    ex z = symbol("z");
    ex e;
    ex f;

    e = pow(pow(x, y) + pow(y, z) + pow(z, x), 500);
    e = expand(e);

    auto t1 = std::chrono::high_resolution_clock::now();
    f = e.diff(x);
    auto t2 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count()
           / 1000000000.0;
}
