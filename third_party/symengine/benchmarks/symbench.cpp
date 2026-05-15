#include <iostream>
#include <chrono>
#include <iomanip>

#include <symengine/basic.h>
#include <symengine/add.h>
#include <symengine/symbol.h>
#include <symengine/integer.h>
#include <symengine/mul.h>
#include <symengine/pow.h>
#include <symengine/constants.h>
#include <symengine/real_double.h>
#include <symengine/functions.h>

using SymEngine::add;
using SymEngine::Basic;
using SymEngine::cos;
using SymEngine::div;
using SymEngine::expand;
using SymEngine::I;
using SymEngine::iaddnum;
using SymEngine::integer;
using SymEngine::Integer;
using SymEngine::mul;
using SymEngine::Number;
using SymEngine::one;
using SymEngine::pi;
using SymEngine::pow;
using SymEngine::RCP;
using SymEngine::rcp_static_cast;
using SymEngine::RCPBasicKeyLess;
using SymEngine::real_double;
using SymEngine::sin;
using SymEngine::sub;
using SymEngine::symbol;
using SymEngine::Symbol;
using SymEngine::vec_basic;
using SymEngine::zero;

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
    SymEngine::print_stack_on_segfault();
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

RCP<const Basic> f(RCP<const Basic> z)
{
    return add(mul(sqrt(div(one, integer(3))), pow(z, integer(2))),
               div(I, integer(3)));
}

double R1()
{
    RCP<const Basic> g;
    RCP<const Basic> h = div(I, integer(2));
    auto t1 = std::chrono::high_resolution_clock::now();
    RCP<const Basic> real, imag;
    as_real_imag(f(f(f(f(f(f(f(f(f(f(h)))))))))), outArg(real), outArg(imag));
    auto t2 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(t2 - t1).count();
}

RCP<const Basic> hermite(RCP<const Integer> n, RCP<const Basic> y)
{
    if (eq(*n, *one))
        return mul(y, integer(2));
    if (eq(*n, *zero))
        return one;
    return expand(
        sub(mul(mul(integer(2), y), hermite(n->subint(*one), y)),
            mul(integer(2),
                mul(n->subint(*one), hermite(n->subint(*integer(2)), y)))));
}

double R2()
{
    RCP<const Basic> g;
    RCP<const Integer> n = integer(15);
    RCP<const Basic> y = symbol("y");
    auto t1 = std::chrono::high_resolution_clock::now();
    g = hermite(n, y);
    auto t2 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(t2 - t1).count();
}

double R3()
{
    RCP<const Basic> x = symbol("x");
    RCP<const Basic> y = symbol("y");
    RCP<const Basic> z = symbol("z");
    RCP<const Basic> f = add(x, add(y, z));
    std::vector<bool> vec;
    vec.reserve(10);
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10; i++) {
        vec.push_back(eq(*f, *f));
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(t2 - t1).count();
}

double R5()
{
    RCP<const Basic> x = symbol("x");
    RCP<const Basic> y = symbol("y");
    RCP<const Basic> z = symbol("z");
    RCP<const Basic> f = add(x, add(y, z));
    vec_basic v;

    v.push_back(x);
    v.push_back(y);
    v.push_back(z);
    for (int i = 0; i < 8; i++) {
        v.push_back(add(v[i], add(v[i + 1], v[i + 2])));
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    std::set<RCP<const Basic>, RCPBasicKeyLess> s(v.begin(), v.end());
    auto t2 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(t2 - t1).count();
}

double R7()
{
    RCP<const Basic> x = symbol("x");
    RCP<const Basic> f
        = add(pow(x, integer(24)),
              add(mul(integer(34), pow(x, integer(12))),
                  add(mul(integer(45), pow(x, integer(3))),
                      add(mul(integer(9), pow(x, integer(18))),
                          add(mul(integer(34), pow(x, integer(10))),
                              mul(integer(32), pow(x, integer(21))))))));
    vec_basic v;
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10000; ++i) {
        v.push_back(f->subs({{x, real_double(0.5)}}));
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(t2 - t1).count();
}

RCP<const Basic> right(const RCP<const Basic> &f, const RCP<const Number> &a,
                       const RCP<const Number> &b, const RCP<const Basic> &x,
                       int n)
{
    RCP<const Number> Deltax = b->sub(*a)->div(*integer(n));
    RCP<const Number> c = a;
    RCP<const Number> est = integer(0);
    for (int i = 0; i < n; i++) {
        iaddnum(outArg(c), Deltax);
        iaddnum(outArg(est), rcp_static_cast<const Number>(f->subs({{x, c}})));
    }
    return mulnum(est, Deltax);
}

double R8()
{
    RCP<const Basic> x = symbol("x");
    auto t1 = std::chrono::high_resolution_clock::now();
    x = right(pow(x, integer(2)), integer(0), integer(5), x, 10000);
    auto t2 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(t2 - t1).count();
}

double S1()
{
    RCP<const Basic> x = symbol("x");
    RCP<const Basic> y = symbol("y");
    RCP<const Basic> z = symbol("z");
    RCP<const Basic> e;
    RCP<const Basic> f;

    e = pow(add(x, add(y, add(z, one))), integer(7));
    f = mul(e, add(e, one));

    auto t1 = std::chrono::high_resolution_clock::now();
    f = expand(f);
    auto t2 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(t2 - t1).count();
}

double S2()
{
    RCP<const Basic> x = symbol("x");
    RCP<const Basic> y = symbol("y");
    RCP<const Basic> z = symbol("z");
    RCP<const Basic> e;
    RCP<const Basic> f;

    e = pow(add(pow(x, sin(x)), add(pow(y, cos(y)), pow(z, add(x, y)))),
            integer(100));

    auto t1 = std::chrono::high_resolution_clock::now();
    f = expand(e);
    auto t2 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(t2 - t1).count();
}

double S3()
{
    RCP<const Basic> x = symbol("x");
    RCP<const Basic> y = symbol("y");
    RCP<const Basic> z = symbol("z");
    RCP<const Basic> e;
    RCP<const Basic> f;

    e = pow(add(pow(x, y), add(pow(y, z), pow(z, x))), integer(50));
    e = expand(e);

    auto t1 = std::chrono::high_resolution_clock::now();
    f = e->diff(rcp_static_cast<const Symbol>(x));
    auto t2 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(t2 - t1).count();
}

double S3a()
{
    RCP<const Basic> x = symbol("x");
    RCP<const Basic> y = symbol("y");
    RCP<const Basic> z = symbol("z");
    RCP<const Basic> e;
    RCP<const Basic> f;

    e = pow(add(pow(x, y), add(pow(y, z), pow(z, x))), integer(500));
    e = expand(e);

    auto t1 = std::chrono::high_resolution_clock::now();
    f = e->diff(rcp_static_cast<const Symbol>(x));
    auto t2 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(t2 - t1).count();
}
