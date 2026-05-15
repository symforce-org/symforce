#include <iostream>
#include <chrono>
#include <iomanip>

#include <symengine/basic.h>
#include <symengine/add.h>
#include <symengine/symbol.h>
#include <symengine/integer.h>
#include <symengine/mul.h>
#include <symengine/pow.h>
#include <symengine/functions.h>

using SymEngine::add;
using SymEngine::Basic;
using SymEngine::integer;
using SymEngine::log;
using SymEngine::pow;
using SymEngine::RCP;
using SymEngine::rcp_static_cast;
using SymEngine::sin;
using SymEngine::sqrt;
using SymEngine::symbol;
using SymEngine::Symbol;

double CommonSubexprDiff(bool cache)
{
    RCP<const Basic> e;
    RCP<const Basic> f;

    std::string tmp_str = "a";
    std::vector<RCP<const Basic>> v;
    for (int i = 0; i < 10; ++i) {
        v.push_back(symbol(tmp_str));
        tmp_str += "a";
    }

    e = integer(23);
    for (unsigned int i = 0; i < v.size(); ++i) {
        RCP<const Basic> z = symbol(tmp_str);
        e = pow(e, add(cos(sqrt(log(sin(pow(v[v.size() - i - 1], v[i]))))), e));
    }
    e = expand(e);

    auto t1 = std::chrono::high_resolution_clock::now();
    f = e->diff(rcp_static_cast<const Symbol>(v[0]), cache);
    auto t2 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(t2 - t1).count();
}

double NoCommonSubexprDiff(bool cache)
{
    RCP<const Basic> e;
    RCP<const Basic> f;

    std::string tmp_str = "a";
    std::vector<RCP<const Basic>> v;
    for (int i = 0; i < 10; ++i) {
        v.push_back(symbol(tmp_str));
        tmp_str += "a";
    }

    e = integer(23);
    for (unsigned int i = 0; i < v.size(); ++i) {
        RCP<const Basic> z = symbol(tmp_str);
        e = pow(e, cos(sqrt(log(sin(pow(v[v.size() - i - 1], v[i]))))));
    }
    e = expand(e);

    auto t1 = std::chrono::high_resolution_clock::now();
    f = e->diff(rcp_static_cast<const Symbol>(v[0]), cache);
    auto t2 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(t2 - t1).count();
}

int main(int argc, char *argv[])
{
    SymEngine::print_stack_on_segfault();
    std::cout << "Time for expr with common subexpressions (with cache) : \t "
              << std::setw(15) << std::setprecision(9) << std::fixed
              << CommonSubexprDiff(true) << std::endl;
    std::cout
        << "Time for expr with common subexpressions (without cache) : \t "
        << std::setw(15) << std::setprecision(9) << std::fixed
        << CommonSubexprDiff(false) << std::endl;
    std::cout
        << "Time for expr without common subexpressions (with cache) : \t "
        << std::setw(15) << std::setprecision(9) << std::fixed
        << NoCommonSubexprDiff(true) << std::endl;
    std::cout
        << "Time for expr without common subexpressions (without cache) : \t "
        << std::setw(15) << std::setprecision(9) << std::fixed
        << NoCommonSubexprDiff(false) << std::endl;

    return 0;
}
