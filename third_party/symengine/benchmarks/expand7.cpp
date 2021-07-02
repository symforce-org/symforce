#include <iostream>
#include <chrono>

#include <symengine/basic.h>
#include <symengine/add.h>
#include <symengine/symbol.h>
#include <symengine/mul.h>
#include <symengine/pow.h>
#include <symengine/integer.h>
#include "symengine/constants.h"

using SymEngine::Basic;
using SymEngine::RCP;
using SymEngine::symbol;
using SymEngine::zero;
using SymEngine::one;
using SymEngine::map_basic_basic;
using SymEngine::sqrt;
using SymEngine::integer;

int main(int argc, char *argv[])
{
    SymEngine::print_stack_on_segfault();
    int N;
    if (argc == 2) {
        N = std::atoi(argv[1]);
    } else {
        N = 20;
    }

    RCP<const Basic> x = symbol("x"), y = symbol("y"), e, f;
    e = pow(add(one, add(mul(sqrt(integer(3)), x), mul(sqrt(integer(5)), y))),
            integer(N));
    f = mul(e, add(e, sqrt(integer(7))));
    auto t1 = std::chrono::high_resolution_clock::now();
    f = expand(f);
    auto t2 = std::chrono::high_resolution_clock::now();

    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1)
                     .count()
              << "ms" << std::endl;
    // std::cout << f->__str__() << std::endl;

    return 0;
}
