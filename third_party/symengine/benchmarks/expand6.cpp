#include <iostream>
#include <chrono>

#include <symengine/basic.h>
#include <symengine/add.h>
#include <symengine/symbol.h>
#include <symengine/mul.h>
#include <symengine/pow.h>
#include <symengine/integer.h>
#include <symengine/constants.h>

using SymEngine::Basic;
using SymEngine::RCP;
using SymEngine::symbol;
using SymEngine::zero;
using SymEngine::map_basic_basic;
using SymEngine::sqrt;
using SymEngine::integer;
using SymEngine::expand;

int main(int argc, char *argv[])
{
    SymEngine::print_stack_on_segfault();
    int N;
    if (argc == 2) {
        N = std::atoi(argv[1]);
    } else {
        N = 100;
    }

    RCP<const Basic> e, f, s, a0, a1;
    a0 = symbol("a0");
    a1 = symbol("a1");
    e = add(a0, a1);
    f = zero;
    for (long long i = 2; i < N; i++) {
        std::ostringstream o;
        o << "a" << i;
        s = symbol(o.str());
        e = add(e, s);
        f = add(f, s);
    }
    f = neg(f);
    auto t1 = std::chrono::high_resolution_clock::now();
    e = expand(pow(e, integer(2)));
    e = e->subs({{a0, f}});
    e = expand(e);
    auto t2 = std::chrono::high_resolution_clock::now();

    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1)
                     .count()
              << "ms" << std::endl;
    std::cout << e->__str__() << std::endl;

    return 0;
}
