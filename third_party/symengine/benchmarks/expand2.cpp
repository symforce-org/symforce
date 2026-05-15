#include <iostream>
#include <chrono>

#include <symengine/basic.h>
#include <symengine/add.h>
#include <symengine/symbol.h>
#include <symengine/dict.h>
#include <symengine/integer.h>
#include <symengine/mul.h>
#include <symengine/pow.h>

using SymEngine::Add;
using SymEngine::Basic;
using SymEngine::Integer;
using SymEngine::integer;
using SymEngine::Mul;
using SymEngine::multinomial_coefficients;
using SymEngine::Pow;
using SymEngine::RCP;
using SymEngine::rcp_dynamic_cast;
using SymEngine::Symbol;
using SymEngine::symbol;
using SymEngine::umap_basic_num;

int main(int argc, char *argv[])
{
    int N;
    if (argc == 2) {
        N = std::atoi(argv[1]);
    } else {
        N = 15;
    }
    SymEngine::print_stack_on_segfault();

    RCP<const Basic> x = symbol("x");
    RCP<const Basic> y = symbol("y");
    RCP<const Basic> z = symbol("z");
    RCP<const Basic> w = symbol("w");
    RCP<const Basic> i = integer(N);

    RCP<const Basic> e, f, r;

    e = pow(add(add(add(x, y), z), w), i);
    f = mul(e, add(e, w));

    // std::cout << "Expanding: " << *f << std::endl;

    auto t1 = std::chrono::high_resolution_clock::now();
    r = expand(f);
    auto t2 = std::chrono::high_resolution_clock::now();
    // std::cout << *r << std::endl;
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1)
                     .count()
              << "ms" << std::endl;
    std::cout << "number of terms: "
              << rcp_dynamic_cast<const Add>(r)->get_dict().size() << std::endl;

    return 0;
}
