#include <iostream>
#include <chrono>

#include <symengine/basic.h>
#include <symengine/add.h>
#include <symengine/symbol.h>
#include <symengine/dict.h>
#include <symengine/integer.h>
#include <symengine/mul.h>
#include <symengine/pow.h>
#include <symengine/rings.h>
#include <symengine/monomials.h>

using SymEngine::Basic;
using SymEngine::Symbol;
using SymEngine::symbol;
using SymEngine::umap_basic_num;
using SymEngine::Integer;
using SymEngine::integer;
using SymEngine::expr2poly;
using SymEngine::poly_mul;
using SymEngine::umap_vec_mpz;
using SymEngine::RCP;
using SymEngine::print_stack_on_segfault;

int main(int argc, char *argv[])
{
    print_stack_on_segfault();
    RCP<const Basic> x = symbol("x");
    RCP<const Basic> y = symbol("y");
    RCP<const Basic> z = symbol("z");
    RCP<const Basic> w = symbol("w");
    RCP<const Basic> i15 = integer(15);

    RCP<const Basic> e, f1, f2, r;

    e = pow(add(add(add(x, y), z), w), i15);
    f1 = expand(e);
    f2 = expand(add(e, w));

    umap_basic_num syms;
    insert(syms, x, integer(0));
    insert(syms, y, integer(1));
    insert(syms, z, integer(2));
    insert(syms, w, integer(3));

    umap_vec_mpz P1, P2, C;

    expr2poly(f1, syms, P1);
    expr2poly(f2, syms, P2);
    std::cout << "poly_mul start" << std::endl;
    auto t1 = std::chrono::high_resolution_clock::now();
    poly_mul(P1, P2, C);
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "poly_mul stop" << std::endl;

    /*
    std::cout << *e << std::endl;
    std::cout << *f1 << std::endl;
    std::cout << P1 << std::endl;
    std::cout << *f2 << std::endl;
    std::cout << P2 << std::endl;
    std::cout << "RESULT:" << std::endl;
    std::cout << C << std::endl;
    */
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1)
                     .count()
              << "ms" << std::endl;
    std::cout << "number of terms: " << C.size() << std::endl;

    return 0;
}
