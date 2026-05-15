#include "catch.hpp"
#include <chrono>

#include <symengine/add.h>
#include <symengine/pow.h>
#include <symengine/rings.h>
#include <symengine/monomials.h>
#include <symengine/symengine_exception.h>

using SymEngine::Add;
using SymEngine::Basic;
using SymEngine::expr2poly;
using SymEngine::Integer;
using SymEngine::integer;
using SymEngine::map_vec_mpz;
using SymEngine::monomial_mul;
using SymEngine::Mul;
using SymEngine::poly_mul;
using SymEngine::Pow;
using SymEngine::print_stack_on_segfault;
using SymEngine::RCP;
using SymEngine::rcp_dynamic_cast;
using SymEngine::Symbol;
using SymEngine::symbol;
using SymEngine::SymEngineException;
using SymEngine::umap_basic_num;
using SymEngine::umap_vec_mpz;
using SymEngine::vec_int;

TEST_CASE("monomial_mul: poly", "[poly]")
{
    vec_int a, b, c, d;
    a = {1, 2, 3, 4};
    b = {2, 3, 2, 5};
    c = {0, 0, 0, 0};

    monomial_mul(a, b, c);

    d = {3, 5, 5, 9};
    REQUIRE(c == d);
    d = {5, 6, 5, 5};
    REQUIRE(c != d);

    umap_vec_mpz m;
    m[a] = 4;
}

TEST_CASE("expand: poly", "[poly]")
{
    RCP<const Basic> x = symbol("x");
    RCP<const Basic> y = symbol("y");
    RCP<const Basic> z = symbol("z");
    RCP<const Basic> w = symbol("w");
    RCP<const Basic> i4 = integer(2);

    RCP<const Basic> e, f1, f2, r;

    e = pow(add(add(add(x, y), z), w), i4);
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
}
