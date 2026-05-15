#include <chrono>
#include <iostream>

#include <symengine/series_generic.h>

using SymEngine::Basic;
using SymEngine::Expression;
using SymEngine::integer;
using SymEngine::integer_class;
using SymEngine::map_int_Expr;
using SymEngine::pow;
using SymEngine::RCP;
using SymEngine::rcp_dynamic_cast;
using SymEngine::Symbol;
using SymEngine::symbol;
using SymEngine::UExprDict;
using SymEngine::UExprPoly;
using SymEngine::UnivariateSeries;

int main(int argc, char *argv[])
{
    SymEngine::print_stack_on_segfault();

    RCP<const Symbol> x = symbol("x");
    std::vector<Expression> v;
    int N;

    N = 1000;
    for (int i = 0; i < N; ++i) {
        Expression coef(i);
        v.push_back(coef);
    }

    UExprDict c, p(UExprPoly::from_vec(x, v)->get_dict());
    auto t1 = std::chrono::high_resolution_clock::now();
    c = UnivariateSeries::mul(p, p, 1000);
    auto t2 = std::chrono::high_resolution_clock::now();
    // std::cout << *a << std::endl;
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1)
                     .count()
              << "ms" << std::endl;

    return 0;
}
