#include <iostream>
#include <chrono>

#include <symengine/basic.h>
#include <symengine/add.h>
#include <symengine/symbol.h>
#include <symengine/dict.h>
#include <symengine/integer.h>
#include <symengine/mul.h>
#include <symengine/pow.h>
#include <symengine/functions.h>
#include <symengine/eval_double.h>

using SymEngine::Add;
using SymEngine::Basic;
using SymEngine::eval_double;
using SymEngine::Integer;
using SymEngine::integer;
using SymEngine::Mul;
using SymEngine::multinomial_coefficients;
using SymEngine::Pow;
using SymEngine::RCP;
using SymEngine::rcp_dynamic_cast;
using SymEngine::sin;
using SymEngine::Symbol;
using SymEngine::umap_basic_num;

int main(int argc, char *argv[])
{
    SymEngine::print_stack_on_segfault();

    RCP<const Basic> e = sin(integer(1));
    double r, r_exact;

    for (int i = 0; i < 10000; i++)
        e = pow(add(mul(add(e, pow(integer(2), integer(-3))), integer(3)),
                    integer(1)),
                div(integer(2), integer(3)));

    //  Too long:
    //  std::cout << "Evaluating: " << *e << std::endl;

    auto t1 = std::chrono::high_resolution_clock::now();
    int num = 500;
    for (int i = 0; i < num; i++)
        r = eval_double(*e);
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << std::chrono::duration<double>(t2 - t1).count() * 1000 / num
              << "ms" << std::endl;
    /*
    In SymPy for few iterations:
    In [7]: sympify("(1 + 3*(1/8 + (1 + 3*(1/8 + (1 + 3*(1/8 + (1 + 3*(1/8 + (1
    + 3*(1/8 + sin(1)))^(2/3)))^(2/3)))^(2/3)))^(2/3)))^(2/3)").n(20)
    Out[7]: 8.0152751504518535013

    //    r_exact = 8.0152751504518535013;

    Here is code to use SymPy for more iterations:

    In [5]: e = sin(1)

    In [6]: for i in range(10):
       ...:     e = ((e+2**(-S(3)))*3 + 1)**(S(2)/3)
       ...:

    In [7]: e.n(20)
    Out[7]: 9.6473976427977306146

    But unfortunately SymPy can't do more than perhaps 10 or 20 iterations,
    while
    we need to test ~10000. However, the numbers seem to converge to 9.85647...

    */
    r_exact = 9.8564741713701043569;
    std::cout << "r (double) = " << r << std::endl;
    std::cout << "r (exact)  = " << r_exact << std::endl;
    std::cout << "error      = " << std::abs(r - r_exact) << std::endl;

    return 0;
}
