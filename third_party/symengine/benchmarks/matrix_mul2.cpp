#include <iostream>
#include <chrono>

#include <symengine/basic.h>
#include <symengine/matrix.h>
#include <symengine/symbol.h>

using SymEngine::Basic;
using SymEngine::RCP;
using SymEngine::DenseMatrix;
using SymEngine::symbol;

int main(int argc, char *argv[])
{
    SymEngine::print_stack_on_segfault();

    DenseMatrix A = DenseMatrix(3, 3, {symbol("a"), symbol("b"), symbol("c"),
                                       symbol("d"), symbol("e"), symbol("f"),
                                       symbol("g"), symbol("h"), symbol("i")});

    DenseMatrix B = DenseMatrix(3, 3, {symbol("x"), symbol("y"), symbol("z"),
                                       symbol("p"), symbol("q"), symbol("r"),
                                       symbol("u"), symbol("v"), symbol("w")});

    DenseMatrix C(3, 3);

    std::cout << "Multiplying Two Matrices; matrix dimensions: 3 x 3"
              << std::endl;

    unsigned N = 10000;
    auto t1 = std::chrono::high_resolution_clock::now();
    for (unsigned i = 0; i < N; i++)
        mul_dense_dense(A, B, C);
    auto t2 = std::chrono::high_resolution_clock::now();

    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1)
                         .count()
                     / N
              << " microseconds" << std::endl;

    return 0;
}
