#include <iostream>
#include <chrono>

#include <symengine/basic.h>
#include <symengine/integer.h>
#include <symengine/matrix.h>

using SymEngine::Basic;
using SymEngine::Integer;
using SymEngine::RCP;
using SymEngine::integer;
using SymEngine::DenseMatrix;

int main(int argc, char *argv[])
{
    SymEngine::print_stack_on_segfault();

    DenseMatrix A = DenseMatrix(
        4, 4, {integer(-23), integer(67), integer(3), integer(4), integer(54),
               integer(61), integer(7), integer(8), integer(32), integer(15),
               integer(12), integer(13), integer(100), integer(17), integer(15),
               integer(178)});

    DenseMatrix B
        = DenseMatrix(4, 4, {integer(12), integer(22), integer(30), integer(40),
                             integer(45), integer(6), integer(37), integer(80),
                             integer(91), integer(10), integer(16), integer(52),
                             integer(45), integer(14), integer(2), integer(6)});

    DenseMatrix C(4, 4);

    std::cout << "Multiplying Two Matrices; matrix dimensions: 4 x 4"
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
