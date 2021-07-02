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
        4, 4,
        {integer(1), integer(2), integer(3), integer(4), integer(5), integer(6),
         integer(7), integer(8), integer(9), integer(10), integer(11),
         integer(12), integer(13), integer(14), integer(15), integer(16)});

    DenseMatrix B = DenseMatrix(
        4, 4,
        {integer(1), integer(2), integer(3), integer(4), integer(5), integer(6),
         integer(7), integer(8), integer(9), integer(10), integer(11),
         integer(12), integer(13), integer(14), integer(15), integer(16)});

    DenseMatrix C(4, 4);

    std::cout << "Adding Two Matrices; matrix dimensions: 4 x 4" << std::endl;

    // We are taking an average time since time for a single addition varied in
    // a range of 40-50 microseconds
    unsigned N = 10000;
    auto t1 = std::chrono::high_resolution_clock::now();
    for (unsigned i = 0; i < N; i++)
        add_dense_dense(A, B, C);
    auto t2 = std::chrono::high_resolution_clock::now();

    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1)
                         .count()
                     / N
              << " microseconds" << std::endl;

    return 0;
}
