/**
 * \file diophantine.h
 * Algorithms for Diophantine equations
 *
 **/

#ifndef SYMENGINE_DIOPHANTINE_H
#define SYMENGINE_DIOPHANTINE_H

#include <symengine/matrix.h>

namespace SymEngine
{

// Solve the diophantine system Ax = 0 and return a basis set for solutions
void homogeneous_lde(std::vector<DenseMatrix> &basis, const DenseMatrix &A);
} // namespace SymEngine

#endif
