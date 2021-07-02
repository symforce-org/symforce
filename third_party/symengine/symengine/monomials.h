/**
 *  \file monomials.h
 *  Monomial Multiplication
 *
 **/

#ifndef SYMENGINE_MONOMIALS_H
#define SYMENGINE_MONOMIALS_H

#include <symengine/basic.h>

namespace SymEngine
{
//! Monomial multiplication
void monomial_mul(const vec_int &A, const vec_int &B, vec_int &C);

} // SymEngine

#endif
