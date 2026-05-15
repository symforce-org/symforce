/**
 *  \file rings.h
 *  Polynomial Manipulation
 *
 **/
#ifndef SYMENGINE_RINGS_H
#define SYMENGINE_RINGS_H

#include <symengine/basic.h>

namespace SymEngine
{

//! Converts expression `p` into a polynomial `P`, with symbols `sym`
void expr2poly(const RCP<const Basic> &p, umap_basic_num &syms,
               umap_vec_mpz &P);

//! Multiply two polynomials: `C = A*B`
void poly_mul(const umap_vec_mpz &A, const umap_vec_mpz &B, umap_vec_mpz &C);

} // namespace SymEngine

#endif
