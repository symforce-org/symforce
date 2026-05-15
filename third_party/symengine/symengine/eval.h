/**
 *  \file eval.h
 *
 **/
#ifndef SYMENGINE_EVAL_H
#define SYMENGINE_EVAL_H

#include <symengine/basic.h>
#include <symengine/dict.h>
#include <symengine/symengine_rcp.h>

#ifdef HAVE_SYMENGINE_MPFR
#include <mpfr.h>
#endif // HAVE_SYMENGINE_MPFR

#ifdef SYMENGINE_HAVE_MPC
#include <mpc.h>
#endif // HAVE_SYMENGINE_MPC

namespace SymEngine
{

/*
 * Evaluates basic b, according to the number of significant bits
 * in the given domain
 */

enum class EvalfDomain {
    Complex = 0,
    Real = 1,
    Symbolic = 2,
};

RCP<const Basic> evalf(const Basic &b, unsigned long bits,
                       EvalfDomain domain = EvalfDomain::Symbolic);

} // namespace SymEngine

#endif // SYMENGINE_EVAL_H
