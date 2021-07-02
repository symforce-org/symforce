/**
 *  \file eval_mpfr.h
 *  Evaluation of numeric expressions using MPFR
 *
 **/

#ifndef SYMENGINE_EVAL_MPFR_H
#define SYMENGINE_EVAL_MPFR_H

#include <symengine/symengine_config.h>

#ifdef HAVE_SYMENGINE_MPFR
#include <symengine/basic.h>
#include <mpfr.h>

namespace SymEngine
{

void eval_mpfr(mpfr_ptr result, const Basic &b, mpfr_rnd_t rnd);

} // SymEngine

#endif // HAVE_SYMENGINE_MPFR

#endif
