/**
 *  \file eval_mpc.h
 *  Evaluation of numeric expressions using MPC
 *
 **/

#ifndef SYMENGINE_EVAL_MPC_H
#define SYMENGINE_EVAL_MPC_H

#include <symengine/symengine_config.h>

#ifdef HAVE_SYMENGINE_MPC
#include <symengine/basic.h>
#include <mpc.h>

namespace SymEngine
{

//! Evaluate expression `b` and store it in `result` with rounding mode rnd
// Different precisions for real and imaginary parts of `result` is not
// supported
// Use `mpc_init2` to initialize `result`
void eval_mpc(mpc_ptr result, const Basic &b, mpfr_rnd_t rnd);

} // SymEngine

#endif // HAVE_SYMENGINE_MPC

#endif
