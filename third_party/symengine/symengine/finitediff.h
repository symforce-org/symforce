/**
 *  \file finitediff.h
 *  Includes function to generate finitedifference weights
 *
 **/

#ifndef SYMENGINE_FINITEDIFF_H
#define SYMENGINE_FINITEDIFF_H

#include <symengine/basic.h>

namespace SymEngine
{

vec_basic generate_fdiff_weights_vector(const vec_basic &grid,
                                        const unsigned max_deriv,
                                        const RCP<const Basic> around);
}

#endif
