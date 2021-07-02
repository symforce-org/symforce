// @HEADER
// ***********************************************************************
//
//                    Teuchos: Common Tools Package
//                 Copyright (2004) Sandia Corporation
//
// Under terms of Contract DE-AC04-94AL85000, there is a non-exclusive
// license for use of this work by or on behalf of the U.S. Government.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Michael A. Heroux (maherou@sandia.gov)
//
// ***********************************************************************
// @HEADER

#ifndef TEUCHOS_ASSERT_HPP
#define TEUCHOS_ASSERT_HPP


#include "Teuchos_TestForException.hpp"


/** \brief This macro is throws when an assert fails.
 *
 * \note <tt>The std::exception</tt> thrown is <tt>std::logic_error</tt>.
 *
 * \ingroup TestForException_grp
 */
#define TEUCHOS_ASSERT(assertion_test) TEUCHOS_TEST_FOR_EXCEPT(!(assertion_test))


/** \brief This macro asserts that an integral number fallis in the range
 * <tt>[lower_inclusive,upper_exclusive)</tt>
 *
 * \note <tt>The std::exception</tt> thrown is <tt>std::out_of_range</tt>.
 *
 * WARNING: This assert will evaluate <tt>index</tt>,
 * <tt>lower_inclusive</tt>, and <tt>upper_inclusive</tt> more than once if
 * there is a failure which will cause the side-effect of an additional
 * evaluation.  This is needed because the return types of these values are
 * unknown.  Therefore, only pass in arguments that are objects or function
 * calls that have not side-effects!
 *
 * \ingroup TestForException_grp
 */
#define TEUCHOS_ASSERT_IN_RANGE_UPPER_EXCLUSIVE( index, lower_inclusive, upper_exclusive ) \
  { \
    TEUCHOS_TEST_FOR_EXCEPTION( \
      !( (lower_inclusive) <= (index) && (index) < (upper_exclusive) ), \
      std::out_of_range, \
      "Error, the index " #index " = " << (index) << " does not fall in the range" \
      "["<<(lower_inclusive)<<","<<(upper_exclusive)<<")!" ); \
  }


/** \brief This macro is checks that to numbers are equal and if not then
 * throws an exception with a good error message.
 *
 * \note The <tt>std::exception</tt> thrown is <tt>std::out_of_range</tt>.
 *
 * WARNING: This assert will evaluate <tt>val1</tt> and <tt>val2</tt> more
 * than once if there is a failure which will cause the side-effect of an
 * additional evaluation.  This is needed because the return types of
 * <tt>val1</tt> and <tt>val2</tt> are unknown.  Therefore, only pass in
 * arguments that are objects or function calls that have not side-effects!
 *
 * \ingroup TestForException_grp
 */
#define TEUCHOS_ASSERT_EQUALITY( val1, val2 ) \
  { \
    TEUCHOS_TEST_FOR_EXCEPTION( \
      (val1) != (val2), std::out_of_range, \
      "Error, (" #val1 " = " << (val1) << ") != (" #val2 " = " << (val2) << ")!" ); \
  }


/** \brief This macro is checks that an inequality between two numbers is
 * satisified and if not then throws a good exception message.
 *
 * \note The <tt>std::exception</tt> thrown is <tt>std::out_of_range</tt>.
 *
 * WARNING: This assert will evaluate <tt>val1</tt> and <tt>val2</tt> more
 * than once if there is a failure which will cause the side-effect of an
 * additional evaluation.  This is needed because the return types of
 * <tt>val1</tt> and <tt>val2</tt> are unknown.  Therefore, only pass in
 * arguments that are objects or function calls that have not side-effects!
 *
 * \ingroup TestForException_grp
 */
#define TEUCHOS_ASSERT_INEQUALITY( val1, comp, val2 ) \
  { \
    TEUCHOS_TEST_FOR_EXCEPTION( \
      !( (val1) comp (val2) ), std::out_of_range, \
      "Error, (" #val1 " = " << (val1) << ") " \
      #comp " (" #val2 " = " << (val2) << ")! FAILED!" ); \
  }


#endif // TEUCHOS_ASSERT_HPP
