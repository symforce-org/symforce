/*
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
*/

#ifndef TEUCHOS_CONFIGDEFS_HPP
#define TEUCHOS_CONFIGDEFS_HPP

/*! \file Teuchos_ConfigDefs.hpp
    \brief Teuchos header file which uses auto-configuration information 
	to include necessary C++ headers.
*/

#include "Teuchos_config.h"

#ifdef HAVE_TEUCHOS_DEBUG
#  define TEUCHOS_DEBUG
#  define HAVE_TEUCHOS_ARRAY_BOUNDSCHECK
#endif

#ifdef __cplusplus

#if defined(_MSC_VER) || defined(__APPLE__)
#  define TEUCHOS_NO_ZERO_ITERATOR_CONVERSION
#endif

#if defined(__IBMC__) || defined(__IBMCPP__)
#  ifndef TEMPLATE_FRIENDS_NOT_SUPPORTED
#    define TEMPLATE_FRIENDS_NOT_SUPPORTED
#  endif
#  ifndef TEUCHOS_PRIVIATE_DELETE_NOT_SUPPORTED
#    define TEUCHOS_PRIVIATE_DELETE_NOT_SUPPORTED
#  endif
#endif

/* Deprecated */
#ifndef HAVE_COMPLEX
#  define HAVE_COMPLEX
#endif

#include <cstdio>
#include <cstdarg>
#include <cerrno>
#include <climits>
#include <cstdlib>
#include <string>
#include <cstring>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cassert>
#include <complex>
#include <map>
#include <vector>
#include <deque>
#include <algorithm>
#include <numeric>
#include <list>
#include <set>
#include <typeinfo>
#include <limits>
#include <memory>
#include <cstddef>

/* Avoid duplicating instantiation provided by IBM XL C++ runtime library. */
#if defined(__IBMCPP__)
# pragma do_not_instantiate std::fpos<mbstate_t>
#endif

namespace Teuchos { class DummyDummyClass; }
// Above, is used for a dumb reason (see
// Teuchs_StandardMemberCompositionMacros.hpp).

const double Teuchos_MinDouble = 1.0E-100;
const double Teuchos_MaxDouble = 1.0E+100;
const double Teuchos_Overflow = 1.79E308; // Used to test if equilibration should be done.
const double Teuchos_Underflow = 2.23E-308;

// 2007/06/29: These are hacks for std::ostringstream that should be removed
// now what we assume that a faily complete standard C++ library is available.

#define TEUCHOS_OSTRINGSTREAM_GET_C_STR(OSS) (OSS).str().c_str()
typedef std::ostringstream TeuchosOStringStream;

#else /* __cplusplus */

#include <stddef.h>

#endif /* __cplusplus */

/* Delete any previous definition of TEUCHOS_NO_ERROR_REPORTS */

#ifdef TEUCHOS_CHK_ERR
#undef TEUCHOS_CHK_ERR
#endif
#ifdef TEUCHOS_CHK_PTR
#undef TEUCHOS_CHK_PTR
#endif
#ifdef TEUCHOS_CHK_REF
#undef TEUCHOS_CHK_REF
#endif

/* The integral type that is used for the largest ordinal values on this
 * machine.
 *
 * On a 32 bit machine, ptrdiff_t will be an unsighed 32 bit integer and on a
 * 64 bit machine it will be an unsigned 64 bit integer.  Just what I want!
*/
typedef TEUCHOS_ORDINAL_TYPE Teuchos_Ordinal;

#ifdef __cplusplus
namespace Teuchos { typedef Teuchos_Ordinal Ordinal; }
#endif /* __cplusplus */

/* Deprecated (use Teuchos_Ordinal) */
typedef Teuchos_Ordinal Teuchos_Index;

/* Make error report silent by defining TEUCHOS_NO_ERROR_REPORTS */

#define TEUCHOS_CHK_ERR(a) { if (a != 0)  return(a);}
#define TEUCHOS_CHK_PTR(a) { return(a);}
#define TEUCHOS_CHK_REF(a) { return(a);}

#ifdef __cplusplus
const int Teuchos_DefaultTracebackMode = 1; /* Default value for traceback behavior */
#endif /* __cplusplus */

/* Define some macros */
#define TEUCHOS_MAX(x,y) (( (x) > (y) ) ? (x)  : (y) )     /* max function  */
#define TEUCHOS_MIN(x,y) (( (x) < (y) ) ? (x)  : (y) )     /* min function  */
#define TEUCHOS_SGN(x)   (( (x) < 0.0 ) ? -1.0 : 1.0 )     /* sign function */

#ifndef HAVE_FORTRAN_SUPPORT
#  ifndef FORTRAN_DISABLED
#    define FORTRAN_DISABLED
#  endif
#endif

#include "Teuchos_DLLExportMacro.h"

#endif /* TEUCHOS_CONFIGDEFS_HPP */
