/*
Copyright (c) 2010, Ondrej Certik
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.
* Neither the name of the Sandia Corporation nor the names of its contributors
  may be used to endorse or promote products derived from this software without
  specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef TEUCHOS_STACKTRACE_HPP
#define TEUCHOS_STACKTRACE_HPP

/*! \file Teuchos_stacktrace.hpp

\brief Functions for returning stacktrace info (GCC only initially).
*/

#include "Teuchos_ConfigDefs.hpp"


#ifdef HAVE_TEUCHOS_STACKTRACE


/*! \defgroup TeuchosStackTrace_grp Utility code for generating stacktraces.
 *
 * \ingroup teuchos_language_support_grp
 */

namespace Teuchos {


/** \brief Stores the current stacktrace into an internal global variable.
 *
 * \ingroup TeuchosStackTrace_grp
 */
void store_stacktrace();

/** \brief Returns the last stored stacktrace as a string.
 *
 * \ingroup TeuchosStackTrace_grp
 */
std::string get_stored_stacktrace();

/** \brief Returns the current stacktrace as a string.
 *
 * \param impl_stacktrace_depth [in] The stacktrace depth to remove from the
 * stacktrace printout to avoid showing users implementation functions in the
 * stacktrace.
 *
 * \ingroup TeuchosStackTrace_grp
 */
std::string get_stacktrace(int impl_stacktrace_depth=0);

/** \brief Prints the current stacktrace to stdout.
 *
 * \ingroup TeuchosStackTrace_grp
 */
void show_stacktrace();

/** \brief Prints the current stacktrace to stdout on segfault.
 *
 * \ingroup TeuchosStackTrace_grp
 */
void print_stack_on_segfault();

} // end namespace Teuchos

#endif // HAVE_TEUCHOS_STACKTRACE

#endif // TEUCHOS_STACKTRACE_HPP

