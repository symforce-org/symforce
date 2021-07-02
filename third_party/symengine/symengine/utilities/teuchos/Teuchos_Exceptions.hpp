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


#ifndef TEUCHOS_EXCEPTIONS_HPP
#define TEUCHOS_EXCEPTIONS_HPP


#include "Teuchos_ConfigDefs.hpp"


namespace Teuchos {


/** \brief Base exception class for Teuchos
 *
 * \ingroup teuchos_mem_mng_grp
 */
class ExceptionBase : public std::logic_error
{public:ExceptionBase(const std::string& what_arg) : std::logic_error(what_arg) {}};
// 2007/11/07: rabartl: Above, I had to change the name from Exception to
// ExceptionBase because Marzio did a 'using namespace Teuchos' and then he
// declared his own Exception class.  The file Laplacian3D.cpp failed to
// compile.  STOP DOING USING NAMESPACE BLAH!!!!!!


/** \brief Thrown if a duplicate owning RCP is creatd the the same object.
 *
 * \ingroup teuchos_mem_mng_grp
 */
class DuplicateOwningRCPError : public ExceptionBase
{public:DuplicateOwningRCPError(const std::string& what_arg) : ExceptionBase(what_arg) {}};


/** \brief Null reference error exception class.
 *
 * \ingroup teuchos_mem_mng_grp
 */
class NullReferenceError : public ExceptionBase
{public:NullReferenceError(const std::string& what_arg) : ExceptionBase(what_arg) {}};


/** \brief Null reference error exception class.
 *
 * \ingroup teuchos_mem_mng_grp
 */
class NonconstAccessError : public ExceptionBase
{public:NonconstAccessError(const std::string& what_arg) : ExceptionBase(what_arg) {}};


/** \brief Range error exception class.
 *
 * \ingroup teuchos_mem_mng_grp
 */
class RangeError : public ExceptionBase
{public:RangeError(const std::string& what_arg) : ExceptionBase(what_arg) {}};


/** \brief Dangling reference error exception class.
 *
 * \ingroup teuchos_mem_mng_grp
 */
class DanglingReferenceError : public ExceptionBase
{public:DanglingReferenceError(const std::string& what_arg) : ExceptionBase(what_arg) {}};


/** \brief Incompatiable iterators error exception class.
 *
 * \ingroup teuchos_mem_mng_grp
 */
class IncompatibleIteratorsError : public ExceptionBase
{public:IncompatibleIteratorsError(const std::string& what_arg) : ExceptionBase(what_arg) {}};

/** \brief Thrown when a Parameter Entry that is already being tracked
 * is attempted to be inserted again into the masterParameterEntryMap
 * and masterIDMap
 * 
 * \relates \c ParameterEntry
 */
class DuplicateParameterEntryException : public ExceptionBase {

public:
  DuplicateParameterEntryException(const std::string& what_arg):
    ExceptionBase(what_arg){}
    
};

/** \brief Thrown when a Parameter Entry ID that is already being used
 * is attempted to be reused again.
 * 
 * \relates \c ParameterEntry
 */
class DuplicateParameterEntryIDException : public ExceptionBase {

public:
  DuplicateParameterEntryIDException(const std::string& what_arg):
    ExceptionBase(what_arg){}
    
};

/** \brief Thrown when a ParameterEntryValidatorID that 
 * is already being used is attempted to be reused again.
 * 
 * \relates ParameterEntryValidator
 */
class DuplicateValidatorIDException : public ExceptionBase {

public:
  DuplicateValidatorIDException(const std::string& what_arg):
    ExceptionBase(what_arg){}
    
};


} // end namespace Teuchos


#endif	// TEUCHOS_EXCEPTIONS_HPP
