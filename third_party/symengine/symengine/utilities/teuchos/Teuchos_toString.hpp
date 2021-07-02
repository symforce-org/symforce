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

#ifndef TEUCHOS_TO_STRING_HPP
#define TEUCHOS_TO_STRING_HPP

#include "Teuchos_ConfigDefs.hpp"


namespace Teuchos {


/** \brief Default traits class for converting objects into strings.
 *
 * NOTE: This default implementation relies on opeator<<(std::ostream&, ...) 
 * being defined for the data type T.
 *
 * \ingroup teuchos_language_support_grp
 */
template<typename T>
class ToStringTraits {
public:
  static std::string toString( const T &t )
    {
      std::ostringstream oss;
      oss << t;
      return oss.str();
    }
};


/** \brief Utility function for returning a pretty string representation of
 * a object of type T.
 *
 * NOTE: This helper function simply returns ToStringTraits<T>::toString(t)
 * and the right way to speicalize the behavior is to specialize
 * ToStringTraits.
 *
 * \ingroup teuchos_language_support_grp
 */
template<typename T>
inline
std::string toString(const T& t)
{
  return ToStringTraits<T>::toString(t);
}


/** \brief Specialization for bool. */
template<>
class ToStringTraits<bool> {
public:
  static std::string toString( const bool &t )
    {
      if (t)
        return "true";
      return "false";
    }
};


/** \brief Specialization for std::string. */
template<>
class ToStringTraits<std::string> {
public:
  static std::string toString( const std::string &t )
    {
      return t;
    }
};


} // end namespace Teuchos


#endif // TEUCHOS_TO_STRING_HPP
