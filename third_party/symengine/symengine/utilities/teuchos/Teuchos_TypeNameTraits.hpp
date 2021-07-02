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

#ifndef _TEUCHOS_TYPE_NAME_TRAITS_HPP_
#define _TEUCHOS_TYPE_NAME_TRAITS_HPP_

/*! \file Teuchos_TypeNameTraits.hpp
 \brief Defines basic traits returning the
    name of a type in a portable and readable way.
*/

#include "Teuchos_ConstTypeTraits.hpp"

#if defined(__IBMCPP__) && __IBMCPP__ < 900
# define TEUCHOS_TYPE_NAME_TRAITS_OLD_IBM
#endif

namespace  Teuchos {


/** \brief Demangle a C++ name if valid.
 *
 * The name must have come from <tt>typeid(...).name()</tt> in order to be
 * valid name to pass to this function.
 *
 * \ingroup teuchos_language_support_grp
 */
TEUCHOS_LIB_DLL_EXPORT std::string demangleName( const std::string &mangledName );


/** \brief Default traits class that just returns <tt>typeid(T).name()</tt>.
 *
 * \ingroup teuchos_language_support_grp
 */
template<typename T>
class TypeNameTraits {
public:
  /** \brief . */
  static std::string name()
    {
      return demangleName(typeid(T).name());
    }
  /** \brief . */
#ifndef TEUCHOS_TYPE_NAME_TRAITS_OLD_IBM
  static std::string concreteName( const T& t )
#else
  // the IBM compilers on AIX have a problem with const
  static std::string concreteName( T t )
#endif
    {
      return demangleName(typeid(t).name());
    }
};


/** \brief Template function for returning the concrete type name of a
 * passed-in object.
 *
 * Uses the traits class TypeNameTraits so the behavior of this function can
 * be specialized in every possible way.  The default return value is
 * typically derived from <tt>typeid(t).name()</tt>.
 *
 * \ingroup teuchos_language_support_grp
 */
template<typename T>
std::string typeName( const T &t )
{
  typedef typename ConstTypeTraits<T>::NonConstType ncT;
#ifndef TEUCHOS_TYPE_NAME_TRAITS_OLD_IBM
  return TypeNameTraits<ncT>::concreteName(t);
#else
  // You can't pass general objects to AIX by value as above.  This means that
  // you will not get the concrete name printed on AIX but that is life on
  // such compilers.
  return TypeNameTraits<ncT>::name();
#endif
}


/** \brief Template function for returning the type name of the actual
 * concrete name of a passed-in object.
 *
 * Uses the traits class TypeNameTraits so the behavior of this function can
 * be specialized in every possible way.
 *
 * \ingroup teuchos_language_support_grp
 */
template<typename T>
std::string concreteTypeName( const T &t )
{
  typedef typename ConstTypeTraits<T>::NonConstType ncT;
  return TypeNameTraits<ncT>::concreteName(t);
}


#define TEUCHOS_TYPE_NAME_TRAITS_BUILTIN_TYPE_SPECIALIZATION(TYPE) \
template<> \
class TEUCHOS_LIB_DLL_EXPORT TypeNameTraits<TYPE> { \
public: \
  static std::string name() { return (#TYPE); } \
  static std::string concreteName(const TYPE&) { return name(); } \
} \

TEUCHOS_TYPE_NAME_TRAITS_BUILTIN_TYPE_SPECIALIZATION(bool);
TEUCHOS_TYPE_NAME_TRAITS_BUILTIN_TYPE_SPECIALIZATION(char);
TEUCHOS_TYPE_NAME_TRAITS_BUILTIN_TYPE_SPECIALIZATION(int);
TEUCHOS_TYPE_NAME_TRAITS_BUILTIN_TYPE_SPECIALIZATION(short int);
TEUCHOS_TYPE_NAME_TRAITS_BUILTIN_TYPE_SPECIALIZATION(long int);
TEUCHOS_TYPE_NAME_TRAITS_BUILTIN_TYPE_SPECIALIZATION(float);
TEUCHOS_TYPE_NAME_TRAITS_BUILTIN_TYPE_SPECIALIZATION(double);


template<typename T>
class TEUCHOS_LIB_DLL_EXPORT TypeNameTraits<T*> {
public:
  typedef T* T_ptr;
  static std::string name() { return TypeNameTraits<T>::name() + "*"; }
  static std::string concreteName(T_ptr) { return name(); }
};


template<>
class TEUCHOS_LIB_DLL_EXPORT TypeNameTraits<std::string> {
public:
  static std::string name() { return "string"; }
  static std::string concreteName(const std::string&)
    { return name(); }
};


template<>
class TEUCHOS_LIB_DLL_EXPORT TypeNameTraits<void*> {
public:
  static std::string name() { return "void*"; }
  static std::string concreteName(const std::string&) { return name(); }
};


#ifdef HAVE_TEUCHOS_COMPLEX


template<typename T>
class TEUCHOS_LIB_DLL_EXPORT TypeNameTraits<std::complex<T> > {
public:
  static std::string name()
    { return "complex<"+TypeNameTraits<T>::name()+">"; }
  static std::string concreteName(const std::complex<T>&)
    { return name(); }
};


#endif // HAVE_TEUCHOS_COMPLEX

 

} // namespace Teuchos


#endif // _TEUCHOS_TYPE_NAME_TRAITS_HPP_
