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

#ifndef TEUCHOS_GET_BASE_OBJ_VOID_PTR
#define TEUCHOS_GET_BASE_OBJ_VOID_PTR


#include "Teuchos_ConfigDefs.hpp"


#if defined(HAVE_TEUCHOS_BOOST) && defined(HAS_TEUCHOS_BOOST_IS_POLYMORPHIC)


// Internal Trilinos code should check if the macro is defined to see if
// getBaseObjPtr() is supported or not.
#define HAS_TEUCHOS_GET_BASE_OBJ_VOID_PTR 1


#include <boost/type_traits/is_polymorphic.hpp>


namespace Teuchos {


template<bool isPolymorphic, typename T>
class GetBaseObjVoidPtrImpl {};

template<typename T>
class GetBaseObjVoidPtrImpl<true, T> {
public:
  static const void* getBaseObjVoidPtr(T *p)
    {
      return dynamic_cast<const void*>(p);
    } 
};


template<typename T>
class GetBaseObjVoidPtrImpl<false, T> {
public:
  static const void* getBaseObjVoidPtr(T *p)
    {
      return static_cast<const void*>(p);
    }
};


/** \brief Return a const void* pointing to the base of an object.
 *
 * This function uses the boost::is_polymorphic traits class to determine if
 * type T is a polymorphic type or a non-polymorphic and then calls
 * dynamic_cast or static_cast, respectively, to return the base pointer to
 * the object.
 *
 * The base pointer to an object is crtical to know if you need to determine
 * if two pointers are pointing to the same object or not.
 *
 * NOTE: This function will not even be defined if
 * HAS_TEUCHOS_GET_BASE_OBJ_VOID_PTR is not defined (which currently requires
 * boost support but that may change later).
 *
 * \ingroup teuchos_language_support_grp
 */
template<typename T>
const void* getBaseObjVoidPtr(T *p)
{
  typedef GetBaseObjVoidPtrImpl<boost::is_polymorphic<T>::value, T> GBOVPT;
  return GBOVPT::getBaseObjVoidPtr(p);
}


}	// end namespace Teuchos


#endif // defined(HAVE_TEUCHOS_BOOST) && defined(HAS_TEUCHOS_BOOST_IS_POLYMORPHIC)


#endif // TEUCHOS_GET_BASE_OBJ_VOID_PTR
