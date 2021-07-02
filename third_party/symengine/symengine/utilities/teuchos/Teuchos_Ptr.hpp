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


#ifndef TEUCHOS_PTR_HPP
#define TEUCHOS_PTR_HPP


#include "Teuchos_PtrDecl.hpp"
#include "Teuchos_RCP.hpp"


namespace Teuchos {


namespace PtrPrivateUtilityPack {
void throw_null( const std::string &type_name );
} // namespace PtrPrivateUtilityPack


template<class T> inline
Ptr<T>::Ptr( ENull /*null_in*/ )
  : ptr_(0)
{}


template<class T> inline
Ptr<T>::Ptr( T *ptr_in )
  : ptr_(ptr_in)
{}


template<class T> inline
Ptr<T>::Ptr(const Ptr<T>& ptr_in)
  :ptr_(ptr_in.ptr_)
{}


template<class T>
template<class T2> inline
Ptr<T>::Ptr(const Ptr<T2>& ptr_in)
  :ptr_(ptr_in.get())
{}


template<class T> inline
Ptr<T>& Ptr<T>::operator=(const Ptr<T>& ptr_in)
{
  ptr_ = ptr_in.get();
  return *this;
}


template<class T> inline
T* Ptr<T>::operator->() const
{
  debug_assert_not_null();
  debug_assert_valid_ptr();
  return ptr_;
}


template<class T> inline
T& Ptr<T>::operator*() const
{
  debug_assert_not_null();
  debug_assert_valid_ptr();
  return *ptr_;
}


template<class T> inline
T* Ptr<T>::get() const
{
  debug_assert_valid_ptr();
  return ptr_;
}


template<class T> inline
T* Ptr<T>::getRawPtr() const
{
  return get();
}


template<class T> inline
const Ptr<T>& Ptr<T>::assert_not_null() const
{
  if(!ptr_)
    PtrPrivateUtilityPack::throw_null(TypeNameTraits<T>::name());
  return *this;
}


template<class T> inline
const Ptr<T> Ptr<T>::ptr() const
{
  return *this;
}


template<class T> inline
Ptr<const T> Ptr<T>::getConst() const
{
  return ptr_implicit_cast<const T>(*this);
}


template<class T> inline
void Ptr<T>::debug_assert_valid_ptr() const
{
#ifdef TEUCHOS_DEBUG
  rcp_.access_private_node().assert_valid_ptr(*this);
#endif
}


#ifdef TEUCHOS_DEBUG


template<class T> inline
Ptr<T>::Ptr( const RCP<T> &p )
  : ptr_(p.getRawPtr()), rcp_(p)
{}


#endif // TEUCHOS_DEBUG


} // namespace Teuchos


template<class T>
std::ostream& Teuchos::operator<<( std::ostream& out, const Ptr<T>& p )
{
  out
    << TypeNameTraits<RCP<T> >::name() << "{"
    << "ptr="<<(const void*)(p.get()) // I can't find any alternative to this C cast :-(
    <<"}";
  return out;
}


#endif // TEUCHOS_PTR_HPP
