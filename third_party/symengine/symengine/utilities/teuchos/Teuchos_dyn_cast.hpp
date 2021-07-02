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

#ifndef TEUCHOS_DYN_CAST_HPP
#define TEUCHOS_DYN_CAST_HPP


#include "Teuchos_TypeNameTraits.hpp"


namespace Teuchos {


/** \brief Exception class for bad cast.

\ingroup teuchos_language_support_grp

We create this class so that we may throw a bad_cast when appropriate and
still use the TEUCHOS_TEST_FOR_EXCEPTION macro.  We recommend users try to catch a
bad_cast.
*/
class m_bad_cast : public std::bad_cast {
	std::string msg;
public:
	explicit m_bad_cast(const std::string&  what_arg ) : msg(what_arg) {}
	virtual ~m_bad_cast() throw() {}
	virtual const char* what() const throw() { return msg.data(); }
};


// Throw <tt>m_bad_cast</tt> for below function
TEUCHOS_LIB_DLL_EXPORT void dyn_cast_throw_exception(
  const std::string &T_from,
  const std::string &T_from_concr,
  const std::string &T_to
  );


/** \brief Dynamic casting utility function meant to replace
 * <tt>dynamic_cast<T&></tt> by throwing a better documented error
 * message.
 *
 * \ingroup teuchos_language_support_grp
 *
 * Existing uses of the built-in <tt>dynamic_cast<T&>()</tt> operator
 * such as:
 
 \code
 C &c = dynamic_cast<C&>(a);
 \endcode

 * are easily replaced as:

 \code
 C &c = dyn_cast<C>(a);
 \endcode

 * and that is it.  One could write a perl script to do this
 * automatically.
 *
 * This utility function is designed to cast an object reference of
 * type <tt>T_From</tt> to type <tt>T_To</tt> and if the cast fails at
 * runtime then an std::exception (derived from <tt>std::bad_cast</tt>) is
 * thrown that contains a very good error message.
 *
 * Consider the following class hierarchy:

 \code
 class A {};
 class B : public A {};
 class C : public A {};
 \endcode
 *
 * Now consider the following program:
 \code
  int main( int argc, char* argv[] ) {
    B b;
    A &a = b;
    try {
      std::cout << "\nTrying: dynamic_cast<C&>(a);\n";
      dynamic_cast<C&>(a);
    }
    catch( const std::bad_cast &e ) {
      std::cout << "\nCaught std::bad_cast std::exception e where e.what() = \"" << e.what() << "\"\n";
    }
    try {
      std::cout << "\nTrying: Teuchos::dyn_cast<C>(a);\n";
      Teuchos::dyn_cast<C>(a);
    }
    catch( const std::bad_cast &e ) {
      std::cout << "\nCaught std::bad_cast std::exception e where e.what() = \"" << e.what() << "\"\n";
    }
  	return 0;
  }
 \endcode
 
 * The above program will print something that looks like (compiled
 * with g++ for example):

 \verbatim

  Trying: dynamic_cast<C&>(a);

  Caught std::bad_cast std::exception e where e.what() = "St8bad_cast"

  Trying: Teuchos::dyn_cast<C>(a);

  Caught std::bad_cast std::exception e where e.what() = "../../../../packages/teuchos/src/Teuchos_dyn_cast.cpp:46: true:
  dyn_cast<1C>(1A) : Error, the object with the concrete type '1B' (passed in through the interface type '1A')  does
  not support the interface '1C' and the dynamic cast failed!"

 \endverbatim
 
 * The above program shows that the standard implementation of
 * <tt>dynamic_cast<T&>()</tt> does not return any useful debugging
 * information at all but the templated function
 * <tt>Teuchos::dyn_cast<T>()</tt> returns all kinds of useful
 * information.  The generated error message gives the type of the
 * interface that the object was passed in as (i.e. <tt>A</tt>), what
 * the actual concrete type of the object is (i.e. <tt>B</tt>) and
 * what type is trying to be dynamically casted to (i.e. <tt>C</tt>).
 * This type of information is extremely valuable when trying to track
 * down these type of runtime dynamic casting errors.  In some cases
 * (such as with <tt>gdb</tt>), debuggers do not even give the type of
 * concrete object so this function is very important on these
 * platforms.  In many cases, a debugger does not even need to be
 * opened to diagnose what the problem is and how to fix it.
 *
 * Note that this function is inlined and does not incur any
 * significant runtime performance penalty over the raw
 * <tt>dynamic_cast<T&>()</tt> operator.
 */
template <class T_To, class T_From>
inline
T_To& dyn_cast(T_From &from)
{
  T_To *to_ = dynamic_cast<T_To*>(&from);
  if(!to_)
    dyn_cast_throw_exception(
      TypeNameTraits<T_From>::name(),
      typeName(from),
      TypeNameTraits<T_To>::name()
      );
  return *to_;
}


} // namespace Teuchos


#endif // TEUCHOS_DYN_CAST_HPP
