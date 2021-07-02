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

#ifndef TEUCHOS_TEST_FOR_EXCEPTION_H
#define TEUCHOS_TEST_FOR_EXCEPTION_H

/*! \file Teuchos_TestForException.hpp
\brief Standard test and throw macros.
*/

#include "Teuchos_TypeNameTraits.hpp"
#include "Teuchos_stacktrace.hpp"


namespace Teuchos {


/*! \defgroup TestForException_grp Utility code for throwing exceptions and setting breakpoints. 
\ingroup teuchos_language_support_grp
*/

/** \brief Increment the throw number.  \ingroup TestForException_grp */
TEUCHOS_LIB_DLL_EXPORT void TestForException_incrThrowNumber();

/** \brief Increment the throw number.  \ingroup TestForException_grp */
TEUCHOS_LIB_DLL_EXPORT int TestForException_getThrowNumber();

/** \brief The only purpose for this function is to set a breakpoint.
    \ingroup TestForException_grp */
TEUCHOS_LIB_DLL_EXPORT void TestForException_break( const std::string &msg );

/** \brief Set at runtime if stacktracing functionality is enabled when *
    exceptions are thrown.  \ingroup TestForException_grp */
TEUCHOS_LIB_DLL_EXPORT void TestForException_setEnableStacktrace(bool enableStrackTrace);

/** \brief Get at runtime if stacktracing functionality is enabled when
 * exceptions are thrown. */
TEUCHOS_LIB_DLL_EXPORT bool TestForException_getEnableStacktrace();


} // namespace Teuchos


#ifdef HAVE_TEUCHOS_STACKTRACE
#  define TEUCHOS_STORE_STACKTRACE() \
  if (Teuchos::TestForException_getEnableStacktrace()) { \
    Teuchos::store_stacktrace(); \
  }
#else
#  define TEUCHOS_STORE_STACKTRACE()
#endif


/** \brief Macro for throwing an exception with breakpointing to ease debugging
 *
 * \param throw_exception_test [in] Test for when to throw the exception.
 * This can and should be an expression that may mean something to the user.
 * The text verbatim of this expression is included in the formed error
 * string.
 *
 * \param Exception [in] This should be the name of an exception class.  The
 * only requirement for this class is that it have a constructor that accepts
 * an std::string object (as all of the standard exception classes do).
 *
 * \param msg [in] This is any expression that can be included in an output
 * stream operation.  This is useful when buinding error messages on the fly.
 * Note that the code in this argument only gets evaluated if
 * <tt>throw_exception_test</tt> evaluates to <tt>true</tt> when an exception
 * is throw.
 *
 * The way that this macro is intended to be used is to 
 * call it in the source code like a function.  For example,
 * suppose that in a piece of code in the file <tt>my_source_file.cpp</tt>
 * that the exception <tt>std::out_of_range</tt> is thrown if <tt>n > 100</tt>.
 * To use the macro, the source code would contain (at line 225
 * for instance):
 \verbatim

 TEUCHOS_TEST_FOR_EXCEPTION( n > 100, std::out_of_range,
    "Error, n = " << n << is bad" );
 \endverbatim
 * When the program runs and with <tt>n = 125 > 100</tt> for instance,
 * the <tt>std::out_of_range</tt> exception would be thrown with the
 * error message:
 \verbatim

 /home/bob/project/src/my_source_file.cpp:225: n > 100: Error, n = 125 is bad
 \endverbatim
 *
 * In order to debug this, simply open your debugger (gdb for instance),
 * set a break point at <tt>my_soure_file.cpp:225</tt> and then set the condition
 * to break for <tt>n > 100</tt> (e.g. in gdb the command
 * is <tt>cond break_point_number n > 100</tt> and then run the
 * program.  The program should stop a the point in the source file
 * right where the exception will be thrown at but before the exception
 * is thrown.  Try not to use expression for <tt>throw_exception_test</tt> that
 * includes virtual function calls, etc. as most debuggers will not be able to check
 * these types of conditions in order to stop at a breakpoint.  For example,
 * instead of:
 \verbatim

 TEUCHOS_TEST_FOR_EXCEPTION( obj1->val() > obj2->val(), std::logic_error, "Oh no!" );
 \endverbatim
 * try:
 \verbatim

 double obj1_val = obj1->val(), obj2_val = obj2->val();
 TEUCHOS_TEST_FOR_EXCEPTION( obj1_val > obj2_val, std::logic_error, "Oh no!" );
 \endverbatim
 * If the developer goes to the line in the source file that is contained
 * in the error message of the exception thrown, he/she will see the
 * underlying condition.
 *
 * As an alternative, you can set a breakpoint for any exception thrown
 * by setting a breakpoint in the function <tt>ThrowException_break()</tt>.
 *
 * NOTE: This macro will only evaluate <tt>throw_exception_test</tt> once
 * reguardless if the test fails and the exception is thrown or
 * not. Therefore, it is safe to call a function with side-effects as the
 * <tt>throw_exception_test</tt> argument.
 *
 * NOTE: This macro will result in creating a stacktrace snapshot in some
 * cases (see the main doc page for details) and will be printed automatically
 * when main() uses TEUCHOS_STANDARD_CATCH_STATEMENTS() to catch uncaught
 * excpetions.
 *
 * \ingroup TestForException_grp
 */
#define TEUCHOS_TEST_FOR_EXCEPTION(throw_exception_test, Exception, msg) \
{ \
  const bool throw_exception = (throw_exception_test); \
  if(throw_exception) { \
    Teuchos::TestForException_incrThrowNumber(); \
    std::ostringstream omsg; \
    omsg \
      << __FILE__ << ":" << __LINE__ << ":\n\n" \
      << "Throw number = " << Teuchos::TestForException_getThrowNumber() \
      << "\n\n" \
      << "Throw test that evaluated to true: "#throw_exception_test \
      << "\n\n" \
      << msg; \
    const std::string &omsgstr = omsg.str(); \
    TEUCHOS_STORE_STACKTRACE(); \
    Teuchos::TestForException_break(omsgstr); \
    throw Exception(omsgstr); \
  } \
}


/** \brief Macro for throwing an exception from within a class method with
 * breakpointing to ease debugging.
 *
 * \param throw_exception_test [in] Test for when to throw the exception.
 * This can and should be an expression that may mean something to the user.
 * The text verbatim of this expression is included in the formed error
 * string.
 *
 * \param Exception [in] This should be the name of an exception class.  The
 * only requirement for this class is that it have a constructor that accepts
 * an std::string object (as all of the standard exception classes do).
 *
 * \param msg [in] This is any expression that can be included in an output
 * stream operation.  This is useful when buinding error messages on the fly.
 * Note that the code in this argument only gets evaluated if
 * <tt>throw_exception_test</tt> evaluates to <tt>true</tt> when an exception
 * is throw.
 *
 * \param tfecfFuncName [implicit] This is a variable in the current scope that is 
 * required to exist and assumed to contain the name of the current class method. 
 * 
 * \param this [implicit] This is the variable (*this), used for printing the
 * typename of the enclosing class.
 *
 * The way that this macro is intended to be used is to call it from a member
 * of of a class. It is used similarly to TEUCHOS_TEST_FOR_EXCEPTION, except that it
 * assumes that the (above) variables <tt>this</tt> and <tt>fecfFuncName</tt>
 * exist and are properly defined. Example usage is:
 
 \code

   std::string tfecfFuncName("someMethod");
    TEUCHOS_TEST_FOR_EXCEPTION_CLASS_FUNC( test, std::runtime_error,
      ": can't call this method in that way.");

 \endcode

 * See <tt>TEUCHOS_TEST_FOR_EXCEPTION()</tt> for more details.
 *
 * \ingroup TestForException_grp
 */
#define TEUCHOS_TEST_FOR_EXCEPTION_CLASS_FUNC(throw_exception_test, Exception, msg) \
{ \
   TEUCHOS_TEST_FOR_EXCEPTION( (throw_exception_test), Exception, \
   typeName(*this) << "::" << tfecfFuncName << msg ) \
}


/** \brief Macro for throwing an exception with breakpointing to ease debugging
 *
 * This macro is equivalent to the <tt>TEUCHOS_TEST_FOR_EXCEPTION()</tt> macro except
 * the file name, line number, and test condition are not printed.
 *
 * \ingroup TestForException_grp
 */
#define TEUCHOS_TEST_FOR_EXCEPTION_PURE_MSG(throw_exception_test, Exception, msg) \
{ \
    const bool throw_exception = (throw_exception_test); \
    if(throw_exception) { \
      Teuchos::TestForException_incrThrowNumber(); \
      std::ostringstream omsg; \
	    omsg << msg; \
      omsg << "\n\nThrow number = " << Teuchos::TestForException_getThrowNumber() << "\n\n"; \
      const std::string &omsgstr = omsg.str(); \
      Teuchos::TestForException_break(omsgstr); \
      TEUCHOS_STORE_STACKTRACE(); \
      throw Exception(omsgstr); \
    } \
}


/** \brief This macro is the same as <tt>TEUCHOS_TEST_FOR_EXCEPTION()</tt> except that the
 * exception will be caught, the message printed, and then rethrown.
 *
 * \param throw_exception_test [in] See <tt>TEUCHOS_TEST_FOR_EXCEPTION()</tt>.
 *
 * \param Exception [in] See <tt>TEUCHOS_TEST_FOR_EXCEPTION()</tt>.
 *
 * \param msg [in] See <tt>TEUCHOS_TEST_FOR_EXCEPTION()</tt>.
 *
 * \param out_ptr [in] If <tt>out_ptr!=NULL</tt> then <tt>*out_ptr</tt> will
 * receive a printout of a line of output that gives the exception type and
 * the error message that is generated.
 *
 * See <tt>TEUCHOS_TEST_FOR_EXCEPTION()</tt> for more details.
 *
 * \ingroup TestForException_grp
 */
#define TEUCHOS_TEST_FOR_EXCEPTION_PRINT(throw_exception_test, Exception, msg, out_ptr) \
try { \
  TEUCHOS_TEST_FOR_EXCEPTION(throw_exception_test, Exception, msg); \
} \
catch(const std::exception &except) { \
  std::ostream *l_out_ptr = (out_ptr); \
  if(l_out_ptr) { \
    *l_out_ptr \
      << "\nThrowing an std::exception of type \'"<<Teuchos::typeName(except) \
      <<"\' with the error message: " \
      << except.what(); \
  } \
  throw; \
}


/** \brief This macro is designed to be a short version of
 * <tt>TEUCHOS_TEST_FOR_EXCEPTION()</tt> that is easier to call.
 *
 * \param throw_exception_test [in] Test for when to throw the exception.
 * This can and should be an expression that may mean something to the user.
 * The text verbatim of this expression is included in the formed error
 * string.
 *
 * \note The exception thrown is <tt>std::logic_error</tt>.
 *
 * \ingroup TestForException_grp
 */
#define TEUCHOS_TEST_FOR_EXCEPT(throw_exception_test) \
  TEUCHOS_TEST_FOR_EXCEPTION(throw_exception_test, std::logic_error, "Error!")


/** \brief This macro is designed to be a short version of
 * <tt>TEUCHOS_TEST_FOR_EXCEPTION()</tt> that is easier to call.
 *
 * \param throw_exception_test [in] Test for when to throw the exception.
 * This can and should be an expression that may mean something to the user.
 * The text verbatim of this expression is included in the formed error
 * string.
 *
 * \param msg [in] The error message.
 *
 * \note The exception thrown is <tt>std::logic_error</tt>.
 *
 * See <tt>TEUCHOS_TEST_FOR_EXCEPTION()</tt> for more details.
 *
 * \ingroup TestForException_grp
 */
#define TEUCHOS_TEST_FOR_EXCEPT_MSG(throw_exception_test, msg) \
  TEUCHOS_TEST_FOR_EXCEPTION(throw_exception_test, std::logic_error, msg)


/** \brief This macro is the same as <tt>TEUCHOS_TEST_FOR_EXCEPT()</tt> except that the
 * exception will be caught, the message printed, and then rethrown.
 *
 * \param throw_exception_test [in] See <tt>TEUCHOS_TEST_FOR_EXCEPT()</tt>.
 *
 * \param out_ptr [in] If <tt>out_ptr!=NULL</tt> then <tt>*out_ptr</tt> will
 * receive a printout of a line of output that gives the exception type and
 * the error message that is generated.
 *
 * See <tt>TEUCHOS_TEST_FOR_EXCEPTION()</tt> for more details.
 *
 * \ingroup TestForException_grp
 */
#define TEUCHOS_TEST_FOR_EXCEPT_PRINT(throw_exception_test, out_ptr) \
  TEUCHOS_TEST_FOR_EXCEPTION_PRINT(throw_exception_test, std::logic_error, "Error!", out_ptr)


/** \brief This macro intercepts an exception, prints a standardized message
 * including the current filename and line number, and then throws the
 * exception up the stack.
 *
 * \param exc [in] the exception that has been caught
 *
 * \ingroup TestForException_grp
 */
#define TEUCHOS_TRACE(exc)\
{ \
  std::ostringstream omsg; \
	omsg << exc.what() << std::endl \
       << "caught in " << __FILE__ << ":" << __LINE__ << std::endl ; \
  throw std::runtime_error(omsg.str()); \
}


//
// Deprecated functions
//


/** \brief Deprecated. */
TEUCHOS_DEPRECATED inline
void TestForException_incrThrowNumber()
{
  Teuchos::TestForException_incrThrowNumber();
}


/** \brief Deprecated. */
TEUCHOS_DEPRECATED inline
int TestForException_getThrowNumber()
{
  return Teuchos::TestForException_getThrowNumber();
}


/** \brief Deprecated. */
TEUCHOS_DEPRECATED inline
void TestForException_break( const std::string &msg )
{
  Teuchos::TestForException_break(msg);
}


/** \brief Deprecated. */
TEUCHOS_DEPRECATED inline
void TestForException_setEnableStacktrace(bool enableStrackTrace)
{
  Teuchos::TestForException_setEnableStacktrace(enableStrackTrace);
}


/** \brief Deprecated. */
TEUCHOS_DEPRECATED inline
bool TestForException_getEnableStacktrace()
{
  return Teuchos::TestForException_getEnableStacktrace();
}


//
// Deprecated macros
//
// NOTE: You can't deprecate macros but you can deprecate functions that
// deprecated macros can call!  This way, as people get a fairly good
// deprecated warning when they use the non-namespaced macros.
//


TEUCHOS_DEPRECATED inline void TEST_FOR_EXCEPTION_this_macro_is_deprecated() {}
/** \brief Deprecated. */
#define TEST_FOR_EXCEPTION(throw_exception_test, Exception, msg) \
  { \
    TEST_FOR_EXCEPTION_this_macro_is_deprecated(); \
    TEUCHOS_TEST_FOR_EXCEPTION(throw_exception_test, Exception, msg); \
  }


TEUCHOS_DEPRECATED inline void TEST_FOR_EXCEPTION_CLASS_FUNC_this_macro_is_deprecated() {}
/** \brief Deprecated. */
#define TEST_FOR_EXCEPTION_CLASS_FUNC(throw_exception_test, Exception, msg) \
  { \
    TEST_FOR_EXCEPTION_CLASS_FUNC_this_macro_is_deprecated(); \
    TEUCHOS_TEST_FOR_EXCEPTION_CLASS_FUNC(throw_exception_test, Exception, msg); \
  }


TEUCHOS_DEPRECATED inline void TEST_FOR_EXCEPTION_PURE_MSG_this_macro_is_deprecated() {}
/** \brief Deprecated. */
#define TEST_FOR_EXCEPTION_PURE_MSG(throw_exception_test, Exception, msg) \
  { \
    TEST_FOR_EXCEPTION_PURE_MSG_this_macro_is_deprecated(); \
    TEUCHOS_TEST_FOR_EXCEPTION_PURE_MSG(throw_exception_test, Exception, msg); \
  }


TEUCHOS_DEPRECATED inline void TEST_FOR_EXCEPTION_PRINT_this_macro_is_deprecated() {}
/** \brief Deprecated. */
#define TEST_FOR_EXCEPTION_PRINT(throw_exception_test, Exception, msg, out_ptr) \
  { \
    TEST_FOR_EXCEPTION_PRINT_this_macro_is_deprecated(); \
    TEUCHOS_TEST_FOR_EXCEPTION_PRINT(throw_exception_test, Exception, msg, out_ptr); \
  }


TEUCHOS_DEPRECATED inline void TEST_FOR_EXCEPT_this_macro_is_deprecated() {}
/** \brief Deprecated. */
#define TEST_FOR_EXCEPT(throw_exception_test) \
  { \
    TEST_FOR_EXCEPT_this_macro_is_deprecated(); \
    TEUCHOS_TEST_FOR_EXCEPT(throw_exception_test); \
  }


TEUCHOS_DEPRECATED inline void TEST_FOR_EXCEPT_MSG_this_macro_is_deprecated() {}
/** \brief Deprecated. */
#define TEST_FOR_EXCEPT_MSG(throw_exception_test, msg) \
  { \
    TEST_FOR_EXCEPT_MSG_this_macro_is_deprecated(); \
    TEUCHOS_TEST_FOR_EXCEPT_MSG(throw_exception_test, msg); \
  }


TEUCHOS_DEPRECATED inline void TEST_FOR_EXCEPT_PRINT_this_macro_is_deprecated() {}
/** \brief Deprecated. */
#define TEST_FOR_EXCEPT_PRINT(throw_exception_test, out_ptr) \
  { \
    TEST_FOR_EXCEPT_PRINT_this_macro_is_deprecated(); \
    TEUCHOS_TEST_FOR_EXCEPT_PRINT(throw_exception_test, out_ptr); \
  }


#endif // TEUCHOS_TEST_FOR_EXCEPTION_H
