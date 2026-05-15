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

#include "Teuchos_TestForException.hpp"

#include <iostream>


//
// ToDo: Make these functions thread-safe!
//


namespace {


int throwNumber = 0;


bool& loc_enableStackTrace()
{
  static bool static_enableStackTrace =
#ifdef HAVE_TEUCHOS_DEFAULT_STACKTRACE
    true
#else
    false
#endif
    ;
  return static_enableStackTrace;
}


} // namespace


void Teuchos::TestForException_incrThrowNumber()
{
  ++throwNumber;
}


int Teuchos::TestForException_getThrowNumber()
{
  return throwNumber;
}


void Teuchos::TestForException_break( const std::string &errorMsg )
{
  int break_on_me;
  break_on_me = errorMsg.length(); // Use errMsg to avoid compiler warning.
  (void)break_on_me;
  // Above is just some statement for the debugger to break on.  Note: now is
  // a good time to examine the stack trace and look at the error message in
  // 'errorMsg' to see what happened.  In GDB just type 'where' or you can go
  // up by typing 'up' and moving up in the stack trace to see where you are
  // and how you got to this point in the code where you are throwning this
  // exception!  Typing in a 'p errorMsg' will show you what the error message
  // is.  Also, you should consider adding a conditional breakpoint in this
  // function based on a specific value of 'throwNumber' if the exception you
  // want to examine is not the first exception thrown.
}


void Teuchos::TestForException_setEnableStacktrace(bool enableStrackTrace)
{
  loc_enableStackTrace() = enableStrackTrace;
}


bool Teuchos::TestForException_getEnableStacktrace()
{
  return loc_enableStackTrace();
}

void Teuchos::TestForTermination_terminate(const std::string &msg) {
    std::cerr << msg;
    std::terminate();
}
