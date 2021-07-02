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


#ifndef TEUCHOS_PTR_DECL_HPP
#define TEUCHOS_PTR_DECL_HPP


#include "Teuchos_RCPDecl.hpp"
#include "Teuchos_dyn_cast.hpp"


namespace Teuchos {


/** \brief Simple wrapper class for raw pointers to single objects where no
 * persisting relationship exists.
 *
 * This class is meant to replace all but the lowest-level use of raw pointers
 * that point to single objects where the use of <tt>RCP</tt> is not justified
 * for performance or semantic reasons.  When built in optimized mode, this
 * class should impart little time overhead and should be exactly equivalent
 * in the memory footprint to a raw C++ pointer and the only extra runtime
 * overhead will be the default initalization to NULL.
 *
 * The main advantages of using this class over a raw pointer however are:
 *
 * <ul>
 *
 * <li> <tt>Ptr</tt> objects always default construct to null
 *
 * <li> <tt>Ptr</tt> objects will throw exceptions on attempts to dereference
 * the underlying null pointer when debugging support is compiled in.
 *
 * <li> <tt>Ptr</tt> does not allow array-like operations like
 * <tt>ptr[i]</tt>, <tt>++ptr</tt> or <tt>ptr+i</tt> that can only result in
 * disaster when the a pointer points to only a single object that can not be
 * assumed to be part of an array of objects.
 *
 * <li> <tt>Ptr</tt> is part of a system of types defined in <tt>Teuchos</tt>
 * that keeps your code away from raw pointers which are the cause of most
 * defects in C++ code.
 *
 * </ul>
 *
 * Debugging support is compiled in when the macro <tt>TEUCHOS_DEBUG</tt> is
 * defined which happens automatically when <tt>--enable-teuchos-debug</tt> is
 * specified on the configure line.  When debugging support is not compiled
 * in, the only overhead imparted by this class is it's default initialization
 * to null.  Therefore, this class can provide for very high performance on
 * optimized builds of the code.
 *
 * An implicit conversion from a raw pointer to a <tt>Ptr</tt> object is okay
 * since we don't assume any ownership of the object, hense the constructor
 * taking a raw pointer is not declared explicit.  However, this class does
 * not support an implicit conversion to a raw pointer since we want to limit
 * the exposure of raw pointers in our software.  If we have to convert back
 * to a raw pointer, then we want to make that explicit by calling
 * <tt>get()</tt>.
 *
 * This class should be used to replace most raw uses of C++ pointers to
 * single objects where using the <tt>RCP</tt> class is not appropriate,
 * unless the runtime cost of null-initialization it too expensive.
 */
template<class T>
class Ptr {
public:

  /** \brief Default construct to NULL.
   *
   * <b>Postconditons:</b><ul>
   * <li> <tt>this->get() == NULL</tt>
   * </ul>
   */
  inline Ptr( ENull null_in = null );

  /** \brief Construct given a raw pointer.
   *
   * <b>Postconditons:</b><ul>
   * <li> <tt>this->get() == ptr</tt>
   * </ul>
   *
   * Note: This constructor is declared <tt>explicit</tt> so there is no
   * implicit conversion from a raw C++ pointer to a <tt>Ptr</tt> object.
   * This is meant to avoid cases where an uninitialized pointer is used to
   * implicitly initialize one of these objects.
   */
  inline explicit Ptr( T *ptr );

  /** \brief Copy construct from same type.
   *
   * <b>Postconditons:</b><ul>
   * <li> <tt>this->get() == ptr.get()</tt>
   * </ul>
   */
  inline Ptr(const Ptr<T>& ptr);

  /** \brief Copy construct from another type.
   *
   * <b>Postconditons:</b><ul>
   * <li> <tt>this->get() == ptr.get()</tt> (unless virtual base classes
   *      are involved)
   * </ul>
   */
  template<class T2>
  inline Ptr(const Ptr<T2>& ptr);

  /** \brief Shallow copy of the underlying pointer.
   *
   * <b>Postconditons:</b><ul>
   * <li> <tt>this->get() == ptr.get()</tt>
   * </ul>
   */
  Ptr<T>& operator=(const Ptr<T>& ptr);

  /** \brief Pointer (<tt>-></tt>) access to members of underlying object.
   *
   * <b>Preconditions:</b><ul>
   * <li> <tt>this->get() != NULL</tt> (throws <tt>std::logic_error</tt>)
   * </ul>
   */
  inline T* operator->() const;

  /** \brief Dereference the underlying object.
   *
   * <b>Preconditions:</b><ul>
   * <li> <tt>this->get() != NULL</tt> (throws <tt>std::logic_error</tt>)
   * </ul>
   */
  inline T& operator*() const;

  /** \brief Get the raw C++ pointer to the underlying object. */
  inline T* get() const;

  /** \brief Get the raw C++ pointer to the underlying object. */
  inline T* getRawPtr() const;

  /** \brief Throws <tt>std::logic_error</tt> if <tt>this->get()==NULL</tt>,
   * otherwise returns reference to <tt>*this</tt>.
   */
  inline const Ptr<T>& assert_not_null() const;

  /** \brief Return a copy of *this. */
  inline const Ptr<T> ptr() const;

  /** \brief Return a Ptr<const T> version of *this. */
  inline Ptr<const T> getConst() const;

private:

  T *ptr_;

#ifdef TEUCHOS_DEBUG
  RCP<T> rcp_;
#endif

  void debug_assert_not_null() const
    {
#ifdef TEUCHOS_DEBUG
      assert_not_null();
#endif
    }

  inline void debug_assert_valid_ptr() const;

public: // Bad bad bad

#ifdef TEUCHOS_DEBUG
  Ptr( const RCP<T> &p );
  T* access_private_ptr() const
    { return ptr_; }
  const RCP<T> access_rcp() const
    { return rcp_; }
#endif


};


/** \brief create a non-persisting (required or optional) output
 * argument for a function call.
 *
 * \relates Ptr
 */
template<typename T> inline
Ptr<T> outArg( T& arg )
{
  return Ptr<T>(&arg);
}


/** \brief create a non-persisting (required or optional) input/output
 * argument for a function call.
 *
 * \relates Ptr
 */
template<typename T> inline
Ptr<T> inOutArg( T& arg )
{
  return Ptr<T>(&arg);
}


/** \brief create a non-persisting (required or optional) input/output
 * argument for a function call.
 *
 * \relates Ptr
 */
template<typename T> inline
Ptr<T> inoutArg( T& arg )
{
  return Ptr<T>(&arg);
}


/** \brief create a general <tt>Ptr</tt> input argument for a function call
 * from a reference.
 *
 * \relates Ptr
 */
template<typename T> inline
Ptr<const T> ptrInArg( T& arg )
{
  return Ptr<const T>(&arg);
}


/** \brief create a non-persisting non-const optional input argument
for a function call.
 *
 * \relates Ptr
 */
template<typename T> inline
Ptr<T> optInArg( T& arg )
{
  return Ptr<T>(&arg);
}


/** \brief create a non-persisting const optional input argument for a
function call.
 *
 * \relates Ptr
 */
template<typename T> inline
Ptr<const T> constOptInArg( T& arg )
{
  return Ptr<const T>(&arg);
}


/** \brief Create a pointer to a object from an object reference.
 *
 * \relates Ptr
 */
template<typename T> inline
Ptr<T> ptrFromRef( T& arg )
{
  return Ptr<T>(&arg);
}


/** \brief Create an RCP<T> from a Ptr<T> object.
 *
 * \relates RCP
 */
template<typename T> inline
RCP<T> rcpFromPtr( const Ptr<T>& ptr )
{
  if (is_null(ptr))
    return null;
#ifdef TEUCHOS_DEBUG
  // In a debug build, just grab out the WEAK RCP and return it.  That way we
  // can get dangling reference checking without having to turn on more
  // expensive RCPNode tracing.
  if (!is_null(ptr.access_rcp()))
    return ptr.access_rcp();
#endif
  return rcpFromRef(*ptr);
}


/** \brief Create a pointer to an object from a raw pointer.
 *
 * \relates Ptr
 */
template<typename T> inline
Ptr<T> ptr( T* p )
{
  return Ptr<T>(p);
}


/** \brief Create a pointer from a const object given a non-const object
 * reference.
 *
 * <b>Warning!</b> Do not call this function if <tt>T</tt> is already const or
 * a compilation error will occur!
 *
 * \relates Ptr
 */
template<typename T> inline
Ptr<const T> constPtr( T& arg )
{
  return Ptr<const T>(&arg);
}


/** \brief Returns true if <tt>p.get()==NULL</tt>.
 *
 * \relates Ptr
 */
template<class T> inline
bool is_null( const Ptr<T> &p )
{
  return p.get() == 0;
}


/** \brief Returns true if <tt>p.get()!=NULL</tt>
 *
 * \relates Ptr
 */
template<class T> inline
bool nonnull( const Ptr<T> &p )
{
  return p.get() != 0;
}


/** \brief Returns true if <tt>p.get()==NULL</tt>.
 *
 * \relates Ptr
 */
template<class T> inline
bool operator==( const Ptr<T> &p, ENull )
{
  return p.get() == 0;
}


/** \brief Returns true if <tt>p.get()!=NULL</tt>.
 *
 * \relates Ptr
 */
template<class T>
bool operator!=( const Ptr<T> &p, ENull )
{
  return p.get() != 0;
}


/** \brief Return true if two <tt>Ptr</tt> objects point to the same object.
 *
 * \relates Ptr
 */
template<class T1, class T2>
bool operator==( const Ptr<T1> &p1, const Ptr<T2> &p2 )
{
  return p1.get() == p2.get();
}


/** \brief Return true if two <tt>Ptr</tt> objects do not point to the same
 * object.
 *
 * \relates Ptr
 */
template<class T1, class T2>
bool operator!=( const Ptr<T1> &p1, const Ptr<T2> &p2 )
{
  return p1.get() != p2.get();
}


/** \brief Implicit cast of underlying <tt>Ptr</tt> type from <tt>T1*</tt> to
 * <tt>T2*</tt>.
 *
 * The function will compile only if (<tt>T2* p2 = p1.get();</tt>) compiles.
 *
 * This is to be used for conversions up an inheritance hierarchy and from
 * non-const to const and any other standard implicit pointer conversions
 * allowed by C++.
 *
 * \relates Ptr
 */
template<class T2, class T1>
Ptr<T2> ptr_implicit_cast(const Ptr<T1>& p1)
{
  return Ptr<T2>(p1.get()); // Will only compile if conversion is legal!
}


/** \brief Static cast of underlying <tt>Ptr</tt> type from <tt>T1*</tt> to
 * <tt>T2*</tt>.
 *
 * The function will compile only if (<tt>static_cast<T2*>(p1.get());</tt>)
 * compiles.
 *
 * This can safely be used for conversion down an inheritance hierarchy with
 * polymorphic types only if <tt>dynamic_cast<T2>(p1.get()) ==
 * static_cast<T2>(p1.get())</tt>.  If not then you have to use
 * <tt>ptr_dynamic_cast<tt><T2>(p1)</tt>.
 *
 * \relates Ptr
 */
template<class T2, class T1>
Ptr<T2> ptr_static_cast(const Ptr<T1>& p1)
{
  return Ptr<T2>(static_cast<T2*>(p1.get())); // Will only compile if conversion is legal!
}


/** \brief Constant cast of underlying <tt>Ptr</tt> type from <tt>T1*</tt> to
 * <tt>T2*</tt>.
 *
 * This function will compile only if (<tt>const_cast<T2*>(p1.get());</tt>)
 * compiles.
 *
 * \relates Ptr
 */
template<class T2, class T1>
Ptr<T2> ptr_const_cast(const Ptr<T1>& p1)
{
  return Ptr<T2>(const_cast<T2*>(p1.get())); // Will only compile if conversion is legal!
}


/** \brief Dynamic cast of underlying <tt>Ptr</tt> type from <tt>T1*</tt> to
 * <tt>T2*</tt>.
 *
 * \param p1 [in] The smart pointer casting from
 *
 * \param throw_on_fail [in] If <tt>true</tt> then if the cast fails (for
 * <tt>p1.get()!=NULL) then a <tt>std::bad_cast</tt> std::exception is thrown
 * with a very informative error message.
 *
 * <b>Postconditions:</b><ul>
 * <li> If <tt>( p1.get()!=NULL && throw_on_fail==true && dynamic_cast<T2*>(p1.get())==NULL ) == true</tt>
 *      then an <tt>std::bad_cast</tt> std::exception is thrown with a very informative error message.
 * <li> If <tt>( p1.get()!=NULL && dynamic_cast<T2*>(p1.get())!=NULL ) == true</tt>
 *      then <tt>return.get() == dynamic_cast<T2*>(p1.get())</tt>.
 * <li> If <tt>( p1.get()!=NULL && throw_on_fail==false && dynamic_cast<T2*>(p1.get())==NULL ) == true</tt>
 *      then <tt>return.get() == NULL</tt>.
 * <li> If <tt>( p1.get()==NULL ) == true</tt>
 *      then <tt>return.get() == NULL</tt>.
 * </ul>
 *
 * This function will compile only if (<tt>dynamic_cast<T2*>(p1.get());</tt>)
 * compiles.
 *
 * \relates Ptr
 */
template<class T2, class T1>
Ptr<T2> ptr_dynamic_cast(
  const Ptr<T1>& p1, bool throw_on_fail = false
  )
{
  if( p1.get() ) {
    T2 *check = NULL;
    if(throw_on_fail)
      check = &dyn_cast<T2>(*p1);
    else
      check = dynamic_cast<T2*>(p1.get());
    if(check) {
      return Ptr<T2>(check);
    }
  }
  return null;
}


/** \brief Output stream inserter.
 *
 * The implementation of this function just print pointer addresses and
 * therefore puts no restrictions on the data types involved.
 *
 * \relates Ptr
 */
template<class T>
std::ostream& operator<<( std::ostream& out, const Ptr<T>& p );


} // namespace Teuchos


#endif // TEUCHOS_PTR_DECL_HPP
