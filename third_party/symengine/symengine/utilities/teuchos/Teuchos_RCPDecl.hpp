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

#ifndef TEUCHOS_RCP_DECL_HPP
#define TEUCHOS_RCP_DECL_HPP


/*! \file Teuchos_RCPDecl.hpp
    \brief Reference-counted pointer class and non-member templated function implementations.
*/


#include "Teuchos_RCPNode.hpp"
#include "Teuchos_ENull.hpp"
#include "Teuchos_NullIteratorTraits.hpp"


#ifdef REFCOUNTPTR_INLINE_FUNCS
#  define REFCOUNTPTR_INLINE inline
#else
#  define REFCOUNTPTR_INLINE
#endif


#ifdef TEUCHOS_DEBUG
#  define TEUCHOS_REFCOUNTPTR_ASSERT_NONNULL
#endif


namespace Teuchos {


/** \brief . */
template<class T> class Ptr;


enum ERCPWeakNoDealloc { RCP_WEAK_NO_DEALLOC };
enum ERCPUndefinedWeakNoDealloc { RCP_UNDEFINED_WEAK_NO_DEALLOC };
enum ERCPUndefinedWithDealloc { RCP_UNDEFINED_WITH_DEALLOC };


/** \brief Smart reference counting pointer class for automatic garbage
  collection.
  
For a carefully written discussion about what this class is and basic details
on how to use it see the <A
HREF="../../../teuchos/doc/html/RefCountPtrBeginnersGuideSAND.pdf">beginners
guide</A>.

<b>Quickstart for <tt>RCP</tt></b>
 
Here we present a short, but fairly comprehensive, quick-start for the
use of <tt>RCP<></tt>.  The use cases described here
should cover the overwhelming majority of the use instances of
<tt>RCP<></tt> in a typical program.

The following class hierarchy will be used in the C++ examples given
below.

\code

class A { public: virtual ~A(){} virtual void f(){} };
class B1 : virtual public A {};
class B2 : virtual public A {};
class C : virtual public B1, virtual public B2 {};

class D {};
class E : public D {};

\endcode

All of the following code examples used in this quickstart are assumed to be
in the namespace <tt>Teuchos</tt> or have appropriate <tt>using
Teuchos::...</tt> declarations.  This removes the need to explicitly use
<tt>Teuchos::</tt> to qualify classes, functions and other declarations from
the <tt>Teuchos</tt> namespace.  Note that some of the runtime checks are
denoted as "debug runtime checked" which means that checking will only be
performed in a debug build (that is one where the macro <tt>TEUCHOS_DEBUG</tt>
is defined at compile time).

<ol>

<li> <b>Creation of <tt>RCP<></tt> objects</b>

<ol>

<li> <b>Creating an <tt>RCP<></tt> object using <tt>new</tt></b>

\code
RCP<C> c_ptr = rcp(new C);
\endcode

<li> <b>Creating a <tt>RCP<></tt> object equipped with a specialized
deallocator function</b> : <tt>Teuchos::DeallocFunctorDelete</tt>

\code
void someDeallocFunction(C* c_ptr);

RCP<C> c_ptr = rcp(new deallocFunctorDelete<C>(someDeallocFunction),true);
\endcode

<li> <b>Initializing a <tt>RCP<></tt> object to NULL</b>

\code
RCP<C> c_ptr;
\endcode

or

\code
RCP<C> c_ptr = null;
\endcode

<li> <b>Initializing a <tt>RCP<></tt> object to an object
       \underline{not} allocated with <tt>new</tt></b>

\code
C c;
RCP<C> c_ptr = rcpFromRef(c);
\endcode

<li> <b>Copy constructor (implicit casting)</b>

\code
RCP<C>       c_ptr  = rcp(new C); // No cast
RCP<A>       a_ptr  = c_ptr;      // Cast to base class
RCP<const A> ca_ptr = a_ptr;      // Cast from non-const to const
\endcode

<li> <b>Representing constantness and non-constantness</b>

<ol>

<li> <b>Non-constant pointer to non-constant object</b>
\code
RCP<C> c_ptr;
\endcode

<li> <b>Constant pointer to non-constant object</b>
\code
const RCP<C> c_ptr;
\endcode

<li> <b>Non-Constant pointer to constant object</b>
\code
RCP<const C> c_ptr;
\endcode

<li> <b>Constant pointer to constant object</b>
\code
const RCP<const C> c_ptr;
\endcode

</ol>

</ol>

<li> <b>Reinitialization of <tt>RCP<></tt> objects (using assignment
operator)</b>

<ol>

<li> <b>Resetting from a raw pointer</b>

\code
RCP<A> a_ptr;
a_ptr = rcp(new C());
\endcode

<li> <b>Resetting to null</b>

\code
RCP<A> a_ptr = rcp(new C());
a_ptr = null; // The C object will be deleted here
\endcode

<li> <b>Assigning from a <tt>RCP<></tt> object</b>

\code
RCP<A> a_ptr1;
RCP<A> a_ptr2 = rcp(new C());
a_ptr1 = a_ptr2; // Now a_ptr1 and a_ptr2 point to same C object
\endcode

</ol>

<li> <b>Accessing the reference-counted object</b>

<ol>

<li> <b>Access to object reference (debug runtime checked)</b> :
<tt>Teuchos::RCP::operator*()</tt>

\code
C &c_ref = *c_ptr;
\endcode

<li> <b>Access to object pointer (unchecked, may return <tt>NULL</tt>)</b> :
<tt>Teuchos::RCP::get()</tt>

\code
C *c_rptr = c_ptr.get();
\endcode

or

\code
C *c_rptr = c_ptr.getRawPtr();
\endcode

<b>WARNING:</b>: Avoid exposing raw C++ pointers!

<li> <b>Access to object pointer (debug runtime checked, will not return
<tt>NULL</tt>)</b> : <tt>Teuchos::RCP::operator*()</tt>

\code
C *c_rptr = &*c_ptr;
\endcode

<b>WARNING:</b>: Avoid exposing raw C++ pointers!

<li> <b>Access of object's member (debug runtime checked)</b> :
<tt>Teuchos::RCP::operator->()</tt>

\code
c_ptr->f();
\endcode

<li> <b>Testing for non-null</b> : <tt>Teuchos::RCP::get()</tt>,
<tt>Teuchos::operator==()</tt>, <tt>Teuchos::operator!=()</tt>

\code
if (a_ptr.is_null) std::cout << "a_ptr is not null!\n";
\endcode

or

\code
if (a_ptr != null) std::cout << "a_ptr is not null!\n";
\endcode

or

<li> <b>Testing for null</b>

\code
if (!a_ptr.is_null()) std::cout << "a_ptr is null!\n";
\endcode

or

\code
if (a_ptr == null) std::cout << "a_ptr is null!\n";
\endcode

or

\code
if (is_null(a_ptr)) std::cout << "a_ptr is null!\n";
\endcode

</ol>

<li> <b>Casting</b>

<ol>

<li> <b>Implicit casting (see copy constructor above)</b>

<ol>

<li> <b>Using copy constructor (see above)</b>

<li> <b>Using conversion function</b>

\code
RCP<C>       c_ptr  = rcp(new C);                       // No cast
RCP<A>       a_ptr  = rcp_implicit_cast<A>(c_ptr);      // To base
RCP<const A> ca_ptr = rcp_implicit_cast<const A>(a_ptr);// To const
\endcode

</ol>

<li> <b>Casting away <tt>const</tt></b> : <tt>rcp_const_cast()</tt>

\code
RCP<const A>  ca_ptr = rcp(new C);
RCP<A>        a_ptr  = rcp_const_cast<A>(ca_ptr); // cast away const!
\endcode

<li> <b>Static cast (no runtime check)</b> : <tt>rcp_static_cast()</tt>

\code
RCP<D>     d_ptr = rcp(new E);
RCP<E>     e_ptr = rcp_static_cast<E>(d_ptr); // Unchecked, unsafe?
\endcode

<li> <b>Dynamic cast (runtime checked, failed cast allowed)</b> : <tt>rcp_dynamic_cast()</tt>

\code
RCP<A>     a_ptr  = rcp(new C);
RCP<B1>    b1_ptr = rcp_dynamic_cast<B1>(a_ptr);  // Checked, safe!
RCP<B2>    b2_ptr = rcp_dynamic_cast<B2>(b1_ptr); // Checked, safe!
RCP<C>     c_ptr  = rcp_dynamic_cast<C>(b2_ptr);  // Checked, safe!
\endcode

<li> <b>Dynamic cast (runtime checked, failed cast not allowed)</b> : <tt>rcp_dynamic_cast()</tt>

\code
RCP<A>     a_ptr1  = rcp(new C);
RCP<A>     a_ptr2  = rcp(new A);
RCP<B1>    b1_ptr1 = rcp_dynamic_cast<B1>(a_ptr1, true);  // Success!
RCP<B1>    b1_ptr2 = rcp_dynamic_cast<B1>(a_ptr2, true);  // Throw std::bad_cast!
\endcode

</ol>


<li> <b>Customized deallocators</b>

<ol>

<li> <b>Creating a <tt>RCP<></tt> object with a custom deallocator</b> : <tt>rcp()</tt>

TODO: Update this example!

<li> <b>Access customized deallocator (runtime checked, throws on failure)</b> : <tt>Teuchos::get_dealloc()</tt>

\code
const MyCustomDealloc<C>
  &dealloc = get_dealloc<MyCustomDealloc<C> >(c_ptr);
\endcode

<li> <b>Access optional customized deallocator</b> : <tt>Teuchos::get_optional_dealloc()</tt>

\code
const Ptr<const MyCustomDealloc<C> > dealloc =
  get_optional_dealloc<MyCustomDealloc<C> >(c_ptr);
if (!is_null(dealloc))
  std::cout << "This deallocator exits!\n";
\endcode

</ol>

<li> <b>Managing extra data</b>

<ol>

<li> <b>Adding extra data (post destruction of extra data)</b> : <tt>Teuchos::set_extra_data()</tt>

\code
set_extra_data(rcp(new B1), "A:B1", inOutArg(a_ptr));
\endcode

<li> <b>Adding extra data (pre destruction of extra data)</b> : <tt>Teuchos::get_extra_data()</tt>

\code
set_extra_data(rcp(new B1),"A:B1", inOutArg(a_ptr), PRE_DESTORY);
\endcode

<li> <b>Retrieving extra data</b> : <tt>Teuchos::get_extra_data()</tt>

\code
get_extra_data<RCP<B1> >(a_ptr, "A:B1")->f();
\endcode

<li> <b>Resetting extra data</b> : <tt>Teuchos::get_extra_data()</tt>

\code
get_extra_data<RCP<B1> >(a_ptr, "A:B1") = rcp(new C);
\endcode

<li> <b>Retrieving optional extra data</b> : <tt>Teuchos::get_optional_extra_data()</tt>

\code
const Ptr<const RCP<B1> > b1 =
  get_optional_extra_data<RCP<B1> >(a_ptr, "A:B1");
if (!is_null(b1))
  (*b1)->f();
\endcode

</ol>

</ol>

\ingroup teuchos_mem_mng_grp

 */

template<class T>
class RCP {
public:

  /** \brief . */
  typedef T  element_type;

  /** \name Constructors/destructors/initializers. */
  //@{

  /** \brief Initialize <tt>RCP<T></tt> to NULL.
   *
   * <b>Postconditons:</b> <ul>
   * <li> <tt>this->get() == 0</tt>
   * <li> <tt>this->strength() == RCP_STRENGTH_INVALID</tt>
   * <li> <tt>this->is_vali_ptr() == true</tt>
   * <li> <tt>this->strong_count() == 0</tt>
   * <li> <tt>this->weak_count() == 0</tt>
   * <li> <tt>this->has_ownership() == false</tt>
   * </ul>
   *
   * This allows clients to write code like:
   \code
   RCP<int> p = null;
   \endcode
   or
   \code
   RCP<int> p;
   \endcode
   * and construct to <tt>NULL</tt>
   */
  inline RCP(ENull null_arg = null);

  /** \brief Construct from a raw pointer.
   *
   * Note that this constructor is declared explicit so there is no implicit
   * conversion from a raw pointer to an RCP allowed.  If
   * <tt>has_ownership==false</tt>, then no attempt to delete the object will
   * occur.
   *
   * <b>Postconditons:</b><ul>
   * <li> <tt>this->get() == p</tt>
   * <li> <tt>this->strength() == RCP_STRONG</tt>
   * <li> <tt>this->is_vali_ptr() == true</tt>
   * <li> <tt>this->strong_count() == 1</tt>
   * <li> <tt>this->weak_count() == 0</tt>
   * <li> <tt>this->has_ownership() == has_ownership</tt>
   * </ul>
   *
   * NOTE: It is recommended that this constructor never be called directly
   * but only through a type-specific non-member constructor function or at
   * least through the general non-member <tt>rcp()</tt> function.
   */
  inline explicit RCP( T* p, bool has_ownership = true );

  /** \brief Construct from a raw pointer and a custom deallocator.
   *
   * \param p [in] Pointer to the reference-counted object to be wrapped
   *
   * \param dealloc [in] Deallocator policy object that will be copied by
   * value and will perform the custom deallocation of the object pointed to
   * by <tt>p</tt> when the last <tt>RCP</tt> object goes away.  See the class
   * <tt>DeallocDelete</tt> for the specfication and behavior of this policy
   * interface.
   *
   * <b>Postconditons:</b><ul>
   * <li> <tt>this->get() == p</tt>
   * <li> <tt>this->strength() == RCP_STRONG</tt>
   * <li> <tt>this->is_vali_ptr() == true</tt>
   * <li> <tt>this->strong_count() == 1</tt>
   * <li> <tt>this->weak_count() == 0</tt>
   * <li> <tt>this->has_ownership() == has_ownership</tt>
   * <li> <tt>get_dealloc<Delalloc_T>(*this)</tt> returns a copy of the
   *   custom deallocator object <tt>dealloc>/tt>.
   * </ul>
   */
  template<class Dealloc_T>
  inline RCP(T* p, Dealloc_T dealloc, bool has_ownership);

  /** \brief Initialize from another <tt>RCP<T></tt> object.
   *
   * After construction, <tt>this</tt> and <tt>r_ptr</tt> will
   * reference the same object.
   *
   * This form of the copy constructor is required even though the
   * below more general templated version is sufficient since some
   * compilers will generate this function automatically which will
   * give an incorrect implementation.
   *
   * <b>Postconditons:</b><ul>
   * <li> <tt>this->get() == r_ptr.get()</tt>
   * <li> <tt>this->strong_count() == r_ptr.strong_count()</tt>
   * <li> <tt>this->has_ownership() == r_ptr.has_ownership()</tt>
   * <li> If <tt>r_ptr.get() != NULL</tt> then <tt>r_ptr.strong_count()</tt> is incremented by 1
   * </ul>
   */
  inline RCP(const RCP<T>& r_ptr);

  /** \brief Initialize from another <tt>RCP<T2></tt> object (implicit conversion only).
   *
   * This function allows the implicit conversion of smart pointer objects just
   * like with raw C++ pointers.  Note that this function will only compile
   * if the statement <tt>T1 *ptr = r_ptr.get()</tt> will compile.
   *
   * <b>Postconditons:</b> <ul>
   * <li> <tt>this->get() == r_ptr.get()</tt>
   * <li> <tt>this->strong_count() == r_ptr.strong_count()</tt>
   * <li> <tt>this->has_ownership() == r_ptr.has_ownership()</tt>
   * <li> If <tt>r_ptr.get() != NULL</tt> then <tt>r_ptr.strong_count()</tt> is incremented by 1
   * </ul>
   */
  template<class T2>
  inline RCP(const RCP<T2>& r_ptr);

  /** \brief Removes a reference to a dynamically allocated object and possibly deletes
   * the object if owned.
   *
   * Deletes the object if <tt>this->has_ownership() == true</tt> and
   * <tt>this->strong_count() == 1</tt>.  If <tt>this->strong_count() ==
   * 1</tt> but <tt>this->has_ownership() == false</tt> then the object is not
   * deleted.  If <tt>this->strong_count() > 1</tt> then the internal
   * reference count shared by all the other related <tt>RCP<...></tt> objects
   * for this shared object is deincremented by one.  If <tt>this->get() ==
   * NULL</tt> then nothing happens.
   */
  inline ~RCP();

  /** \brief Copy the pointer to the referenced object and increment the
   * reference count.
   *
   * If <tt>this->has_ownership() == true</tt> and <tt>this->strong_count() == 1</tt>
   * before this operation is called, then the object pointed to by
   * <tt>this->get()</tt> will be deleted (usually using <tt>delete</tt>)
   * prior to binding to the pointer (possibly <tt>NULL</tt>) pointed to in
   * <tt>r_ptr</tt>.  Assignment to self (i.e. <tt>this->get() ==
   * r_ptr.get()</tt>) is harmless and this function does nothing.
   *
   * <b>Postconditons:</b><ul>
   * <li> <tt>this->get() == r_ptr.get()</tt>
   * <li> <tt>this->strong_count() == r_ptr.strong_count()</tt>
   * <li> <tt>this->has_ownership() == r_ptr.has_ownership()</tt>
   * <li> If <tt>r_ptr.get() != NULL</tt> then <tt>r_ptr.strong_count()</tt> is incremented by 1
   * </ul>
   *
   * Provides the "strong guarantee" in a debug build!
   */
  inline RCP<T>& operator=(const RCP<T>& r_ptr);

  /** \brief Assign to null.
   *
   * If <tt>this->has_ownership() == true</tt> and <tt>this->strong_count() == 1</tt>
   * before this operation is called, then the object pointed to by
   * <tt>this->get()</tt> will be deleted (usually using <tt>delete</tt>)
   * prior to binding to the pointer (possibly <tt>NULL</tt>) pointed to in
   * <tt>r_ptr</tt>.
   *
   * <b>Postconditons:</b><ul>
   * <li> See <tt>RCP(ENull)</tt>
   * </ul>
   */
  inline RCP<T>& operator=(ENull);

  /** \brief Swap the contents with some other RCP object. */
  inline void swap(RCP<T> &r_ptr);

  //@}

  /** \name Object/Pointer Access Functions */
  //@{

  /** \brief Returns true if the underlying pointer is null. */
  inline bool is_null() const;

  /** \brief Pointer (<tt>-></tt>) access to members of underlying object.
   *
   * <b>Preconditions:</b><ul>
   * <li> <tt>this->get() != NULL</tt> (throws <tt>NullReferenceError</tt>)
   * </ul>
   */
  inline T* operator->() const;

  /** \brief Dereference the underlying object.
   *
   * <b>Preconditions:</b><ul>
   * <li> <tt>this->get() != NULL</tt> (throws <tt>NullReferenceError</tt>)
   * </ul>
   */
  inline T& operator*() const;

  /** \brief Get the raw C++ pointer to the underlying object.
   *
   * NOTE: Prefer to get the safer Ptr<T> object from <tt>this->ptr()</tt>!
   */
  inline T* get() const;

  /** \brief Get the raw C++ pointer to the underlying object.
   *
   * NOTE: Prefer to get the safer Ptr<T> object from <tt>this->ptr()</tt>!
   */
  inline T* getRawPtr() const;

  /** \brief Get a safer wrapper raw C++ pointer to the underlying object. */
  inline Ptr<T> ptr() const;

  /** \brief Shorthand for ptr(). */
  inline Ptr<T> operator()() const;

  /** \brief Return an RCP<const T> version of *this. */
  inline RCP<const T> getConst() const;

  //@}

  /** \name Reference counting */
  //@{

  /** \brief Strength of the pointer.
   *
   * Return values:<ul>
   * <li><tt>RCP_STRONG</tt>: Underlying reference-counted object will be deleted
   *     when <tt>*this</tt> is destroyed if <tt>strong_count()==1</tt>. 
   * <li><tt>RCP_WEAK</tt>: Underlying reference-counted object will not be deleted
   *     when <tt>*this</tt> is destroyed if <tt>strong_count() > 0</tt>. 
   * <li><tt>RCP_STRENGTH_INVALID</tt>: <tt>*this</tt> is not strong or weak but
   *     is null.
   * </ul>
   */
  inline ERCPStrength strength() const;

  /** \brief Return if the underlying object pointer is still valid or not.
   *
   * The underlying object will not be valid if the strong count has gone to
   * zero but the weak count thas not.
   *
   * NOTE: Null is a valid object pointer.  If you want to know if there is a
   * non-null object and it is valid then <tt>!is_null() &&
   * is_valid_ptr()</tt> will be <tt>true</tt>.
   */
  inline bool is_valid_ptr() const;

  /** \brief Return the number of active <tt>RCP<></tt> objects that have a
   * "strong" reference to the underlying reference-counted object.
   *
   * \return If <tt>this->get() == NULL</tt> then this function returns 0.
   */
  inline int strong_count() const;

  /** \brief Return the number of active <tt>RCP<></tt> objects that have a
   * "weak" reference to the underlying reference-counted object.
   *
   * \return If <tt>this->get() == NULL</tt> then this function returns 0.
   */
  inline int weak_count() const;

  /** \brief Total count (strong_count() + weak_count()). */
  inline int total_count() const;

  /** \brief Give <tt>this</tt> and other <tt>RCP<></tt> objects ownership 
   * of the referenced object <tt>this->get()</tt>.
   *
   * See ~RCP() above.  This function
   * does nothing if <tt>this->get() == NULL</tt>.
   *
   * <b>Postconditions:</b>
   * <ul>
   * <li> If <tt>this->get() == NULL</tt> then
   *   <ul>
   *   <li> <tt>this->has_ownership() == false</tt> (always!).
   *   </ul>
   * <li> else
   *   <ul>
   *   <li> <tt>this->has_ownership() == true</tt>
   *   </ul>
   * </ul>
   */
  inline void set_has_ownership();

  /** \brief Returns true if <tt>this</tt> has ownership of object pointed to
   * by <tt>this->get()</tt> in order to delete it.
   *
   * See ~RCP() above.
   *
   * \return If this->get() <tt>== NULL</tt> then this function always returns
   * <tt>false</tt>.  Otherwise the value returned from this function depends
   * on which function was called most recently, if any; set_has_ownership()
   * (<tt>true</tt>) or release() (<tt>false</tt>).
   */
  inline bool has_ownership() const;

  /** \brief Release the ownership of the underlying dynamically allocated
   * object.
   *
   * <b>WARNING!</b> Never call <tt>delete rcp.release().get()</tt> as this
   * can cause all kinds of segfaults.  Instead, release your use of the
   * shared object by simply assigning the <tt>RCP</tt> object to
   * <tt>Teuchos::null</tt>.
   *
   * This function should only be used as last result when all hell has broken
   * loose and memory management control has broken down.  This function is
   * not to be used lightly!
   *
   * After this function is called then the client is responsible for
   * deallocating the shared object no matter how many
   * <tt>ref_count_prt<T></tt> objects have a reference to it.  If
   * <tt>this-></tt>get()<tt>== NULL</tt>, then this call is meaningless.
   *
   * Note that this function does not have the exact same semantics as does
   * <tt>auto_ptr<T>::release()</tt>.  In <tt>auto_ptr<T>::release()</tt>,
   * <tt>this</tt> is set to <tt>NULL</tt> while here in RCP<T>::
   * release() only an ownership flag is set and <tt>*this</tt> still points
   * to the same object.  It would be difficult to duplicate the behavior of
   * <tt>auto_ptr<T>::release()</tt> for this class.
   *
   * <b>Postconditions:</b>
   * <ul>
   * <li> <tt>this->has_ownership() == false</tt>
   * </ul>
   *
   * @return Returns the value of <tt>this->get()</tt>
   */
  inline Ptr<T> release();

  /** \brief Create a new weak RCP object from another (strong) RCP object.
   *
   * ToDo: Explain this!
   *
   * <b>Preconditons:</b> <ul>
   * <li> <tt>returnVal.is_valid_ptr()==true</tt>
   * </ul>
   *
   * <b>Postconditons:</b> <ul>
   * <li> <tt>returnVal.get() == this->get()</tt>
   * <li> <tt>returnVal.strong_count() == this->strong_count()</tt>
   * <li> <tt>returnVal.weak_count() == this->weak_count()+1</tt>
   * <li> <tt>returnVal.strength() == RCP_WEAK</tt>
   * <li> <tt>returnVal.has_ownership() == this->has_ownership()</tt>
   * </ul>
   */
  inline RCP<T> create_weak() const;

  /** \brief Create a new strong RCP object from another (weak) RCP object.
   *
   * ToDo: Explain this!
   *
   * <b>Preconditons:</b> <ul>
   * <li> <tt>returnVal.is_valid_ptr()==true</tt>
   * </ul>
   *
   * <b>Postconditons:</b> <ul>
   * <li> <tt>returnVal.get() == this->get()</tt>
   * <li> <tt>returnVal.strong_count() == this->strong_count() + 1</tt>
   * <li> <tt>returnVal.weak_count() == this->weak_count()</tt>
   * <li> <tt>returnVal.strength() == RCP_STRONG</tt>
   * <li> <tt>returnVal.has_ownership() == this->has_ownership()</tt>
   * </ul>
   */
  inline RCP<T> create_strong() const;

  /** \brief Returns true if the smart pointers share the same underlying
   * reference-counted object.
   *
   * This method does more than just check if <tt>this->get() == r_ptr.get()</tt>.
   * It also checks to see if the underlying reference counting machinary is the
   * same.
   */
  template<class T2>
  inline bool shares_resource(const RCP<T2>& r_ptr) const;

  //@}

  /** \name Assertions */
  //@{

  /** \brief Throws <tt>NullReferenceError</tt> if <tt>this->get()==NULL</tt>,
   * otherwise returns reference to <tt>*this</tt>.
   */
  inline const RCP<T>& assert_not_null() const;

  /** \brief If the object pointer is non-null, assert that it is still valid.
   *
   * If <tt>is_null()==false && strong_count()==0</tt>, this will throw
   * <tt>DanglingReferenceErorr</tt> with a great error message.
   *
   * If <tt>is_null()==true</tt>, then this will not throw any exception.
   *
   * In this context, null is a valid object.
   */
  inline const RCP<T>& assert_valid_ptr() const;

  /** \brief Calls <tt>assert_not_null()</tt> in a debug build. */
  inline const RCP<T>& debug_assert_not_null() const
    {
#ifdef TEUCHOS_REFCOUNTPTR_ASSERT_NONNULL
      assert_not_null();
#endif
      return *this;
    }

  /** \brief Calls <tt>assert_valid_ptr()</tt> in a debug build. */
  inline const RCP<T>& debug_assert_valid_ptr() const
    {
#ifdef TEUCHOS_DEBUG
      assert_valid_ptr();
#endif
      return *this;
    }

  //@}

  /** \name boost::shared_ptr compatiblity funtions. */
  //@{

  /** \brief Reset to null. */
  inline void reset();

  /** \brief Reset the raw pointer with default ownership to delete.
   *
   * Equivalent to calling:
   
   \code

     r_rcp = rcp(p)

   \endcode
   */
  template<class T2>
  inline void reset(T2* p, bool has_ownership = true);

  /** \brief Returns <tt>strong_count()</tt> [deprecated]. */
  inline int count() const;

  //@}

private:

  // //////////////////////////////////////////////////////////////
  // Private data members

  T *ptr_; // NULL if this pointer is null
  RCPNodeHandle node_; // NULL if this pointer is null

public: // Bad bad bad

  // These constructors are put here because we don't want to confuse users
  // who would otherwise see them.

  /** \brief Construct a non-owning RCP from a raw pointer to a type that *is*
   * defined.
   *
   * This version avoids adding a deallocator but still requires the type to
   * be defined since it looks up the base object's address when doing RCPNode
   * tracing.
   *
   * NOTE: It is recommended that this constructor never be called directly
   * but only through a type-specific non-member constructor function or at
   * least through the general non-member <tt>rcpFromRef()</tt> function.
   */
  inline explicit RCP(T* p, ERCPWeakNoDealloc);

  /** \brief Construct a non-owning RCP from a raw pointer to a type that is
   * *not* defined.
   *
   * This version avoids any type of compile-time queries of the type that
   * would fail due to the type being undefined.
   *
   * NOTE: It is recommended that this constructor never be called directly
   * but only through a type-specific non-member constructor function or at
   * least through the general non-member <tt>rcpFromUndefRef()</tt> function.
   */
  inline explicit RCP(T* p, ERCPUndefinedWeakNoDealloc);

  /** \brief Construct from a raw pointer and a custom deallocator for an
   * undefined type.
   *
   * This version avoids any type of compile-time queries of the type that
   * would fail due to the type being undefined.
   */
  template<class Dealloc_T>
  inline RCP(T* p, Dealloc_T dealloc, ERCPUndefinedWithDealloc,
    bool has_ownership = true);

#ifndef DOXYGEN_COMPILE

  // WARNING: A general user should *never* call these functions!
  inline RCP(T* p, const RCPNodeHandle &node);
  inline T* access_private_ptr() const; // Does not throw
  inline RCPNodeHandle& nonconst_access_private_node(); // Does not thorw
  inline const RCPNodeHandle& access_private_node() const; // Does not thorw

#endif

};

/** \brief Struct for comparing two RCPs. Simply compares
* the raw pointers contained within the RCPs*/
struct RCPComp {
  /** \brief . */
  template<class T1, class T2> inline
  bool operator() (const RCP<T1> p1, const RCP<T2> p2) const{
    return p1.get() < p2.get();
  }
};

/** \brief Struct for comparing two RCPs. Simply compares
* the raw pointers contained within the RCPs*/
struct RCPConstComp {
  /** \brief . */
  template<class T1, class T2> inline
  bool operator() (const RCP<const T1> p1, const RCP<const T2> p2) const{
    return p1.get() < p2.get();
  }
};



// 2008/09/22: rabartl: NOTE: I removed the TypeNameTraits<RCP<T> >
// specialization since I want to be able to print the type name of an RCP
// that does not have the type T fully defined!


/** \brief Traits specialization for RCP.
 *
 * \ingroup teuchos_mem_mng_grp
 */
template<typename T>
class NullIteratorTraits<RCP<T> > {
public:
  static RCP<T> getNull() { return null; }
};


/** \brief Policy class for deallocator for non-owned RCPs.
 *
 * \ingroup teuchos_mem_mng_grp
 */
template<class T>
class DeallocNull
{
public:
  /// Gives the type (required)
  typedef T ptr_t;
  /// Deallocates a pointer <tt>ptr</tt> using <tt>delete ptr</tt> (required).
  void free( T* ptr ) { }
};


/** \brief Policy class for deallocator that uses <tt>delete</tt> to delete a
 * pointer which is used by <tt>RCP</tt>.
 *
 * \ingroup teuchos_mem_mng_grp
 */
template<class T>
class DeallocDelete
{
public:
  /// Gives the type (required)
  typedef T ptr_t;
  /// Deallocates a pointer <tt>ptr</tt> using <tt>delete ptr</tt> (required).
  void free( T* ptr ) { if(ptr) delete ptr; }
};


/** \brief Deallocator class that uses <tt>delete []</tt> to delete memory
 * allocated uisng <tt>new []</tt>
 *
 * \ingroup teuchos_mem_mng_grp
 */
template<class T>
class DeallocArrayDelete
{
public:
  /// Gives the type (required)
  typedef T ptr_t;
  /// Deallocates a pointer <tt>ptr</tt> using <tt>delete [] ptr</tt> (required).
  void free( T* ptr ) { if(ptr) delete [] ptr; }
};


/** \brief Deallocator subclass that Allows any functor object (including a
 * function pointer) to be used to free an object.
 *
 * Note, the only requirement is that deleteFuctor(ptr) can be called (which
 * is true for a function pointer).
 *
 * Note, a client should generally use the function
 * <tt>deallocFunctorDelete()</tt> to create this object and not try to
 * construct it directly.
 *
 * \ingroup teuchos_mem_mng_grp
 */
template<class T, class DeleteFunctor>
class DeallocFunctorDelete
{
public:
  DeallocFunctorDelete( DeleteFunctor deleteFunctor ) : deleteFunctor_(deleteFunctor) {}
  typedef T ptr_t;
  void free( T* ptr ) { if(ptr) deleteFunctor_(ptr); }
private:
  DeleteFunctor deleteFunctor_;
  DeallocFunctorDelete(); // Not defined and not to be called!
};


/** \brief A simple function used to create a functor deallocator object.
 *
 * \relates DeallocFunctorDelete
 */
template<class T, class DeleteFunctor>
DeallocFunctorDelete<T,DeleteFunctor>
deallocFunctorDelete( DeleteFunctor deleteFunctor )
{
  return DeallocFunctorDelete<T,DeleteFunctor>(deleteFunctor);
}


/** \brief Deallocator subclass that Allows any functor object (including a
 * function pointer) to be used to free a handle (i.e. pointer to pointer) to
 * an object.
 *
 * Note, the only requirement is that deleteFuctor(ptrptr) can be called
 * (which is true for a function pointer).
 *
 * Note, a client should generally use the function
 * <tt>deallocFunctorDelete()</tt> to create this object and not try to
 * construct it directly.
 *
 * \ingroup teuchos_mem_mng_grp
 */
template<class T, class DeleteHandleFunctor>
class DeallocFunctorHandleDelete
{
public:
  DeallocFunctorHandleDelete( DeleteHandleFunctor deleteHandleFunctor )
    : deleteHandleFunctor_(deleteHandleFunctor) {}
  typedef T ptr_t;
  void free( T* ptr ) { if(ptr) { T **hdl = &ptr; deleteHandleFunctor_(hdl); } }
private:
  DeleteHandleFunctor deleteHandleFunctor_;
  DeallocFunctorHandleDelete(); // Not defined and not to be called!
};


/** \brief A simple function used to create a functor deallocator object.
 *
 * \relates DeallocFunctorHandleDelete
 */
template<class T, class DeleteHandleFunctor>
DeallocFunctorHandleDelete<T,DeleteHandleFunctor>
deallocFunctorHandleDelete( DeleteHandleFunctor deleteHandleFunctor )
{
  return DeallocFunctorHandleDelete<T,DeleteHandleFunctor>(deleteHandleFunctor);
}


/** \brief A deallocator class that wraps a simple value object and delegates
 * to another deallocator object.
 *
 * The type <tt>Embedded</tt> must be a true value object with a default
 * constructor, a copy constructor, and an assignment operator.
 *
 * \ingroup teuchos_mem_mng_grp
 */
template<class T, class Embedded, class Dealloc>
class EmbeddedObjDealloc
{
public:
  typedef typename Dealloc::ptr_t ptr_t;
  EmbeddedObjDealloc(
    const Embedded &embedded, EPrePostDestruction prePostDestroy,
    Dealloc dealloc
    ) : embedded_(embedded), prePostDestroy_(prePostDestroy), dealloc_(dealloc)
    {}
  void setObj( const Embedded &embedded ) { embedded_ = embedded; }
  const Embedded& getObj() const { return embedded_; }
  Embedded& getNonconstObj() { return embedded_; }
  void free( T* ptr )
    {
      if (prePostDestroy_ == PRE_DESTROY)
        embedded_ = Embedded();
      dealloc_.free(ptr);
      if (prePostDestroy_ == POST_DESTROY)
        embedded_ = Embedded();
    }
private:
  Embedded embedded_;
  EPrePostDestruction prePostDestroy_;
  Dealloc dealloc_;
  EmbeddedObjDealloc(); // Not defined and not to be called!
};


/** \brief Create a dealocator with an embedded object using delete.
 *
 * \relates EmbeddedObjDealloc
 */
template<class T, class Embedded >
EmbeddedObjDealloc<T,Embedded,DeallocDelete<T> >
embeddedObjDeallocDelete(const Embedded &embedded, EPrePostDestruction prePostDestroy)
{
  return EmbeddedObjDealloc<T,Embedded,DeallocDelete<T> >(
    embedded, prePostDestroy,DeallocDelete<T>());
}


/** \brief Create a dealocator with an embedded object using delete [].
 *
 * \relates EmbeddedObjDealloc
 */
template<class T, class Embedded >
EmbeddedObjDealloc<T,Embedded,DeallocArrayDelete<T> >
embeddedObjDeallocArrayDelete(const Embedded &embedded, EPrePostDestruction prePostDestroy)
{
  return EmbeddedObjDealloc<T,Embedded,DeallocArrayDelete<T> >(
    embedded, prePostDestroy,DeallocArrayDelete<T>());
}


/** \brief Create a <tt>RCP</tt> object properly typed.
 *
 * \param p [in] Pointer to an object to be reference counted.
 *
 * \param owns_mem [in] If <tt>owns_mem==true</tt> then <tt>delete p</tt> will
 * be called when the last reference to this object is removed.  If
 * <tt>owns_mem==false</tt> then nothing will happen to delete the the object
 * pointed to by <tt>p</tt> when the last reference is removed.
 *
 * <b>Preconditions:</b><ul>
 * <li> If <tt>owns_mem==true</tt> then <tt>p</tt> must have been
 *      created by calling <tt>new</tt> to create the object since
 *      <tt>delete p</tt> will be called eventually.
 * </ul>
 *
 * If the pointer <tt>p</tt> did not come from <tt>new</tt> then
 * either the client should use the version of <tt>rcp()</tt> that
 * that uses a deallocator policy object or should pass in 
 * <tt>owns_mem = false</tt>.
 *
 * \relates RCP
 */
template<class T> inline
RCP<T> rcp(T* p, bool owns_mem = true);


/** \brief Initialize from a raw pointer with a deallocation policy.
 *
 * \param p [in] Raw C++ pointer that \c this will represent.
 *
 * \param dealloc [in] Deallocator policy object (copied by value) that
 * defines a function <tt>void Dealloc_T::free(T* p)</tt> that will free the
 * underlying object.
 *
 * \param owns_mem [in] If true then <tt>return</tt> is allowed to delete the
 * underlying pointer by calling <tt>dealloc.free(p)</tt>.  when all
 * references have been removed.
 *
 * <b>Preconditions:</b><ul>
 * <li> The function <tt>void Dealloc_T::free(T* p)</tt> exists.
 * </ul>
 *
 * <b>Postconditions:</b><ul>
 * <li> <tt>return.get() == p</tt>
 * <li> If <tt>p == NULL</tt> then
 *   <ul>
 *   <li> <tt>return.count() == 0</tt>
 *   <li> <tt>return.has_ownership() == false</tt>
 *   </ul>
 * <li> else
 *   <ul>
 *   <li> <tt>return.count() == 1</tt>
 *   <li> <tt>return.has_ownership() == owns_mem</tt>
 *   </ul>
 * </ul>
 *
 * By default, <tt>return</tt> has ownership to delete the object
 * pointed to by <tt>p</tt> when <tt>return</tt> is deleted (see
 * <tt>~RCP())</tt>.  If <tt>owns_mem==true</tt>, it is vital
 * that the address <tt>p</tt>
 * passed in is the same address that was returned by <tt>new</tt>.
 * With multiple inheritance this is not always the case.  See the
 * above discussion.  This class is templated to accept a deallocator
 * object that will free the pointer.  The other functions use a
 * default deallocator of type <tt>DeallocDelete</tt> which has a method
 * <tt>DeallocDelete::free()</tt> which just calls <tt>delete p</tt>.
 *
 * \relates RCP
 */
template<class T, class Dealloc_T> inline
RCP<T> rcpWithDealloc(T* p, Dealloc_T dealloc, bool owns_mem=true);


/** \brief Deprecated. */
template<class T, class Dealloc_T> inline
RCP<T> rcp( T* p, Dealloc_T dealloc, bool owns_mem )
{
  return rcpWithDealloc(p, dealloc, owns_mem);
}


/** \brief Initialize from a raw pointer with a deallocation policy for an
 * undefined type.
 *
 * \param p [in] Raw C++ pointer that \c this will represent.
 *
 * \param dealloc [in] Deallocator policy object (copied by value) that
 * defines a function <tt>void Dealloc_T::free(T* p)</tt> that will free the
 * underlying object.
 *
 * \relates RCP
 */
template<class T, class Dealloc_T> inline
RCP<T> rcpWithDeallocUndef(T* p, Dealloc_T dealloc, bool owns_mem=true);


/** \brief Return a non-owning weak RCP object from a raw object reference for
 * a defined type.
 *
 * NOTE: When debug mode is turned on, in general, the type must be defined.
 * If the type is undefined, then the function <tt>rcpFromUndefRef()</tt>
 * should be called instead.
 *
 * \relates RCP
 */
template<class T> inline
RCP<T> rcpFromRef(T& r);


/** \brief Return a non-owning weak RCP object from a raw object reference for
 * an undefined type.
 *
 * NOTE: This version will not be able to use RCPNode tracing to create a weak
 * reference to an existing RCPNode.  Therefore, you should only use this
 * version with an undefined type.
 *
 * \relates RCP
 */
template<class T> inline
RCP<T> rcpFromUndefRef(T& r);


/* \brief Create an RCP with and also put in an embedded object.
 *
 * In this case the embedded object is destroyed (by setting to Embedded())
 * before the object at <tt>*p</tt> is destroyed.
 *
 * The embedded object can be extracted using <tt>getEmbeddedObj()</tt> and
 * <tt>getNonconstEmbeddedObject()</tt>.
 *
 * \relates RCP
 */
template<class T, class Embedded> inline
RCP<T>
rcpWithEmbeddedObjPreDestroy( T* p, const Embedded &embedded, bool owns_mem = true );


/* \brief Create an RCP with and also put in an embedded object.
 *
 * In this case the embedded object is destroyed (by setting to Embedded())
 * after the object at <tt>*p</tt> is destroyed.
 *
 * The embedded object can be extracted using <tt>getEmbeddedObj()</tt> and
 * <tt>getNonconstEmbeddedObject()</tt>.
 *
 * \relates RCP
 */
template<class T, class Embedded> inline
RCP<T>
rcpWithEmbeddedObjPostDestroy( T* p, const Embedded &embedded, bool owns_mem = true );


/* \brief Create an RCP with and also put in an embedded object.
 *
 * This function should be called when it is not important when the embedded
 * object is destroyed (by setting to Embedded()) with respect to when
 * <tt>*p</tt> is destroyed.
 *
 * The embedded object can be extracted using <tt>getEmbeddedObj()</tt> and
 * <tt>getNonconstEmbeddedObject()</tt>.
 *
 * \relates RCP
 */
template<class T, class Embedded> inline
RCP<T>
rcpWithEmbeddedObj( T* p, const Embedded &embedded, bool owns_mem = true );


// 2007/10/25: rabartl: ToDo: put in versions of
// rcpWithEmbedded[Pre,Post]DestoryWithDealloc(...) that also accept a general
// deallocator!


/** \brief Create a new RCP that inverts the ownership of parent and child.
 *
 * This implements the "inverted object ownership" idiom.
 *
 * NOTE: The parent can be retrieved using the function
 * <tt>getInvertedObjOwnershipParent(...)</tt>.
 *
 * \relates RCP
 */
template<class T, class ParentT>
RCP<T> rcpWithInvertedObjOwnership(const RCP<T> &child, const RCP<ParentT> &parent);


/** \brief Allocate a new RCP object with a new RCPNode with memory pointing
 * to the initial node.
 *
 * The purpose of this function is to create a new "handle" to the underlying
 * memory with its own seprate reference count.  The new RCP object will have
 * a new RCPNodeTmpl object that has a copy of the input RCP object embedded
 * in it.  This maintains the correct reference counting behaviors but now
 * gives a private count.  One would want to use rcpCloneNode(...) whenever it
 * is important to keep a private reference count which is needed for some
 * types of use cases.
 *
 * \relates RCP
 */
template<class T>
RCP<T> rcpCloneNode(const RCP<T> &p);


/** \brief Returns true if <tt>p.get()==NULL</tt>.
 *
 * \relates RCP
 */
template<class T> inline
bool is_null( const RCP<T> &p );


/** \brief Returns true if <tt>p.get()!=NULL</tt>.
 *
 * \relates RCP
 */
template<class T> inline
bool nonnull( const RCP<T> &p );


/** \brief Returns true if <tt>p.get()==NULL</tt>.
 *
 * \relates RCP
 */
template<class T> inline
bool operator==( const RCP<T> &p, ENull );


/** \brief Returns true if <tt>p.get()!=NULL</tt>.
 *
 * \relates RCP
 */
template<class T> inline
bool operator!=( const RCP<T> &p, ENull );


/** \brief Return true if two <tt>RCP</tt> objects point to the same
 * referenced-counted object and have the same node.
 *
 * \relates RCP
 */
template<class T1, class T2> inline
bool operator==( const RCP<T1> &p1, const RCP<T2> &p2 );


/** \brief Return true if two <tt>RCP</tt> objects do not point to the
 * same referenced-counted object and have the same node.
 *
 * \relates RCP
 */
template<class T1, class T2> inline
bool operator!=( const RCP<T1> &p1, const RCP<T2> &p2 );


/** \brief Implicit cast of underlying <tt>RCP</tt> type from <tt>T1*</tt> to <tt>T2*</tt>.
 *
 * The function will compile only if (<tt>T2* p2 = p1.get();</tt>) compiles.
 *
 * This is to be used for conversions up an inheritance hierarchy and from non-const to
 * const and any other standard implicit pointer conversions allowed by C++.
 *
 * \relates RCP
 */
template<class T2, class T1> inline
RCP<T2> rcp_implicit_cast(const RCP<T1>& p1);


/** \brief Static cast of underlying <tt>RCP</tt> type from <tt>T1*</tt> to <tt>T2*</tt>.
 *
 * The function will compile only if (<tt>static_cast<T2*>(p1.get());</tt>) compiles.
 *
 * This can safely be used for conversion down an inheritance hierarchy
 * with polymorphic types only if <tt>dynamic_cast<T2>(p1.get()) == static_cast<T2>(p1.get())</tt>.
 * If not then you have to use <tt>rcp_dynamic_cast<tt><T2>(p1)</tt>.
 *
 * \relates RCP
 */
template<class T2, class T1> inline
RCP<T2> rcp_static_cast(const RCP<T1>& p1);


/** \brief Constant cast of underlying <tt>RCP</tt> type from <tt>T1*</tt> to <tt>T2*</tt>.
 *
 * This function will compile only if (<tt>const_cast<T2*>(p1.get());</tt>) compiles.
 *
 * \relates RCP
 */
template<class T2, class T1> inline
RCP<T2> rcp_const_cast(const RCP<T1>& p1);


/** \brief Dynamic cast of underlying <tt>RCP</tt> type from <tt>T1*</tt> to <tt>T2*</tt>.
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
 * This function will compile only if (<tt>dynamic_cast<T2*>(p1.get());</tt>) compiles.
 *
 * \relates RCP
 */
template<class T2, class T1> inline
RCP<T2> rcp_dynamic_cast(
  const RCP<T1>& p1, bool throw_on_fail = false
  );


/** \brief Set extra data associated with a <tt>RCP</tt> object.
 *
 * \param extra_data [in] Data object that will be set (copied)
 *
 * \param name [in] The name given to the extra data.  The value of
 * <tt>name</tt> together with the data type <tt>T1</tt> of the extra data
 * must be unique from any other such data or the other data will be
 * overwritten.
 *
 * \param p [out] On output, will be updated with the input
 * <tt>extra_data</tt>
 *
 * \param destroy_when [in] Determines when <tt>extra_data</tt> will be
 * destroyed in relation to the underlying reference-counted object.  If
 * <tt>destroy_when==PRE_DESTROY</tt> then <tt>extra_data</tt> will be deleted
 * before the underlying reference-counted object.  If
 * <tt>destroy_when==POST_DESTROY</tt> (the default) then <tt>extra_data</tt>
 * will be deleted after the underlying reference-counted object.
 *
 * \param force_unique [in] Determines if this type and name pair must be
 * unique in which case if an object with this same type and name already
 * exists, then an std::exception will be thrown.  The default is
 * <tt>true</tt> for safety.
 *
 * If there is a call to this function with the same type of extra
 * data <tt>T1</tt> and same arguments <tt>p</tt> and <tt>name</tt>
 * has already been made, then the current piece of extra data already
 * set will be overwritten with <tt>extra_data</tt>.  However, if the
 * type of the extra data <tt>T1</tt> is different, then the extra
 * data can be added and not overwrite existing extra data.  This
 * means that extra data is keyed on both the type and name.  This
 * helps to minimize the chance that clients will unexpectedly
 * overwrite data by accident.
 *
 * When the last <tt>RefcountPtr</tt> object is removed and the
 * reference-count node is deleted, then objects are deleted in the following
 * order: (1) All of the extra data that where added with
 * <tt>destroy_when==PRE_DESTROY</tt> are first, (2) then the underlying
 * reference-counted object is deleted, and (3) the rest of the extra data
 * that was added with <tt>destroy_when==PRE_DESTROY</tt> is then deleted.
 * The order in which the objects are destroyed is not guaranteed.  Therefore,
 * clients should be careful not to add extra data that has deletion
 * dependancies (instead consider using nested RCP objects as extra
 * data which will guarantee the order of deletion).
 *
 * <b>Preconditions:</b><ul>
 * <li> <tt>p->get() != NULL</tt> (throws <tt>NullReferenceError</tt>)
 * <li> If this function has already been called with the same template
 *      type <tt>T1</tt> for <tt>extra_data</tt> and the same std::string <tt>name</tt>
 *      and <tt>force_unique==true</tt>, then an <tt>std::invalid_argument</tt>
 *      std::exception will be thrown.
 * </ul>
 *
 * Note, this function is made a non-member function to be consistent
 * with the non-member <tt>get_extra_data()</tt> functions.
 *
 * \relates RCP
 */
template<class T1, class T2>
void set_extra_data( const T1 &extra_data, const std::string& name,
  const Ptr<RCP<T2> > &p, EPrePostDestruction destroy_when = POST_DESTROY,
  bool force_unique = true);

/** \brief Get a const reference to extra data associated with a <tt>RCP</tt> object.
 *
 * \param p [in] Smart pointer object that extra data is being extraced from.
 *
 * \param name [in] Name of the extra data.
 *
 * @return Returns a const reference to the extra_data object.
 *
 * <b>Preconditions:</b><ul>
 * <li> <tt>p.get() != NULL</tt> (throws <tt>NullReferenceError</tt>)
 * <li> <tt>name</tt> and <tt>T1</tt> must have been used in a previous
 *      call to <tt>set_extra_data()</tt> (throws <tt>std::invalid_argument</tt>).
 * </ul>
 *
 * Note, this function must be a non-member function since the client
 * must manually select the first template argument.
 *
 * \relates RCP
 */
template<class T1, class T2>
const T1& get_extra_data( const RCP<T2>& p, const std::string& name );


/** \brief Get a non-const reference to extra data associated with a <tt>RCP</tt> object.
 *
 * \param p [in] Smart pointer object that extra data is being extraced from.
 *
 * \param name [in] Name of the extra data.
 *
 * @return Returns a non-const reference to the extra_data object.
 *
 * <b>Preconditions:</b><ul>
 * <li> <tt>p.get() != NULL</tt> (throws <tt>NullReferenceError</tt>)
 * <li> <tt>name</tt> and <tt>T1</tt> must have been used in a previous
 *      call to <tt>set_extra_data()</tt> (throws <tt>std::invalid_argument</tt>).
 * </ul>
 *
 * Note, this function must be a non-member function since the client
 * must manually select the first template argument.
 *
 * \relates RCP
 */
template<class T1, class T2>
T1& get_nonconst_extra_data( RCP<T2>& p, const std::string& name );


/** \brief Get a pointer to const extra data (if it exists) associated with a
 * <tt>RCP</tt> object.
 *
 * \param p [in] Smart pointer object that extra data is being extraced from.
 *
 * \param name [in] Name of the extra data.
 *
 * @return Returns a const pointer to the extra_data object if it exists.
 *
 * <b>Preconditions:</b><ul>
 * <li> <tt>p.get() != NULL</tt> (throws <tt>NullReferenceError</tt>)
 * </ul>
 *
 * <b>Postconditions:</b><ul>
 * <li> If <tt>name</tt> and <tt>T1</tt> have been used in a previous
 *      call to <tt>set_extra_data()</tt> then <tt>return !=NULL</tt>
 *      and otherwise <tt>return == NULL</tt>.
 * </ul>
 *
 * Note, this function must be a non-member function since the client
 * must manually select the first template argument.
 *
 * \relates RCP
 */
template<class T1, class T2>
Ptr<const T1> get_optional_extra_data( const RCP<T2>& p, const std::string& name );


/** \brief Get a pointer to non-const extra data (if it exists) associated
 * with a <tt>RCP</tt> object.
 *
 * \param p [in] Smart pointer object that extra data is being extraced from.
 *
 * \param name [in] Name of the extra data.
 *
 * @return Returns a non-const pointer to the extra_data object.
 *
 * <b>Preconditions:</b><ul>
 * <li> <tt>p.get() != NULL</tt> (throws <tt>NullReferenceError</tt>)
 * </ul>
 *
 * <b>Postconditions:</b><ul>
 * <li> If <tt>name</tt> and <tt>T1</tt> have been used in a previous
 *      call to <tt>set_extra_data()</tt> then <tt>return !=NULL</tt>
 *      and otherwise <tt>return == NULL</tt>.
 * </ul>
 *
 * Note, this function must be a non-member function since the client
 * must manually select the first template argument.
 *
 * \relates RCP
 */
template<class T1, class T2>
Ptr<T1> get_optional_nonconst_extra_data( RCP<T2>& p, const std::string& name );


/** \brief Return a <tt>const</tt> reference to the underlying deallocator
 * object.
 *
 * <b>Preconditions:</b><ul>
 * <li> <tt>p.get() != NULL</tt> (throws <tt>NullReferenceError</tt>)
 * <li> The deallocator object type used to construct <tt>p</tt> is same as <tt>Dealloc_T</tt>
 *      (throws <tt>NullReferenceError</tt>)
 * </ul>
 *
 * \relates RCP
 */
template<class Dealloc_T, class T>
const Dealloc_T& get_dealloc( const RCP<T>& p );


/** \brief Return a non-<tt>const</tt> reference to the underlying deallocator
 * object.
 *
 * <b>Preconditions:</b><ul>
 * <li> <tt>p.get() != NULL</tt> (throws <tt>NullReferenceError</tt>)
 * <li> The deallocator object type used to construct <tt>p</tt> is same as <tt>Dealloc_T</tt>
 *      (throws <tt>NullReferenceError</tt>)
 * </ul>
 *
 * \relates RCP
 */
template<class Dealloc_T, class T>
Dealloc_T& get_nonconst_dealloc( const RCP<T>& p );


/** \brief Return a pointer to the underlying <tt>const</tt> deallocator
 * object if it exists.
 *
 * <b>Preconditions:</b><ul>
 * <li> <tt>p.get() != NULL</tt> (throws <tt>NullReferenceError</tt>)
 * </ul>
 *
 * <b>Postconditions:</b><ul>
 * <li> If the deallocator object type used to construct <tt>p</tt> is same as <tt>Dealloc_T</tt>
 *      then <tt>return!=NULL</tt>, otherwise <tt>return==NULL</tt>
 * </ul>
 *
 * \relates RCP
 */
template<class Dealloc_T, class T>
Ptr<const Dealloc_T> get_optional_dealloc( const RCP<T>& p );


/** \brief Return a pointer to the underlying non-<tt>const</tt> deallocator
 * object if it exists.
 *
 * <b>Preconditions:</b><ul>
 * <li> <tt>p.get() != NULL</tt> (throws <tt>NullReferenceError</tt>)
 * </ul>
 *
 * <b>Postconditions:</b><ul>
 * <li> If the deallocator object type used to construct <tt>p</tt> is same as <tt>Dealloc_T</tt>
 *      then <tt>return!=NULL</tt>, otherwise <tt>return==NULL</tt>
 * </ul>
 *
 * \relates RCP
 */
template<class Dealloc_T, class T>
Ptr<Dealloc_T> get_optional_nonconst_dealloc( const RCP<T>& p );


/** \brief Get a const reference to an embedded object that was set by calling
 * <tt>rcpWithEmbeddedObjPreDestroy()</tt>,
 * <tt>rcpWithEmbeddedObjPostDestory()</tt>, or <tt>rcpWithEmbeddedObj()</tt>.
 *
 * \relates RCP
 */
template<class TOrig, class Embedded, class T>
const Embedded& getEmbeddedObj( const RCP<T>& p );


/** \brief Get a non-const reference to an embedded object that was set by
 * calling <tt>rcpWithEmbeddedObjPreDestroy()</tt>,
 * <tt>rcpWithEmbeddedObjPostDestory()</tt>, or <tt>rcpWithEmbeddedObj()</tt>.
 *
 * \relates RCP
 */
template<class TOrig, class Embedded, class T>
Embedded& getNonconstEmbeddedObj( const RCP<T>& p );


/** \brief Get an optional Ptr to a const embedded object if it was set by
 * calling <tt>rcpWithEmbeddedObjPreDestroy()</tt>,
 * <tt>rcpWithEmbeddedObjPostDestory()</tt>, or <tt>rcpWithEmbeddedObj()</tt>.
 *
 * \relates RCP
 */
template<class TOrig, class Embedded, class T>
Ptr<const Embedded> getOptionalEmbeddedObj( const RCP<T>& p );


/** \brief Get an optional Ptr to a non-const embedded object if it was set by
 * calling <tt>rcpWithEmbeddedObjPreDestroy()</tt>,
 * <tt>rcpWithEmbeddedObjPostDestory()</tt>, or <tt>rcpWithEmbeddedObj()</tt>.
 *
 * \relates RCP
 */
template<class TOrig, class Embedded, class T>
Ptr<Embedded> getOptionalNonconstEmbeddedObj( const RCP<T>& p );


/** \brief Get the parent back from an inverted ownership RCP.
 *
 * Retrieves the RCP<ParentT> object set through
 * <tt>rcpWithInvertedObjOwnership()</tt>.
 */
template<class ParentT, class T>
RCP<ParentT> getInvertedObjOwnershipParent(const RCP<T> &invertedChild);


/** \brief Output stream inserter.
 *
 * The implementation of this function just print pointer addresses and
 * therefore puts no restrictions on the data types involved.
 *
 * \relates RCP
 */
template<class T>
std::ostream& operator<<( std::ostream& out, const RCP<T>& p );


} // end namespace Teuchos


#endif  // TEUCHOS_RCP_DECL_HPP
