#ifndef SYMENGINE_RCP_H
#define SYMENGINE_RCP_H

#include <iostream>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <ciso646>

#include <symengine/symengine_config.h>
#include <symengine/symengine_assert.h>

#if defined(WITH_SYMENGINE_RCP)

#if defined(WITH_SYMENGINE_THREAD_SAFE)
#include <atomic>
#endif

#else

// Include all Teuchos headers here:
#include <symengine/utilities/teuchos/Teuchos_RCP.hpp>
#include <symengine/utilities/teuchos/Teuchos_TypeNameTraits.hpp>

#endif

namespace SymEngine
{

#if defined(WITH_SYMENGINE_RCP)

/* Ptr */

// Ptr is always pointing to a valid object (can never be nullptr).

template <class T>
class Ptr
{
public:
    inline explicit Ptr(T *ptr) : ptr_(ptr)
    {
        SYMENGINE_ASSERT(ptr_ != nullptr)
    }
    inline Ptr(const Ptr<T> &ptr) : ptr_(ptr.ptr_) {}
    template <class T2>
    inline Ptr(const Ptr<T2> &ptr) : ptr_(ptr.get())
    {
    }
    Ptr<T> &operator=(const Ptr<T> &ptr)
    {
        ptr_ = ptr.get();
        return *this;
    }
#if defined(HAVE_DEFAULT_CONSTRUCTORS)
    inline Ptr(Ptr &&) = default;
    Ptr<T> &operator=(Ptr &&) = default;
#endif
    inline T *operator->() const
    {
        return ptr_;
    }
    inline T &operator*() const
    {
        return *ptr_;
    }
    inline T *get() const
    {
        return ptr_;
    }
    inline T *getRawPtr() const
    {
        return get();
    }
    inline const Ptr<T> ptr() const
    {
        return *this;
    }

private:
    T *ptr_;
};

template <typename T>
inline Ptr<T> outArg(T &arg)
{
    return Ptr<T>(&arg);
}

/** \brief Create a pointer to a object from an object reference.
 *
 * \relates Ptr
 */
template <typename T>
inline Ptr<T> ptrFromRef(T &arg)
{
    return Ptr<T>(&arg);
}

/* RCP */

enum ENull { null };

// RCP can be null. Functionally it should be equivalent to Teuchos::RCP.

template <class T>
class RCP
{
public:
    RCP(ENull null_arg = null) : ptr_(nullptr) {}
    explicit RCP(T *p) : ptr_(p)
    {
        SYMENGINE_ASSERT(ptr_ != nullptr)
        (ptr_->refcount_)++;
    }
    // Copy constructor
    RCP(const RCP<T> &rp) : ptr_(rp.ptr_)
    {
        if (not is_null())
            (ptr_->refcount_)++;
    }
    // Copy constructor
    template <class T2>
    RCP(const RCP<T2> &r_ptr) : ptr_(r_ptr.get())
    {
        if (not is_null())
            (ptr_->refcount_)++;
    }
    // Move constructor
    RCP(RCP<T> &&rp) SYMENGINE_NOEXCEPT : ptr_(rp.ptr_)
    {
        rp.ptr_ = nullptr;
    }
    // Move constructor
    template <class T2>
    RCP(RCP<T2> &&r_ptr)
    SYMENGINE_NOEXCEPT : ptr_(r_ptr.get())
    {
        r_ptr._set_null();
    }
    ~RCP() SYMENGINE_NOEXCEPT
    {
        if (ptr_ != nullptr and --(ptr_->refcount_) == 0)
            delete ptr_;
    }
    T *operator->() const
    {
        SYMENGINE_ASSERT(ptr_ != nullptr)
        return ptr_;
    }
    T &operator*() const
    {
        SYMENGINE_ASSERT(ptr_ != nullptr)
        return *ptr_;
    }
    T *get() const
    {
        return ptr_;
    }
    Ptr<T> ptr() const
    {
        return Ptr<T>(get());
    }
    bool is_null() const
    {
        return ptr_ == nullptr;
    }
    template <class T2>
    bool operator==(const RCP<T2> &p2) const
    {
        return ptr_ == p2.ptr_;
    }
    template <class T2>
    bool operator!=(const RCP<T2> &p2) const
    {
        return ptr_ != p2.ptr_;
    }
    // Copy assignment
    RCP<T> &operator=(const RCP<T> &r_ptr)
    {
        T *r_ptr_ptr_ = r_ptr.ptr_;
        if (not r_ptr.is_null())
            (r_ptr_ptr_->refcount_)++;
        if (not is_null() and --(ptr_->refcount_) == 0)
            delete ptr_;
        ptr_ = r_ptr_ptr_;
        return *this;
    }
    // Move assignment
    RCP<T> &operator=(RCP<T> &&r_ptr)
    {
        std::swap(ptr_, r_ptr.ptr_);
        return *this;
    }
    void reset()
    {
        if (not is_null() and --(ptr_->refcount_) == 0)
            delete ptr_;
        ptr_ = nullptr;
    }
    // Don't use this function directly:
    void _set_null()
    {
        ptr_ = nullptr;
    }

private:
    T *ptr_;
};

template <class T>
inline RCP<T> rcp(T *p)
{
    return RCP<T>(p);
}

template <class T2, class T1>
inline RCP<T2> rcp_static_cast(const RCP<T1> &p1)
{
    // Make the compiler check if the conversion is legal
    T2 *check = static_cast<T2 *>(p1.get());
    return RCP<T2>(check);
}

template <class T2, class T1>
inline RCP<T2> rcp_dynamic_cast(const RCP<T1> &p1)
{
    if (not p1.is_null()) {
        T2 *p = nullptr;
        // Make the compiler check if the conversion is legal
        p = dynamic_cast<T2 *>(p1.get());
        if (p) {
            return RCP<T2>(p);
        }
    }
    throw std::runtime_error("rcp_dynamic_cast: cannot convert.");
}

template <class T2, class T1>
inline RCP<T2> rcp_const_cast(const RCP<T1> &p1)
{
    // Make the compiler check if the conversion is legal
    T2 *check = const_cast<T2 *>(p1.get());
    return RCP<T2>(check);
}

template <class T>
inline bool operator==(const RCP<T> &p, ENull)
{
    return p.get() == nullptr;
}

template <typename T>
std::string typeName(const T &t)
{
    return "RCP<>";
}

void print_stack_on_segfault();

#else

using Teuchos::null;
using Teuchos::outArg;
using Teuchos::print_stack_on_segfault;
using Teuchos::Ptr;
using Teuchos::ptrFromRef;
using Teuchos::RCP;
using Teuchos::rcp;
using Teuchos::rcp_const_cast;
using Teuchos::rcp_dynamic_cast;
using Teuchos::rcp_static_cast;
using Teuchos::typeName;

#endif

template <class T>
class EnableRCPFromThis
{
    // Public interface
public:
    //! Get RCP<T> pointer to self (it will cast the pointer to T)
    inline RCP<T> rcp_from_this()
    {
#if defined(WITH_SYMENGINE_RCP)
        return rcp(static_cast<T *>(this));
#else
        return rcp_static_cast<T>(weak_self_ptr_.create_strong());
#endif
    }

    //! Get RCP<const T> pointer to self (it will cast the pointer to const T)
    inline RCP<const T> rcp_from_this() const
    {
#if defined(WITH_SYMENGINE_RCP)
        return rcp(static_cast<const T *>(this));
#else
        return rcp_static_cast<const T>(weak_self_ptr_.create_strong());
#endif
    }

    //! Get RCP<T2> pointer to self (it will cast the pointer to T2)
    template <class T2>
    inline RCP<const T2> rcp_from_this_cast() const
    {
#if defined(WITH_SYMENGINE_RCP)
        return rcp(static_cast<const T2 *>(this));
#else
        return rcp_static_cast<const T2>(weak_self_ptr_.create_strong());
#endif
    }

    unsigned int use_count() const
    {
#if defined(WITH_SYMENGINE_RCP)
        return refcount_;
#else
        return weak_self_ptr_.strong_count();
#endif
    }

    // Everything below is private interface
private:
#if defined(WITH_SYMENGINE_RCP)

//! Public variables if defined with SYMENGINE_RCP
// The reference counter is defined either as "unsigned int" (faster, but
// not thread safe) or as std::atomic<unsigned int> (slower, but thread
// safe). Semantically they are almost equivalent, except that the
// pre-decrement operator `operator--()` returns a copy for std::atomic
// instead of a reference to itself.
// The refcount_ is defined as mutable, because it does not change the
// state of the instance, but changes when more copies
// of the same instance are made.
#if defined(WITH_SYMENGINE_THREAD_SAFE)
    mutable std::atomic<unsigned int> refcount_; // reference counter
#else
    mutable unsigned int refcount_; // reference counter
#endif // WITH_SYMENGINE_THREAD_SAFE
public:
    EnableRCPFromThis() : refcount_(0) {}

private:
#else
    mutable RCP<T> weak_self_ptr_;

    void set_weak_self_ptr(const RCP<T> &w)
    {
        weak_self_ptr_ = w;
    }

    void set_weak_self_ptr(const RCP<const T> &w) const
    {
        weak_self_ptr_ = rcp_const_cast<T>(w);
    }
#endif // WITH_SYMENGINE_RCP

#if defined(WITH_SYMENGINE_RCP)
    template <class T_>
    friend class RCP;
#endif

    template <typename T_, typename... Args>
    friend inline RCP<T_> make_rcp(Args &&...args);
};

template <typename T, typename... Args>
inline RCP<T> make_rcp(Args &&...args)
{
#if defined(WITH_SYMENGINE_RCP)
    return rcp(new T(std::forward<Args>(args)...));
#else
    RCP<T> p = rcp(new T(std::forward<Args>(args)...));
    p->set_weak_self_ptr(p.create_weak());
    return p;
#endif
}

} // namespace SymEngine

#endif
