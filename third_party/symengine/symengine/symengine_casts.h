#ifndef SYMENGINE_CASTS_H
#define SYMENGINE_CASTS_H

#include <iostream>
#include <limits>
#include <symengine/symengine_config.h>
#include <symengine/symengine_assert.h>

namespace SymEngine
{

// Reference modifications.
template <typename T>
struct remove_reference {
    typedef T type;
};
template <typename T>
struct remove_reference<T &> {
    typedef T type;
};

// Use implicit_cast as a safe version of static_cast or const_cast
// for upcasting in the type hierarchy (i.e. casting a pointer to Foo
// to a pointer to SuperclassOfFoo or casting a pointer to Foo to
// a const pointer to Foo).
// When you use implicit_cast, the compiler checks that the cast is safe.
// Such explicit implicit_casts are necessary in surprisingly many
// situations where C++ demands an exact type match instead of an
// argument type convertable to a target type.
//
// The From type can be inferred, so the preferred syntax for using
// implicit_cast is the same as for static_cast etc.:
//
//   implicit_cast<ToType>(expr)

template <typename To, typename From>
inline To implicit_cast(const From &f)
{
    return f;
}

// When you upcast (that is, cast a pointer from type Foo to type
// SuperclassOfFoo), it's fine to use implicit_cast<>, since upcasts
// always succeed.  When you downcast (that is, cast a pointer from
// type Foo to type SubclassOfFoo), static_cast<> isn't safe, because
// how do you know the pointer is really of type SubclassOfFoo?  It
// could be a bare Foo, or of type DifferentSubclassOfFoo.  Thus,
// when you downcast, you should use this macro.  In debug mode, we
// use dynamic_cast<> to double-check the downcast is legal (we die
// if it's not).  In normal mode, we do the efficient static_cast<>
// instead.  Thus, it's important to test in debug mode to make sure
// the cast is legal!
//    This is the only place in the code we should use dynamic_cast<>.
// In particular, you SHOULDN'T be using dynamic_cast<> in order to
// do RTTI (eg code like this:
//    if (dynamic_cast<Subclass1>(foo)) HandleASubclass1Object(foo);
//    if (dynamic_cast<Subclass2>(foo)) HandleASubclass2Object(foo);
// You should design the code some other way not to need this.

template <typename To, typename From> // use like this: down_cast<T*>(foo).
inline To down_cast(From *f)          // Only accept pointers.
{
    // Ensures that To is a sub-type of From *.  This test is here only
    // for compile-time type checking, and has no overhead in an
    // optimized build at run-time, as it will be optimized away
    // completely.
    if (false) {
        implicit_cast<From *, To>(0);
    }

    SYMENGINE_ASSERT(f == NULL || dynamic_cast<To>(f) != NULL);

    return static_cast<To>(f);
}

template <typename To, typename From> // use like this: down_cast<T&>(foo);
inline To down_cast(From &f)
{
    typedef typename remove_reference<To>::type *ToAsPointer;
    // Ensures that To is a sub-type of From *.  This test is here only
    // for compile-time type checking, and has no overhead in an
    // optimized build at run-time, as it will be optimized away
    // completely.
    if (false) {
        implicit_cast<From *, ToAsPointer>(0);
    }

    SYMENGINE_ASSERT(dynamic_cast<ToAsPointer>(&f) != NULL);

    return *static_cast<ToAsPointer>(&f);
}

template <typename To, typename From>
inline To numeric_cast(
    From f,
    typename std::enable_if<
        (std::is_signed<From>::value && std::is_signed<To>::value)
        || (std::is_unsigned<From>::value && std::is_unsigned<To>::value)>::type
        * = nullptr)
{
    SYMENGINE_ASSERT(f <= std::numeric_limits<To>::max());
    SYMENGINE_ASSERT(f >= std::numeric_limits<To>::min());
    return static_cast<To>(f);
}

template <typename To, typename From>
inline To numeric_cast(
    From f,
    typename std::enable_if<(std::is_signed<From>::value
                             && std::is_unsigned<To>::value)>::type * = nullptr)
{
#ifdef WITH_SYMENGINE_ASSERT
    // Above ifdef is needed to avoid a warning about unused typedefs
    typedef typename std::make_unsigned<From>::type unsigned_from_type;
    SYMENGINE_ASSERT(f >= 0);
    SYMENGINE_ASSERT(static_cast<unsigned_from_type>(f)
                     <= std::numeric_limits<To>::max());
#endif
    return static_cast<To>(f);
}

template <typename To, typename From>
inline To numeric_cast(
    From f,
    typename std::enable_if<(std::is_unsigned<From>::value
                             && std::is_signed<To>::value)>::type * = nullptr)
{
#ifdef WITH_SYMENGINE_ASSERT
    typedef typename std::make_unsigned<To>::type unsigned_to_type;
    SYMENGINE_ASSERT(
        f <= static_cast<unsigned_to_type>(std::numeric_limits<To>::max()));

#endif
    return static_cast<To>(f);
}

} // namespace SymEngine

#endif // SYMENGINE_CASTS_H
