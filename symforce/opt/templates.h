/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#pragma once

#include <functional>

namespace sym {

// ------------------------------------------------------------------------------------------------
// C++14 implementation of remove_cvref
template <class T>
struct remove_cvref {
  using type = std::remove_cv_t<std::remove_reference_t<T>>;
};

template <typename T>
using remove_cvref_t = typename remove_cvref<T>::type;

// ------------------------------------------------------------------------------------------------
// Function traits
//
// Extracts the number of arguments and types of the arguments and return value.
// Handles:
// - Function pointers
// - Member function pointers
// - Functors (objects with operator())
// - Lambdas
// - std::bind expressions
// ------------------------------------------------------------------------------------------------

template <typename T>
struct function_traits;

// Traits implementation
template <typename ReturnType, typename... Args>
struct function_traits<ReturnType(Args...)> {
  using return_type = ReturnType;
  using base_return_type = typename std::decay_t<return_type>;

  static constexpr std::size_t num_arguments = sizeof...(Args);

  template <std::size_t N>
  struct arg {
    static_assert(N < num_arguments, "error: invalid parameter index.");
    using type = typename std::tuple_element<N, std::tuple<Args...>>::type;
    using base_type = typename std::decay_t<type>;
  };
};

template <typename ReturnType, typename... Args>
constexpr std::size_t function_traits<ReturnType(Args...)>::num_arguments;

// Specializations to remove type modifiers
template <typename T>
struct function_traits<T*> : public function_traits<T> {};
template <typename T>
struct function_traits<T&> : public function_traits<T> {};
template <typename T>
struct function_traits<T&&> : public function_traits<T> {};
template <typename T>
struct function_traits<const T> : public function_traits<T> {};
template <typename T>
struct function_traits<volatile T> : public function_traits<T> {};

// ------------------------------------------------------------------------------------------------
// Specialize for member function pointers
template <typename ClassType, typename ReturnType, typename... Args>
struct function_traits<ReturnType (ClassType::*)(Args...)>
    : public function_traits<ReturnType(Args...)> {};

// ------------------------------------------------------------------------------------------------
// Specialize for const member function pointers
template <typename ClassType, typename ReturnType, typename... Args>
struct function_traits<ReturnType (ClassType::*)(Args...) const>
    : public function_traits<ReturnType(Args...)> {};

// ------------------------------------------------------------------------------------------------
// Specialize for functors
template <typename T>
struct function_traits : public function_traits<decltype(&T::operator())> {};

// ------------------------------------------------------------------------------------------------
// Specialize for std::bind
namespace internal {

/// A template-metaprogramming max<a, b>, where a and b are std::placeholders
template <typename a, typename b>
struct max {
  using type =
      std::conditional_t<(std::is_placeholder<a>::value > std::is_placeholder<b>::value), a, b>;
};

/// The type of a std::bind expression in various standard library implementations - this is
/// unspecified by the standard
#if defined _LIBCPP_VERSION  // libc++ (Clang)
#define SYM_BIND_TYPE std::__1::__bind<ReturnTypeT (&)(Args...), FArgs...>
#elif defined __GLIBCXX__  // glibc++ (GNU C++)
#define SYM_BIND_TYPE std::_Bind<ReturnTypeT (*(FArgs...))(Args...)>
#elif defined _MSC_VER  // MS Visual Studio
#define SYM_BIND_TYPE std::_Binder<std::_Unforced, ReturnTypeT(__cdecl&)(Args...), FArgs...>
#else
#error "Unsupported C++ compiler / standard library"
#endif

/// The tuple of bound args for a std::bind expression
template <typename ReturnTypeT, typename... Args>
struct bound_args {
  using type = std::tuple<Args...>;
};

template <typename ReturnTypeT, typename... Args, typename... FArgs>
struct bound_args<SYM_BIND_TYPE> : bound_args<ReturnTypeT, FArgs...> {};

/// The tuple of original args for a std::bind expression
template <typename... Args>
struct orig_args {
  using type = std::tuple<Args...>;
};

template <typename ReturnTypeT, typename... Args, typename... FArgs>
struct orig_args<SYM_BIND_TYPE> : orig_args<Args...> {};

/// Helper to get the max placeholder in a std::bind expression
template <int i, typename Bind>
struct max_placeholder {
  using type = typename max<typename std::tuple_element<i, typename bound_args<Bind>::type>::type,
                            typename max_placeholder<i - 1, Bind>::type>::type;
};

template <typename Bind>
struct max_placeholder<0, Bind> {
  using type = typename std::tuple_element<0, typename bound_args<Bind>::type>::type;
};

/// Recursive helper to get the type for placeholder `i` in args [0, curr] of a std::bind expression
///
/// Does not check for multiple occurrences of the same placeholder
template <int i, int curr, typename Bind>
struct type_for_placeholder_impl {
  using curr_orig_type = typename std::tuple_element<curr, typename orig_args<Bind>::type>::type;
  using curr_bound_type = typename std::tuple_element<curr, typename bound_args<Bind>::type>::type;
  using type =
      std::conditional_t<(i + 1 == std::is_placeholder<curr_bound_type>::value), curr_orig_type,
                         typename type_for_placeholder_impl<i, curr - 1, Bind>::type>;
};

template <int i, typename Bind>
struct type_for_placeholder_impl<i, -1, Bind> {
  using type = void;
};

/// Helper to get the type for placeholder `i` in a std::bind expression
///
/// Does not check for multiple occurrences of the same placeholder
template <int i, typename Bind>
struct type_for_placeholder {
  using type = typename type_for_placeholder_impl<
      i, std::tuple_size<typename bound_args<Bind>::type>::value - 1, Bind>::type;
};

/// Helper to get the types for all placeholders in a std::bind expression, i.e. the types the
/// expression should be invoked with
///
/// Requires that all placeholders up to the max placeholder are used (no gaps)
template <typename Bind, typename Seq>
struct dispatch_types;

template <typename Bind, size_t... Indices>
struct dispatch_types<Bind, std::index_sequence<Indices...>> {
  using type = typename std::tuple<typename type_for_placeholder<Indices, Bind>::type...>;
};

/// Helper to get the signature of the result of a std::bind expression, i.e. the types the
/// expression should be invoked with
///
/// Requires that all placeholders up to the max placeholder are used (no gaps)
template <typename Bind>
struct bind_signature {
  static constexpr size_t N = std::is_placeholder<typename max_placeholder<
      std::tuple_size<typename bound_args<Bind>::type>::value - 1, Bind>::type>::value;
  using type = typename dispatch_types<Bind, std::make_index_sequence<N>>::type;
};

}  // namespace internal

template <typename ReturnTypeT, typename... Args, typename... FArgs>
struct function_traits<SYM_BIND_TYPE> {
 private:
  using bind_type = SYM_BIND_TYPE;
  using signature = typename internal::bind_signature<bind_type>;

 public:
  using return_type = ReturnTypeT;
  using base_return_type = typename std::decay_t<return_type>;

  static constexpr std::size_t num_arguments = signature::N;

  template <std::size_t N>
  struct arg {
    static_assert(N < num_arguments, "error: invalid parameter index.");
    using type = typename std::tuple_element<N, typename signature::type>::type;
    using base_type = typename std::decay_t<type>;
  };
};

template <typename ReturnTypeT, typename... Args, typename... FArgs>
constexpr std::size_t function_traits<SYM_BIND_TYPE>::num_arguments;

#undef SYM_BIND_TYPE

}  // namespace sym
