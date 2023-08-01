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
// ------------------------------------------------------------------------------------------------

template <typename T>
struct function_traits;

// Traits implementation
template <typename ReturnType, typename... Args>
struct function_traits<ReturnType(Args...)> {
  using return_type = ReturnType;
  using base_return_type = typename std::decay_t<return_type>;
  using std_function_type = typename std::function<ReturnType(Args...)>;

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

// Specialize for member function pointers
template <typename ClassType, typename ReturnType, typename... Args>
struct function_traits<ReturnType (ClassType::*)(Args...)>
    : public function_traits<ReturnType(Args...)> {};

// Specialize for const member function pointers
template <typename ClassType, typename ReturnType, typename... Args>
struct function_traits<ReturnType (ClassType::*)(Args...) const>
    : public function_traits<ReturnType(Args...)> {};

// ------------------------------------------------------------------------------------------------
// Specialize for functors
template <typename T>
struct function_traits : public function_traits<decltype(&T::operator())> {};

}  // namespace sym
