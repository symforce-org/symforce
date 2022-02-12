/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#pragma once

#include <functional>

namespace sym {

// ------------------------------------------------------------------------------------------------
// Sequence generator
//
// TODO(hayk): Might be able to replace with std::integer_sequence
//
// Useful for generating a sequence at compile time, in order to apply the unpacking operator
// to the sequence. For example, this is useful to unpack a tuple when calling a function.
// See http://stackoverflow.com/questions/7858817/
// ------------------------------------------------------------------------------------------------

template <int...>
struct Sequence {};

// Generates a Sequence from 0 to N-1.
template <int N, int... S>
struct RangeGenerator : RangeGenerator<N - 1, N - 1, S...> {};

template <int... S>
struct RangeGenerator<0, S...> {
  typedef Sequence<S...> Range;
};

template <typename F, size_t... Is>
auto indices_impl(F f, std::index_sequence<Is...>) {
  return f(std::integral_constant<size_t, Is>()...);
}

template <size_t N, typename F>
auto indices(F f) {
  return indices_impl(f, std::make_index_sequence<N>());
}

// ------------------------------------------------------------------------------------------------
// Function traits
//
// Extracts the number of arguments and types of the arguments and return value.
// Handle generic functors by looking at the 'operator()'.
// ------------------------------------------------------------------------------------------------

template <typename T>
struct function_traits : public function_traits<decltype(&T::operator())> {};

// Traits implementation
template <typename ReturnType, typename... Args>
struct function_traits<ReturnType(Args...)> {
  using return_type = ReturnType;
  using base_return_type = typename std::decay<return_type>::type;
  using std_function_type = typename std::function<ReturnType(Args...)>;

  static constexpr std::size_t num_arguments = sizeof...(Args);

  template <std::size_t N>
  struct arg {
    static_assert(N < num_arguments, "error: invalid parameter index.");
    using type = typename std::tuple_element<N, std::tuple<Args...>>::type;
    using base_type = typename std::decay<type>::type;
  };
};

// Specialize for function pointers
template <typename ReturnType, typename... Args>
struct function_traits<ReturnType (*)(Args...)> : public function_traits<ReturnType(Args...)> {};

// Specialize for member function pointers
template <typename ClassType, typename ReturnType, typename... Args>
struct function_traits<ReturnType (ClassType::*)(Args...)>
    : public function_traits<ReturnType(Args...)> {};

// Specialize for const member function pointers
template <typename ClassType, typename ReturnType, typename... Args>
struct function_traits<ReturnType (ClassType::*)(Args...) const>
    : public function_traits<ReturnType(Args...)> {};

}  // namespace sym
