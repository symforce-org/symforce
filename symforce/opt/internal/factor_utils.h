/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#pragma once

/** @file
 * Utilities for converting various types of factor functors into what's needed for the polymorphic
 * std::function that's actually stored and used by sym::Factor
 *
 * Primarily intended to be included by factor.tcc and used internally there
 */

#include <utility>

#include "../factor.h"

namespace sym {
namespace internal {

// ------------------------------------------------------------------------------------------------
// Factor::Jacobian constructor dispatcher
//
// Struct to allow partial specialization based on whether matrices are dynamic, sparse, and/or maps
// ------------------------------------------------------------------------------------------------

template <bool IsDynamic, bool IsSparse, bool IsMap, typename Scalar>
struct JacobianDispatcher {
  template <typename Functor>
  auto operator()(Functor&& func);
};

// Struct to help extract matrix types from signature
template <typename Functor>
struct JacobianFuncTypeHelper {
  using Traits = function_traits<Functor>;

  template <int Index>
  using ArgType = typename Traits::template arg<Index>::base_type;

  using ResidualVec = typename std::remove_pointer_t<ArgType<Traits::num_arguments - 2>>;
  using JacobianMat = typename std::remove_pointer_t<ArgType<Traits::num_arguments - 1>>;
};

/**
 * Helper for automatically extracting keys from values.
 */
template <typename Scalar, typename Functor>
struct JacobianFuncValuesExtractor {
  template <int Index>
  using ArgType = typename JacobianFuncTypeHelper<Functor>::template ArgType<Index>;

  using ResidualVec = typename JacobianFuncTypeHelper<Functor>::ResidualVec;
  using JacobianMat = typename JacobianFuncTypeHelper<Functor>::JacobianMat;

  /** Pull out the arg given by Index from the Values. */
  template <int Index>
  static ArgType<Index> GetValue(const sym::Values<Scalar>& values,
                                 const std::vector<index_entry_t>& keys) {
    return values.template At<ArgType<Index>>(keys[Index]);
  }

  template <std::size_t... S>
  static void InvokeAtRange(const Functor& func, const sym::Values<Scalar>& values,
                            const std::vector<index_entry_t>& keys, ResidualVec* residual,
                            JacobianMat* jacobian, std::index_sequence<S...>) {
    func(GetValue<S>(values, keys)..., residual, jacobian);
  }

  /**
   * Invokes the user function with the proper input args extracted from the Values.
   *
   * Overload for pointer-to-matrix outputs
   */
  static void Invoke(const Functor& func, const sym::Values<Scalar>& values,
                     const std::vector<index_entry_t>& keys, ResidualVec* residual,
                     JacobianMat* jacobian) {
    constexpr auto num_inputs = function_traits<Functor>::num_arguments - 2;
    SYM_ASSERT(keys.size() == num_inputs);
    InvokeAtRange(func, values, keys, residual, jacobian, std::make_index_sequence<num_inputs>());
  }

  template <std::size_t... S>
  static void InvokeAtRange(const Functor& func, const sym::Values<Scalar>& values,
                            const std::vector<index_entry_t>& keys, ResidualVec residual,
                            JacobianMat jacobian, std::index_sequence<S...>) {
    func(GetValue<S>(values, keys)..., residual, jacobian);
  }

  /**
   * Invokes the user function with the proper input args extracted from the Values.
   *
   * Overload for Eigen::Map outputs
   */
  static void Invoke(const Functor& func, const sym::Values<Scalar>& values,
                     const std::vector<index_entry_t>& keys, ResidualVec residual,
                     JacobianMat jacobian) {
    constexpr auto num_inputs = function_traits<Functor>::num_arguments - 2;
    SYM_ASSERT(keys.size() == num_inputs);
    InvokeAtRange(func, values, keys, residual, jacobian, std::make_index_sequence<num_inputs>());
  }
};

// ----------------------------------------------------------------------------
// Factor::Jacobian constructor support for dynamic matrices
// ----------------------------------------------------------------------------

template <typename HMatrixType, typename JMatrixType = HMatrixType>
struct RankUpdateHelper {
  void operator()(HMatrixType& hessian, const JMatrixType& jacobian) const {
    // NOTE(aaron):  Testing seemed to show that this matrix multiply is faster than
    // rankUpdate, at least for smallish jacobians that are still large enough for Eigen to
    // not inline the whole multiplication.  More investigation would be necessary to figure
    // out why, and if it's expected to be faster for large jacobians
    hessian.template triangularView<Eigen::Lower>() =
        (jacobian.transpose() * jacobian).template triangularView<Eigen::Lower>();
  }
};

template <typename Scalar>
struct RankUpdateHelper<Eigen::SparseMatrix<Scalar>> {
  void operator()(Eigen::SparseMatrix<Scalar>& hessian,
                  const Eigen::SparseMatrix<Scalar>& jacobian) const {
    hessian.template selfadjointView<Eigen::Lower>() =
        (jacobian.transpose() * jacobian).template selfadjointView<Eigen::Lower>();
  }
};

/**
 * Precondition: residual and jacobian have the same number of rows
 */
template <typename RVecType, typename JMatrixType, typename HMatrixType, typename Scalar>
void CalculateHessianRhs(const RVecType& residual, const JMatrixType& jacobian,
                         HMatrixType* hessian, VectorX<Scalar>* rhs) {
  // Compute the lower triangle of the hessian if needed
  if (hessian != nullptr) {
    hessian->resize(jacobian.cols(), jacobian.cols());

    RankUpdateHelper<HMatrixType, JMatrixType>{}(*hessian, jacobian);
  }

  // Compute RHS if needed
  if (rhs != nullptr) {
    (*rhs) = jacobian.transpose() * residual;
  }
}

template <typename Scalar, typename Functor>
auto JacobianDynamic(Functor&& func) {
  using Mat = typename JacobianFuncValuesExtractor<Scalar, Functor>::JacobianMat;
  using FunctorType = std::decay_t<Functor>;

  return [func = std::forward<Functor>(func)](
             const Values<Scalar>& values, const std::vector<index_entry_t>& keys_to_func,
             VectorX<Scalar>* residual, Mat* jacobian, Mat* hessian, VectorX<Scalar>* rhs) {
    JacobianFuncValuesExtractor<Scalar, FunctorType>::Invoke(func, values, keys_to_func, residual,
                                                             jacobian);
    SYM_ASSERT(residual != nullptr);
    if (jacobian == nullptr) {
      SYM_ASSERT(hessian == nullptr);
      SYM_ASSERT(rhs == nullptr);
    } else {
      SYM_ASSERT(residual->rows() == jacobian->rows());
      CalculateHessianRhs(*residual, *jacobian, hessian, rhs);
    }
  };
}

/** Specialize the dispatch mechanism */
template <typename Scalar, bool IsSparse>
struct JacobianDispatcher<true /* is_dynamic */, IsSparse, false /* is_map */, Scalar> {
  template <typename Functor>
  auto operator()(Functor&& func) {
    return JacobianDynamic<Scalar>(std::forward<Functor>(func));
  }
};

// ----------------------------------------------------------------------------
// Factor::Jacobian constructor support for fixed size matrices
// ----------------------------------------------------------------------------

// NOTE(aaron): This is implemented separately from JacobianDynamic (as opposed to just wrapping
// `func` to return the residual and jacobian as VectorX and MatrixX) so that Eigen can know the
// size of the matrix multiplies at compile time for computing H = J^T * J and rhs = J^T * b.  This
// does produce a noticeable speedup for small factors, and is not expected to be slower for any
// size factors.
template <typename Scalar, typename Functor>
auto JacobianFixed(Functor&& func) {
  using Traits = function_traits<Functor>;
  using FunctorType = std::decay_t<Functor>;

  // Get matrix types from function signature
  using JacobianMat = typename std::remove_pointer_t<
      typename Traits::template arg<Traits::num_arguments - 1>::type>;

  return [func = std::forward<Functor>(func)](const Values<Scalar>& values,
                                              const std::vector<index_entry_t>& keys_to_func,
                                              VectorX<Scalar>* residual, MatrixX<Scalar>* jacobian,
                                              MatrixX<Scalar>* hessian, VectorX<Scalar>* rhs) {
    // Get dimensions (these have already been sanity checked in Factor::Jacobian)
    constexpr int M = JacobianMat::RowsAtCompileTime;
    constexpr int N = JacobianMat::ColsAtCompileTime;

    SYM_ASSERT(residual != nullptr);
    Eigen::Matrix<Scalar, M, 1> residual_fixed;

    if (jacobian != nullptr) {
      // jacobian is requested
      Eigen::Matrix<Scalar, M, N> jacobian_fixed;
      JacobianFuncValuesExtractor<Scalar, FunctorType>::Invoke(func, values, keys_to_func,
                                                               &residual_fixed, &jacobian_fixed);
      (*jacobian) = jacobian_fixed;
      CalculateHessianRhs(residual_fixed, jacobian_fixed, hessian, rhs);
    } else {
      // jacobian not requested
      Eigen::Matrix<Scalar, M, N>* const jacobian_invoke_arg = nullptr;
      JacobianFuncValuesExtractor<Scalar, FunctorType>::Invoke(
          func, values, keys_to_func, &residual_fixed, jacobian_invoke_arg);

      // Check that the hessian and rhs weren't requested without the jacobian
      SYM_ASSERT(hessian == nullptr);
      SYM_ASSERT(rhs == nullptr);
    }

    (*residual) = residual_fixed;
  };
}

/** Specialize the dispatch mechanism */
template <typename Scalar>
struct JacobianDispatcher<false /* is_dynamic */, false /* is_sparse */, false /* is_map */,
                          Scalar> {
  template <typename Functor>
  auto operator()(Functor&& func) {
    return JacobianFixed<Scalar>(std::forward<Functor>(func));
  }
};

template <typename Scalar, typename Functor>
auto JacobianFixedMap(Functor&& func) {
  using Traits = function_traits<Functor>;
  using FunctorType = std::decay_t<Functor>;

  // Get matrix types from function signature
  using JacobianMat = typename std::remove_pointer_t<
      typename Traits::template arg<Traits::num_arguments - 1>::type>;

  return [func = std::forward<Functor>(func)](const Values<Scalar>& values,
                                              const std::vector<index_entry_t>& keys_to_func,
                                              VectorX<Scalar>* residual, MatrixX<Scalar>* jacobian,
                                              MatrixX<Scalar>* hessian, VectorX<Scalar>* rhs) {
    // Get dimensions (these have already been sanity checked in Factor::Jacobian)
    constexpr int M = JacobianMat::RowsAtCompileTime;
    constexpr int N = JacobianMat::ColsAtCompileTime;

    SYM_ASSERT(residual != nullptr);
    residual->resize(M);
    const Eigen::Map<Eigen::Matrix<Scalar, M, 1>> residual_map{residual->data()};

    if (jacobian != nullptr) {
      // jacobian is requested
      jacobian->resize(M, N);
      const Eigen::Map<Eigen::Matrix<Scalar, M, N>> jacobian_map{jacobian->data()};
      JacobianFuncValuesExtractor<Scalar, FunctorType>::Invoke(func, values, keys_to_func,
                                                               residual_map, jacobian_map);
      CalculateHessianRhs(*residual, *jacobian, hessian, rhs);
    } else {
      // jacobian not requested
      Eigen::Map<Eigen::Matrix<Scalar, M, N>> const jacobian_invoke_arg{nullptr, M, N};
      JacobianFuncValuesExtractor<Scalar, FunctorType>::Invoke(func, values, keys_to_func,
                                                               residual_map, jacobian_invoke_arg);

      // Check that the hessian and rhs weren't requested without the jacobian
      SYM_ASSERT(hessian == nullptr);
      SYM_ASSERT(rhs == nullptr);
    }
  };
}

/** Specialize the dispatch mechanism */
template <typename Scalar>
struct JacobianDispatcher<false /* is_dynamic */, false /* is_sparse */, true /* is_map */,
                          Scalar> {
  template <typename Functor>
  auto operator()(Functor&& func) {
    return JacobianFixedMap<Scalar>(std::forward<Functor>(func));
  }
};

// ------------------------------------------------------------------------------------------------
// Factor::Hessian constructor dispatcher
//
// Struct to allow partial specialization based on whether matrices are dynamic, sparse, and/or maps
// ------------------------------------------------------------------------------------------------

template <bool IsDynamic, bool IsSparse, bool IsMap, typename Scalar>
struct HessianDispatcher {
  template <typename Functor>
  auto operator()(Functor&& func);
};

// Struct to help extract matrix types from signature
template <typename Functor>
struct HessianFuncTypeHelper {
  using Traits = function_traits<Functor>;

  template <int Index>
  using ArgType = typename Traits::template arg<Index>::base_type;

  using ResidualVec = typename std::remove_pointer_t<ArgType<Traits::num_arguments - 4>>;
  using JacobianMat = typename std::remove_pointer_t<ArgType<Traits::num_arguments - 3>>;
  using HessianMat = typename std::remove_pointer_t<ArgType<Traits::num_arguments - 2>>;
  using RhsVec = typename std::remove_pointer_t<ArgType<Traits::num_arguments - 1>>;
};

/**
 * Helper for automatically extracting keys from values.
 */
template <typename Scalar, typename Functor>
struct HessianFuncValuesExtractor {
  template <int Index>
  using ArgType = typename HessianFuncTypeHelper<Functor>::template ArgType<Index>;

  using ResidualVec = typename HessianFuncTypeHelper<Functor>::ResidualVec;
  using JacobianMat = typename HessianFuncTypeHelper<Functor>::JacobianMat;
  using HessianMat = typename HessianFuncTypeHelper<Functor>::HessianMat;
  using RhsVec = typename HessianFuncTypeHelper<Functor>::RhsVec;

  /** Pull out the arg given by Index from the Values. */
  template <int Index>
  static ArgType<Index> GetValue(const sym::Values<Scalar>& values,
                                 const std::vector<index_entry_t>& keys) {
    return values.template At<ArgType<Index>>(keys[Index]);
  }

  template <std::size_t... S>
  static void InvokeAtRange(const Functor& func, const sym::Values<Scalar>& values,
                            const std::vector<index_entry_t>& keys, ResidualVec* residual,
                            JacobianMat* jacobian, HessianMat* hessian, RhsVec* rhs,
                            std::index_sequence<S...>) {
    func(GetValue<S>(values, keys)..., residual, jacobian, hessian, rhs);
  }

  /**
   * Invokes the user function with the proper input args extracted from the Values.
   *
   * Overload for pointer-to-matrix outputs
   */
  static void Invoke(const Functor& func, const sym::Values<Scalar>& values,
                     const std::vector<index_entry_t>& keys, ResidualVec* residual,
                     JacobianMat* jacobian, HessianMat* hessian, RhsVec* rhs) {
    constexpr auto num_inputs = function_traits<Functor>::num_arguments - 4;
    SYM_ASSERT(keys.size() == num_inputs);
    InvokeAtRange(func, values, keys, residual, jacobian, hessian, rhs,
                  std::make_index_sequence<num_inputs>());
  }

  template <std::size_t... S>
  static void InvokeAtRange(const Functor& func, const sym::Values<Scalar>& values,
                            const std::vector<index_entry_t>& keys, ResidualVec residual,
                            JacobianMat jacobian, HessianMat hessian, RhsVec rhs,
                            std::index_sequence<S...>) {
    func(GetValue<S>(values, keys)..., residual, jacobian, hessian, rhs);
  }

  /**
   * Invokes the user function with the proper input args extracted from the Values.
   *
   * Overload for Eigen::Map outputs
   */
  static void Invoke(const Functor& func, const sym::Values<Scalar>& values,
                     const std::vector<index_entry_t>& keys, ResidualVec residual,
                     JacobianMat jacobian, HessianMat hessian, RhsVec rhs) {
    constexpr auto num_inputs = function_traits<Functor>::num_arguments - 4;
    SYM_ASSERT(keys.size() == num_inputs);
    InvokeAtRange(func, values, keys, residual, jacobian, hessian, rhs,
                  std::make_index_sequence<num_inputs>());
  }
};

// ----------------------------------------------------------------------------
// Factor::Hessian constructor support for dynamic matrices
// ----------------------------------------------------------------------------

template <typename Scalar, typename Functor>
auto HessianDynamic(Functor&& func) {
  using JacobianMat = typename HessianFuncValuesExtractor<Scalar, Functor>::JacobianMat;
  using HessianMat = typename HessianFuncValuesExtractor<Scalar, Functor>::HessianMat;
  using FunctorType = std::decay_t<Functor>;

  return [func = std::forward<Functor>(func)](const Values<Scalar>& values,
                                              const std::vector<index_entry_t>& keys_to_func,
                                              VectorX<Scalar>* residual, JacobianMat* jacobian,
                                              HessianMat* hessian, VectorX<Scalar>* rhs) {
    HessianFuncValuesExtractor<Scalar, FunctorType>::Invoke(func, values, keys_to_func, residual,
                                                            jacobian, hessian, rhs);
  };
}

/** Specialize the dispatch mechanism */
template <bool IsSparse, typename Scalar>
struct HessianDispatcher<true /* is_dynamic */, IsSparse, false /* is_map */, Scalar> {
  template <typename Functor>
  auto operator()(Functor&& func) {
    return HessianDynamic<Scalar>(std::forward<Functor>(func));
  }
};

// ----------------------------------------------------------------------------
// Factor::Hessian constructor support for fixed size matrices
// ----------------------------------------------------------------------------

template <typename Scalar, typename Functor>
auto HessianFixedDense(Functor&& func) {
  // Get matrix types from function signature
  using JacobianMat = typename HessianFuncValuesExtractor<Scalar, Functor>::JacobianMat;
  using FunctorType = std::decay_t<Functor>;

  // Get dimensions (these have already been sanity checked in Factor::Hessian)
  constexpr int M = JacobianMat::RowsAtCompileTime;
  constexpr int N = JacobianMat::ColsAtCompileTime;

  return [func = std::forward<Functor>(func)](const Values<Scalar>& values,
                                              const std::vector<index_entry_t>& keys_to_func,
                                              VectorX<Scalar>* residual, MatrixX<Scalar>* jacobian,
                                              MatrixX<Scalar>* hessian, VectorX<Scalar>* rhs) {
    Eigen::Matrix<Scalar, M, 1> residual_fixed;
    Eigen::Matrix<Scalar, M, N> jacobian_fixed;
    Eigen::Matrix<Scalar, N, N> hessian_fixed;
    Eigen::Matrix<Scalar, N, 1> rhs_fixed;

    HessianFuncValuesExtractor<Scalar, FunctorType>::Invoke(
        func, values, keys_to_func, residual == nullptr ? nullptr : &residual_fixed,
        jacobian == nullptr ? nullptr : &jacobian_fixed,
        hessian == nullptr ? nullptr : &hessian_fixed, rhs == nullptr ? nullptr : &rhs_fixed);

    if (residual != nullptr) {
      (*residual) = residual_fixed;
    }

    if (jacobian != nullptr) {
      (*jacobian) = jacobian_fixed;
    }

    if (hessian != nullptr) {
      (*hessian) = hessian_fixed;
    }

    if (rhs != nullptr) {
      (*rhs) = rhs_fixed;
    }
  };
}

/** Specialize the dispatch mechanism */
template <typename Scalar>
struct HessianDispatcher<false /* is_dynamic */, false /* is_sparse */, false /* is_map */,
                         Scalar> {
  template <typename Functor>
  auto operator()(Functor&& func) {
    return HessianFixedDense<Scalar>(std::forward<Functor>(func));
  }
};

template <typename Scalar, typename Functor>
auto HessianFixedMapDense(Functor&& func) {
  // Get matrix types from function signature
  using JacobianMat = typename HessianFuncValuesExtractor<Scalar, Functor>::JacobianMat;
  using FunctorType = std::decay_t<Functor>;

  // Get dimensions (these have already been sanity checked in Factor::Hessian)
  constexpr int M = JacobianMat::RowsAtCompileTime;
  constexpr int N = JacobianMat::ColsAtCompileTime;

  return [func = std::forward<Functor>(func)](const Values<Scalar>& values,
                                              const std::vector<index_entry_t>& keys_to_func,
                                              VectorX<Scalar>* residual, MatrixX<Scalar>* jacobian,
                                              MatrixX<Scalar>* hessian, VectorX<Scalar>* rhs) {
    if (residual != nullptr) {
      residual->resize(M);
    }

    if (jacobian != nullptr) {
      jacobian->resize(M, N);
    }

    if (hessian != nullptr) {
      hessian->resize(N, N);
    }

    if (rhs != nullptr) {
      rhs->resize(N);
    }

    const auto residual_map = residual
                                  ? Eigen::Map<Eigen::Matrix<Scalar, M, 1>>{residual->data(), M}
                                  : Eigen::Map<Eigen::Matrix<Scalar, M, 1>>{nullptr, M};
    const auto jacobian_map = jacobian
                                  ? Eigen::Map<Eigen::Matrix<Scalar, M, N>>{jacobian->data(), M, N}
                                  : Eigen::Map<Eigen::Matrix<Scalar, M, N>>{nullptr, M, N};
    const auto hessian_map = hessian
                                 ? Eigen::Map<Eigen::Matrix<Scalar, N, N>>{hessian->data(), N, N}
                                 : Eigen::Map<Eigen::Matrix<Scalar, N, N>>{nullptr, N, N};
    const auto rhs_map = rhs ? Eigen::Map<Eigen::Matrix<Scalar, N, 1>>{rhs->data(), N}
                             : Eigen::Map<Eigen::Matrix<Scalar, N, 1>>{nullptr, N};

    HessianFuncValuesExtractor<Scalar, FunctorType>::Invoke(
        func, values, keys_to_func, residual_map, jacobian_map, hessian_map, rhs_map);
  };
}

/** Specialize the dispatch mechanism */
template <typename Scalar>
struct HessianDispatcher<false /* is_dynamic */, false /* is_sparse */, true /* is_map */, Scalar> {
  template <typename Functor>
  auto operator()(Functor&& func) {
    return HessianFixedMapDense<Scalar>(std::forward<Functor>(func));
  }
};

// ----------------------------------------------------------------------------
// Factor::Hessian constructor support for fixed size vectors and sparse matrices
// ----------------------------------------------------------------------------

template <typename Scalar, typename Functor>
auto HessianFixedSparse(Functor&& func) {
  // Get matrix types from function signature
  using ResidualVec = typename HessianFuncValuesExtractor<Scalar, Functor>::ResidualVec;
  using JacobianMat = typename HessianFuncValuesExtractor<Scalar, Functor>::JacobianMat;
  using HessianMat = typename HessianFuncValuesExtractor<Scalar, Functor>::HessianMat;
  using RhsVec = typename HessianFuncValuesExtractor<Scalar, Functor>::RhsVec;
  using FunctorType = std::decay_t<Functor>;

  return [func = std::forward<Functor>(func)](const Values<Scalar>& values,
                                              const std::vector<index_entry_t>& keys_to_func,
                                              VectorX<Scalar>* residual, JacobianMat* jacobian,
                                              HessianMat* hessian, VectorX<Scalar>* rhs) {
    ResidualVec residual_fixed;
    RhsVec rhs_fixed;

    HessianFuncValuesExtractor<Scalar, FunctorType>::Invoke(
        func, values, keys_to_func, residual == nullptr ? nullptr : &residual_fixed, jacobian,
        hessian, rhs == nullptr ? nullptr : &rhs_fixed);

    if (residual != nullptr) {
      (*residual) = residual_fixed;
    }

    if (rhs != nullptr) {
      (*rhs) = rhs_fixed;
    }
  };
}

/** Specialize the dispatch mechanism */
template <typename Scalar>
struct HessianDispatcher<false /* is_dynamic */, true /* is_sparse */, false /* is_map */, Scalar> {
  template <typename Functor>
  auto operator()(Functor&& func) {
    return HessianFixedSparse<Scalar>(std::forward<Functor>(func));
  }
};

}  // namespace internal
}  // namespace sym
