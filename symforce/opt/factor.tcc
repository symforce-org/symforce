#include "./assert.h"
#include "./factor.h"

namespace sym {

template <typename Scalar>
template <typename HessianFunctor>
Factor<Scalar>::Factor(HessianFunctor&& hessian_func, const std::vector<Key>& keys)
    : hessian_func_(std::move(hessian_func)), keys_(keys) {}

// ------------------------------------------------------------------------------------------------
// Factor::Jacobian constructor dispatcher
//
// This constructor can accept multiple forms of functions. It introspects the function to
// understand its input/output arguments, performs a series of static assertions, and dispatches
// to the correct helper to be processed.
// ------------------------------------------------------------------------------------------------

template <typename Scalar>
template <typename Functor>
Factor<Scalar> Factor<Scalar>::Jacobian(Functor func, const std::vector<Key>& keys) {
  return Jacobian(func, keys, keys);
}

// Struct to allow partial specialization based on whether matrices are dynamic
template <bool IsDynamic, typename Scalar>
struct JacobianDispatcher {
  template <typename Functor>
  Factor<Scalar> operator()(Functor func, const std::vector<Key>& keys_to_func,
                            const std::vector<Key>& keys_to_optimize);
};

template <typename Scalar>
template <typename Functor>
Factor<Scalar> Factor<Scalar>::Jacobian(Functor func, const std::vector<Key>& keys_to_func,
                                        const std::vector<Key>& keys_to_optimize) {
  using Traits = function_traits<Functor>;

  SYM_ASSERT(keys_to_func.size() >= keys_to_optimize.size());
  SYM_ASSERT(Traits::num_arguments == keys_to_func.size() + 2);

  // Get matrix types from function signature
  using ResidualVec = typename std::remove_pointer<
      typename Traits::template arg<Traits::num_arguments - 2>::type>::type;
  using JacobianMat = typename std::remove_pointer<
      typename Traits::template arg<Traits::num_arguments - 1>::type>::type;

  // Get dimensions
  constexpr int M = JacobianMat::RowsAtCompileTime;
  constexpr int N = JacobianMat::ColsAtCompileTime;
  static_assert(ResidualVec::RowsAtCompileTime == M, "Inconsistent sizes.");
  static_assert(ResidualVec::ColsAtCompileTime == 1, "Inconsistent sizes.");

  constexpr bool is_dynamic = (M == Eigen::Dynamic) && (N == Eigen::Dynamic);
  constexpr bool is_fixed = (M != Eigen::Dynamic) && (N != Eigen::Dynamic);
  static_assert((is_dynamic || is_fixed), "Matrices cannot be mixed fixed and dynamic.");

  // Dispatch to either the dynamic size or fixed size implementations
  return JacobianDispatcher<is_dynamic, Scalar>{}(func, keys_to_func, keys_to_optimize);
}

// ----------------------------------------------------------------------------
// Factor::Jacobian constructor support for dynamic matrices
// ----------------------------------------------------------------------------

/**
 * Helper for automatically extracting keys from values.
 */
template <typename Scalar, typename Functor>
struct JacobianFuncValuesExtractor {
  using Traits = function_traits<Functor>;

  template <int Index>
  using ArgType = typename Traits::template arg<Index>::base_type;

  /** Pull out the arg given by Index from the Values. */
  template <int Index>
  inline static ArgType<Index> GetValue(const sym::Values<Scalar>& values,
                                        const std::vector<sym::Key>& keys) {
    return values.template At<ArgType<Index>>(keys[Index]);
  }

  /** Invokes the user function with the proper input args extracted from the Values. */
  template <int... S, int M, int N>
  inline static void InvokeAtRange(Functor func, const sym::Values<Scalar>& values,
                                   const std::vector<sym::Key>& keys,
                                   Eigen::Matrix<Scalar, M, 1>* residual,
                                   Eigen::Matrix<Scalar, M, N>* jacobian, Sequence<S...>) {
    func(GetValue<S>(values, keys)..., residual, jacobian);
  }

  template <int M, int N>
  inline static void Invoke(Functor func, const sym::Values<Scalar>& values,
                            const std::vector<sym::Key>& keys,
                            Eigen::Matrix<Scalar, M, 1>* residual,
                            Eigen::Matrix<Scalar, M, N>* jacobian) {
    InvokeAtRange(func, values, keys, residual, jacobian,
                  typename RangeGenerator<Traits::num_arguments - 2>::Range());
  }
};

/** Wraps a function that takes individual input args into one that takes a Values. */
template <typename Scalar, typename Functor>
Factor<Scalar> JacobianDynamic(Functor func, const std::vector<Key>& keys_to_func,
                               const std::vector<Key>& keys_to_optimize) {
  return Factor<Scalar>(
      [func, keys_to_func](const Values<Scalar>& values, VectorX<Scalar>* residual,
                           MatrixX<Scalar>* jacobian, MatrixX<Scalar>* hessian,
                           VectorX<Scalar>* rhs) {
        SYM_ASSERT(residual != nullptr);
        JacobianFuncValuesExtractor<Scalar, Functor>::Invoke(func, values, keys_to_func, residual,
                                                             jacobian);
        SYM_ASSERT(jacobian == nullptr || residual->rows() == jacobian->rows());

        // Compute the lower triangle of the hessian if needed
        if (hessian != nullptr) {
          SYM_ASSERT(jacobian);
          hessian->resize(jacobian->cols(), jacobian->cols());

          // NOTE(aaron):  Testing seemed to show that this matrix multiply is faster than
          // rankUpdate, at least for smallish jacobians that are still large enough for Eigen to
          // not inline the whole multiplication.  More investigation would be necessary to figure
          // out why, and if it's expected to be faster for large jacobians
          hessian->template triangularView<Eigen::Lower>() =
              (jacobian->transpose() * (*jacobian)).template triangularView<Eigen::Lower>();
        }

        // Compute RHS if needed
        if (rhs != nullptr) {
          SYM_ASSERT(jacobian);
          (*rhs) = jacobian->transpose() * (*residual);
        }
      },
      keys_to_optimize);
}

/** Specialize the dispatch mechanism */
template <typename Scalar>
struct JacobianDispatcher<true /* is_dynamic */, Scalar> {
  template <typename Functor>
  Factor<Scalar> operator()(Functor func, const std::vector<Key>& keys_to_func,
                            const std::vector<Key>& keys_to_optimize) {
    return JacobianDynamic<Scalar>(func, keys_to_func, keys_to_optimize);
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
Factor<Scalar> JacobianFixed(Functor func, const std::vector<Key>& keys_to_func,
                             const std::vector<Key>& keys_to_optimize) {
  using Traits = function_traits<Functor>;

  // Get matrix types from function signature
  using JacobianMat = typename std::remove_pointer<
      typename Traits::template arg<Traits::num_arguments - 1>::type>::type;

  // Get dimensions (these have already been sanity checked in Factor::Jacobian)
  constexpr int M = JacobianMat::RowsAtCompileTime;
  constexpr int N = JacobianMat::ColsAtCompileTime;

  return Factor<Scalar>(
      [func, keys_to_func](const Values<Scalar>& values, VectorX<Scalar>* residual,
                           MatrixX<Scalar>* jacobian, MatrixX<Scalar>* hessian,
                           VectorX<Scalar>* rhs) {
        SYM_ASSERT(residual != nullptr);
        Eigen::Matrix<Scalar, M, 1> residual_fixed;

        if (jacobian != nullptr) {
          // jacobian is requested
          Eigen::Matrix<Scalar, M, N> jacobian_fixed;
          JacobianFuncValuesExtractor<Scalar, Functor>::Invoke(func, values, keys_to_func,
                                                               &residual_fixed, &jacobian_fixed);
          (*jacobian) = jacobian_fixed;

          // Compute the lower triangle of the hessian if needed
          if (hessian != nullptr) {
            hessian->resize(N, N);

            // NOTE(aaron):  Testing seemed to show that this matrix multiply is faster than
            // rankUpdate, at least for smallish jacobians that are still large enough for Eigen to
            // not inline the whole multiplication.  More investigation would be necessary to figure
            // out why, and if it's expected to be faster for large jacobians
            hessian->template triangularView<Eigen::Lower>() =
                (jacobian_fixed.transpose() * jacobian_fixed)
                    .template triangularView<Eigen::Lower>();
          }

          // Compute RHS if needed
          if (rhs != nullptr) {
            (*rhs) = jacobian_fixed.transpose() * residual_fixed;
          }
        } else {
          // jacobian not requested
          Eigen::Matrix<Scalar, M, N>* const jacobian_invoke_arg = nullptr;
          JacobianFuncValuesExtractor<Scalar, Functor>::Invoke(
              func, values, keys_to_func, &residual_fixed, jacobian_invoke_arg);

          // Check that the hessian and rhs weren't requested without the jacobian
          SYM_ASSERT(hessian == nullptr);
          SYM_ASSERT(rhs == nullptr);
        }

        (*residual) = residual_fixed;
      },
      keys_to_optimize);
}

/** Specialize the dispatch mechanism */
template <typename Scalar>
struct JacobianDispatcher<false /* is_dynamic */, Scalar> {
  template <typename Functor>
  Factor<Scalar> operator()(Functor func, const std::vector<Key>& keys_to_func,
                            const std::vector<Key>& keys_to_optimize) {
    return JacobianFixed<Scalar>(func, keys_to_func, keys_to_optimize);
  }
};

// ------------------------------------------------------------------------------------------------
// Factor::Hessian constructor dispatcher
//
// This constructor can accept multiple forms of functions. It introspects the function to
// understand its input/output arguments, performs a series of static assertions, and dispatches
// to the correct helper to be processed.
// ------------------------------------------------------------------------------------------------

template <typename Scalar>
template <typename Functor>
Factor<Scalar> Factor<Scalar>::Hessian(Functor func, const std::vector<Key>& keys) {
  return Hessian(func, keys, keys);
}

// Struct to allow partial specialization based on whether matrices are dynamic
template <bool IsDynamic, typename Scalar>
struct HessianDispatcher {
  template <typename Functor>
  Factor<Scalar> operator()(Functor func, const std::vector<Key>& keys_to_func,
                            const std::vector<Key>& keys_to_optimize);
};

template <typename Scalar>
template <typename Functor>
Factor<Scalar> Factor<Scalar>::Hessian(Functor func, const std::vector<Key>& keys_to_func,
                                       const std::vector<Key>& keys_to_optimize) {
  using Traits = function_traits<Functor>;

  SYM_ASSERT(keys_to_func.size() >= keys_to_optimize.size());
  SYM_ASSERT(Traits::num_arguments == keys_to_func.size() + 4);

  // Get matrix types from function signature
  using ResidualVec = typename std::remove_pointer<
      typename Traits::template arg<Traits::num_arguments - 4>::type>::type;
  using JacobianMat = typename std::remove_pointer<
      typename Traits::template arg<Traits::num_arguments - 3>::type>::type;
  using HessianMat = typename std::remove_pointer<
      typename Traits::template arg<Traits::num_arguments - 2>::type>::type;
  using RhsVec = typename std::remove_pointer<
      typename Traits::template arg<Traits::num_arguments - 1>::type>::type;

  // Get dimensions
  constexpr int M = JacobianMat::RowsAtCompileTime;
  constexpr int N = JacobianMat::ColsAtCompileTime;
  static_assert(ResidualVec::RowsAtCompileTime == M, "Inconsistent sizes.");
  static_assert(ResidualVec::ColsAtCompileTime == 1, "Inconsistent sizes.");
  static_assert(HessianMat::RowsAtCompileTime == N, "Inconsistent sizes.");
  static_assert(HessianMat::ColsAtCompileTime == N, "Inconsistent sizes.");
  static_assert(RhsVec::RowsAtCompileTime == N, "Inconsistent sizes.");
  static_assert(RhsVec::ColsAtCompileTime == 1, "Inconsistent sizes.");

  constexpr bool is_dynamic = (M == Eigen::Dynamic) && (N == Eigen::Dynamic);
  constexpr bool is_fixed = (M != Eigen::Dynamic) && (N != Eigen::Dynamic);
  static_assert((is_dynamic || is_fixed), "Matrices cannot be mixed fixed and dynamic.");

  // Dispatch to either the dynamic size or fixed size implementations
  return HessianDispatcher<is_dynamic, Scalar>{}(func, keys_to_func, keys_to_optimize);
}

/**
 * Helper for automatically extracting keys from values.
 */
template <typename Scalar, typename Functor>
struct HessianFuncValuesExtractor {
  using Traits = function_traits<Functor>;

  template <int Index>
  using ArgType = typename Traits::template arg<Index>::base_type;

  /** Pull out the arg given by Index from the Values. */
  template <int Index>
  inline static ArgType<Index> GetValue(const sym::Values<Scalar>& values,
                                        const std::vector<sym::Key>& keys) {
    return values.template At<ArgType<Index>>(keys[Index]);
  }

  /** Invokes the user function with the proper input args extracted from the Values. */
  template <int... S, int M, int N>
  inline static void InvokeAtRange(Functor func, const sym::Values<Scalar>& values,
                                   const std::vector<sym::Key>& keys,
                                   Eigen::Matrix<Scalar, M, 1>* residual,
                                   Eigen::Matrix<Scalar, M, N>* jacobian,
                                   Eigen::Matrix<Scalar, N, N>* hessian,
                                   Eigen::Matrix<Scalar, N, 1>* rhs, Sequence<S...>) {
    func(GetValue<S>(values, keys)..., residual, jacobian, hessian, rhs);
  }

  template <int M, int N>
  inline static void Invoke(Functor func, const sym::Values<Scalar>& values,
                            const std::vector<sym::Key>& keys,
                            Eigen::Matrix<Scalar, M, 1>* residual,
                            Eigen::Matrix<Scalar, M, N>* jacobian,
                            Eigen::Matrix<Scalar, N, N>* hessian,
                            Eigen::Matrix<Scalar, N, 1>* rhs) {
    InvokeAtRange(func, values, keys, residual, jacobian, hessian, rhs,
                  typename RangeGenerator<Traits::num_arguments - 4>::Range());
  }
};

// ----------------------------------------------------------------------------
// Factor::Hessian constructor support for dynamic matrices
// ----------------------------------------------------------------------------

template <typename Scalar, typename Functor>
Factor<Scalar> HessianDynamic(Functor func, const std::vector<Key>& keys_to_func,
                              const std::vector<Key>& keys_to_optimize) {
  return Factor<Scalar>(
      [func, keys_to_func](const Values<Scalar>& values, VectorX<Scalar>* residual,
                           MatrixX<Scalar>* jacobian, MatrixX<Scalar>* hessian,
                           VectorX<Scalar>* rhs) {
        SYM_ASSERT(residual != nullptr);

        HessianFuncValuesExtractor<Scalar, Functor>::Invoke(func, values, keys_to_func, residual,
                                                            jacobian, hessian, rhs);
      },
      keys_to_optimize);
}

/** Specialize the dispatch mechanism */
template <typename Scalar>
struct HessianDispatcher<true /* is_dynamic */, Scalar> {
  template <typename Functor>
  Factor<Scalar> operator()(Functor func, const std::vector<Key>& keys_to_func,
                            const std::vector<Key>& keys_to_optimize) {
    return HessianDynamic<Scalar>(func, keys_to_func, keys_to_optimize);
  }
};

// ----------------------------------------------------------------------------
// Factor::Hessian constructor support for fixed size matrices
// ----------------------------------------------------------------------------

template <typename Scalar, typename Functor>
Factor<Scalar> HessianFixed(Functor func, const std::vector<Key>& keys_to_func,
                            const std::vector<Key>& keys_to_optimize) {
  using Traits = function_traits<Functor>;

  // Get matrix types from function signature
  using JacobianMat = typename std::remove_pointer<
      typename Traits::template arg<Traits::num_arguments - 3>::type>::type;

  // Get dimensions (these have already been sanity checked in Factor::Hessian)
  constexpr int M = JacobianMat::RowsAtCompileTime;
  constexpr int N = JacobianMat::ColsAtCompileTime;

  return Factor<Scalar>(
      [func, keys_to_func](const Values<Scalar>& values, VectorX<Scalar>* residual,
                           MatrixX<Scalar>* jacobian, MatrixX<Scalar>* hessian,
                           VectorX<Scalar>* rhs) {
        SYM_ASSERT(residual != nullptr);

        Eigen::Matrix<Scalar, M, 1> residual_fixed;
        Eigen::Matrix<Scalar, M, N> jacobian_fixed;
        Eigen::Matrix<Scalar, N, N> hessian_fixed;
        Eigen::Matrix<Scalar, N, 1> rhs_fixed;

        HessianFuncValuesExtractor<Scalar, Functor>::Invoke(
            func, values, keys_to_func, &residual_fixed,
            jacobian == nullptr ? nullptr : &jacobian_fixed,
            hessian == nullptr ? nullptr : &hessian_fixed, rhs == nullptr ? nullptr : &rhs_fixed);

        (*residual) = residual_fixed;

        if (jacobian != nullptr) {
          (*jacobian) = jacobian_fixed;
        }

        if (hessian != nullptr) {
          (*hessian) = hessian_fixed;
        }

        if (rhs != nullptr) {
          (*rhs) = rhs_fixed;
        }
      },
      keys_to_optimize);
}

/** Specialize the dispatch mechanism */
template <typename Scalar>
struct HessianDispatcher<false /* is_dynamic */, Scalar> {
  template <typename Functor>
  Factor<Scalar> operator()(Functor func, const std::vector<Key>& keys_to_func,
                            const std::vector<Key>& keys_to_optimize) {
    return HessianFixed<Scalar>(func, keys_to_func, keys_to_optimize);
  }
};

// ----------------------------------------------------------------------------
// LCM type aliases
// ----------------------------------------------------------------------------

template <typename Scalar>
template <bool _D>
struct Factor<Scalar>::LinearizedFactorTypeHelper<double, _D> {
  using Type = linearized_factor_t;
};

template <typename Scalar>
template <bool _D>
struct Factor<Scalar>::LinearizedFactorTypeHelper<float, _D> {
  using Type = linearized_factorf_t;
};

}  // namespace sym
