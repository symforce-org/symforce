#include "./assert.h"

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
  static_assert(ResidualVec::RowsAtCompileTime == M, "Residual and jacobian rows aren't equal.");

  constexpr bool is_dynamic = (M == Eigen::Dynamic) && (N == Eigen::Dynamic);
  constexpr bool is_fixed = (M > 0) && (N > 0);
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
  template <int... S>
  inline static void InvokeAtRange(Functor func, const sym::Values<Scalar>& values,
                                   const std::vector<sym::Key>& keys,
                                   sym::VectorX<Scalar>* residual, sym::MatrixX<Scalar>* jacobian,
                                   Sequence<S...>) {
    func(GetValue<S>(values, keys)..., residual, jacobian);
  }

  inline static void Invoke(Functor func, const sym::Values<Scalar>& values,
                            const std::vector<sym::Key>& keys, sym::VectorX<Scalar>* residual,
                            sym::MatrixX<Scalar>* jacobian) {
    InvokeAtRange(func, values, keys, residual, jacobian,
                  typename RangeGenerator<Traits::num_arguments - 2>::Range());
  }
};

/** Wraps a function that takes individual input args into one that takes a Values. */
template <typename Scalar, typename Functor>
Factor<Scalar> JacobianDynamic(Functor func, const std::vector<Key>& keys_to_func,
                               const std::vector<Key>& keys_to_optimize) {
  return Factor<Scalar>(
      [func, keys_to_func, keys_to_optimize](const Values<Scalar>& values,
                                             VectorX<Scalar>* residual, MatrixX<Scalar>* jacobian,
                                             MatrixX<Scalar>* hessian, VectorX<Scalar>* rhs) {
        SYM_ASSERT(residual != nullptr);
        JacobianFuncValuesExtractor<Scalar, Functor>::Invoke(func, values, keys_to_func, residual,
                                                             jacobian);
        SYM_ASSERT(jacobian == nullptr || residual->rows() == jacobian->rows());

        // Compute the lower triangle of the hessian if needed
        if (hessian != nullptr) {
          SYM_ASSERT(jacobian);
          hessian->resize(jacobian->cols(), jacobian->cols());
          hessian->template triangularView<Eigen::Lower>().setZero();
          hessian->template selfadjointView<Eigen::Lower>().rankUpdate(jacobian->transpose());
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

/**
 * Helper to return wrap a residual function with fixed size matrices for the
 * residual and jacobian into a function with them as dynamic sized matrices.
 */
template <typename Scalar, typename Functor>
struct JacobianFuncFixedSizeWrapper {
  using Traits = sym::function_traits<Functor>;
  template <int Index>
  using ArgType = typename Traits::template arg<Index>::type;

  /** Call the user function with fixed size matrices then copy into dynamic size. */
  template <int M, int N, int... S>
  static auto Wrap(Functor func, Sequence<S...>) {
    return [func](ArgType<S>... args, VectorX<Scalar>* residual, MatrixX<Scalar>* jacobian) {
      Eigen::Matrix<Scalar, M, 1> residual_fixed;

      if (jacobian != nullptr) {
        // jacobian is requested
        Eigen::Matrix<Scalar, M, N> jacobian_fixed;
        func(args..., &residual_fixed, &jacobian_fixed);
        (*jacobian) = jacobian_fixed;
      } else {
        // jacobian not requested
        func(args..., &residual_fixed, nullptr);
      }

      (*residual) = residual_fixed;
    };
  }

  static auto Invoke(Functor func) {
    // Get matrix types from function signature
    using ResidualVec = typename std::remove_pointer<ArgType<Traits::num_arguments - 2>>::type;
    using JacobianMat = typename std::remove_pointer<ArgType<Traits::num_arguments - 1>>::type;

    // Get and sanity check dimensions
    constexpr int M = JacobianMat::RowsAtCompileTime;
    constexpr int N = JacobianMat::ColsAtCompileTime;
    static_assert(M > 0, "Zero dimension matrix.");
    static_assert(N > 0, "Zero dimension matrix.");
    static_assert(ResidualVec::RowsAtCompileTime == M, "Inconsistent sizes.");
    static_assert(ResidualVec::ColsAtCompileTime == 1, "Inconsistent sizes.");

    return Wrap<M, N>(func, typename RangeGenerator<Traits::num_arguments - 2>::Range());
  }
};

template <typename Scalar, typename Functor>
Factor<Scalar> JacobianFixed(Functor func, const std::vector<Key>& keys_to_func,
                             const std::vector<Key>& keys_to_optimize) {
  return JacobianDynamic<Scalar>(JacobianFuncFixedSizeWrapper<Scalar, Functor>::Invoke(func),
                                 keys_to_func, keys_to_optimize);
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
