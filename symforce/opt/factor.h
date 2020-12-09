#pragma once

#include <ostream>

#include <lcmtypes/sym/linearized_factor_t.hpp>
#include <lcmtypes/sym/linearized_factorf_t.hpp>

#include "./templates.h"
#include "./values.h"

namespace sym {

/**
 * A residual term for optimization.
 *
 * Created from a function and a set of Keys that act as inputs. Given a Values as an evaluation
 * point, generates a linear approximation to the residual function. Contains templated helper
 * constructors to simplify creation.
 */
template <typename ScalarType>
class Factor {
 public:
  using Scalar = ScalarType;

  // Helper to expose the correct LCM type for the scalar type
  template <typename _S, bool _D = true>
  struct LinearizedFactorTypeHelper {};
  using LinearizedFactor = typename LinearizedFactorTypeHelper<Scalar>::Type;

  // ----------------------------------------------------------------------------------------------
  // Residual function forms
  //
  // These are the functions the Factor operates on. In fact it stores only a HessianFunc as any
  // JacobianFunc specification is used to compute a HessianFunc. However, it is not common for
  // the user to specify these directly as usually functions do not accept a Values, rather the
  // individual input arguments. There are helper constructors to automate the task of extracting
  // and forwarding the proper inputs from the Values using the Keys in the Factor.
  // ----------------------------------------------------------------------------------------------

  using JacobianFunc = std::function<void(const Values<Scalar>&,  // Input storage
                                          VectorX<Scalar>*,       // Mx1 residual
                                          MatrixX<Scalar>*        // MxN jacobian
                                          )>;

  using HessianFunc = std::function<void(const Values<Scalar>&,  // Input storage
                                         VectorX<Scalar>*,       // Mx1 residual
                                         MatrixX<Scalar>*,       // MxN jacobian
                                         MatrixX<Scalar>*,       // NxN hessian
                                         VectorX<Scalar>*        // Nx1 right-hand side
                                         )>;

  // ----------------------------------------------------------------------------------------------
  // Constructors
  // ----------------------------------------------------------------------------------------------

  Factor() {}

  /**
   * Create directly from a hessian functor. This is the lowest-level constructor.
   */
  template <typename HessianFunctor>
  Factor(HessianFunctor&& hessian_func, const std::vector<Key>& keys);

  /**
   * Create from a function that computes the jacobian. The hessian will be computed using the
   * Gauss Newton approximation:
   *    H   = J.T * J
   *    rhs = J.T * b
   */
  static Factor Jacobian(const JacobianFunc& jacobian_func, const std::vector<Key>& keys);

  /**
   * Create from a function that computes the jacobian. The hessian will be computed using the
   * Gauss Newton approximation:
   *    H   = J.T * J
   *    rhs = J.T * b
   *
   * This verion handles a variety of functors that take in individual input arguments
   * rather than a Values object - the last two arguments to `func` should be outputs for the
   * residual and jacobian; arguments before that should be inputs to `func`.
   *
   * If generating this factor from a single Python function, you probably want to use the
   * Factor::Hessian constructor instead (it'll likely result in faster linearization).  If you
   * really want to generate a factor and use this constructor, `func` can be generated easily by
   * creating a Codegen object from a Python function which returns the residual, then calling
   * create_with_jacobians with jacobian_generation_mode=STACKED
   *
   * See `sym_factor_test.cc` for many examples.
   */
  template <typename Functor>
  static Factor Jacobian(Functor func, const std::vector<Key>& keys);

  /**
   * Same as the above, but allows extra constant keys into the function which are not optimized.
   * For example, to pass through epsilon.
   *
   * Args:
   *   keys_to_func: The set of input arguments, in order, accepted by func.
   *   keys_to_optimize: The set of input arguments that correspond to the derivative in func. Must
   *                     be a subset of keys_to_func.
   */
  template <typename Functor>
  static Factor Jacobian(Functor func, const std::vector<Key>& keys_to_func,
                         const std::vector<Key>& keys_to_optimize);

  /**
   * Create from a functor that computes the full linearization, but takes in individual input
   * arguments rather than a Values object.  The last four arguments to `func` should be outputs for
   * the residual, jacobian, hessian, and rhs; arguments before that should be inputs to `func`.
   *
   * This should be used in cases where computing J^T J using a matrix multiplication is slower than
   * evaluating J^T J symbolically; for instance, if the jacobian is sparse or J^T J has structure
   * so that CSE is very effective.
   *
   * `func` can be generated easily by creating a Codegen object from a Python function which
   * returns the residual, then calling create_with_jacobians with
   * jacobian_generation_mode=LINEARIZATION
   *
   * See `sym_factor_test.cc` for many examples.
   */
  template <typename Functor>
  static Factor Hessian(Functor func, const std::vector<Key>& keys);

  /**
   * Same as the above, but allows extra constant keys into the function which are not optimized.
   * For example, to pass through epsilon.
   *
   * Args:
   *   keys_to_func: The set of input arguments, in order, accepted by func.
   *   keys_to_optimize: The set of input arguments that correspond to the derivative in func. Must
   *                     be a subset of keys_to_func.
   */
  template <typename Functor>
  static Factor Hessian(Functor func, const std::vector<Key>& keys_to_func,
                        const std::vector<Key>& keys_to_optimize);

  // ----------------------------------------------------------------------------------------------
  // Linearization
  // ----------------------------------------------------------------------------------------------

  /**
   * Evaluate the factor at the given linearization point and output just the
   * numerical values of the residual and jacobian.
   */
  void Linearize(const Values<Scalar>& values, VectorX<ScalarType>* residual,
                 MatrixX<ScalarType>* jacobian = nullptr) const;

  /**
   * Evaluate the factor at the given linearization point and output a LinearizedFactor that
   * contains the numerical values of the residual, jacobian, hessian, and right-hand-side.
   */
  void Linearize(const Values<Scalar>& values, LinearizedFactor* linearized_factor) const;
  LinearizedFactor Linearize(const Values<Scalar>& values) const;

  // ----------------------------------------------------------------------------------------------
  // Helpers
  // ----------------------------------------------------------------------------------------------
  const std::vector<Key>& Keys() const;

 private:
  HessianFunc hessian_func_;

  // Keys to be optimized in this factor, which must match the column order of the jacobian.
  std::vector<Key> keys_;
};

// Shorthand instantiations
using Factord = Factor<double>;
using Factorf = Factor<float>;

}  // namespace sym

template <typename Scalar>
std::ostream& operator<<(std::ostream& os, const sym::Factor<Scalar>& factor);

// TODO(hayk): Why doesn't this work instead of splitting out the types?
// template <typename Scalar>
// std::ostream& operator<<(std::ostream& os,
//                          const typename sym::Factor<Scalar>::LinearizedFactor& factor);
std::ostream& operator<<(std::ostream& os, const sym::linearized_factor_t& factor);
std::ostream& operator<<(std::ostream& os, const sym::linearized_factorf_t& factor);

// Template method implementations
#include "./factor.tcc"
