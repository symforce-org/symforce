/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#pragma once

#include <ostream>
#include <unordered_set>

#include <Eigen/SparseCore>

#include <lcmtypes/sym/index_entry_t.hpp>
#include <lcmtypes/sym/linearized_dense_factor_t.hpp>
#include <lcmtypes/sym/linearized_dense_factorf_t.hpp>

#include "./templates.h"
#include "./values.h"

namespace sym {

template <typename _S>
struct LinearizedDenseFactorTypeHelper;

template <typename _S>
struct LinearizedSparseFactorTypeHelper;

// NOTE(aaron): Unlike the dense versions of these, we don't have SparseMatrix eigen_lcm types, so
// we just defined these as structs, since we don't need to serialize them anyway
struct linearized_sparse_factor_t {
  Eigen::VectorXd residual;              // b
  Eigen::SparseMatrix<double> jacobian;  // J
  Eigen::SparseMatrix<double> hessian;   // H, JtJ
  Eigen::VectorXd rhs;                   // Jtb

  static constexpr const char* getTypeName() {
    return "linearized_sparse_factor_t";
  }
};

struct linearized_sparse_factorf_t {
  Eigen::VectorXf residual;             // b
  Eigen::SparseMatrix<float> jacobian;  // J
  Eigen::SparseMatrix<float> hessian;   // H, JtJ
  Eigen::VectorXf rhs;                  // Jtb

  static constexpr const char* getTypeName() {
    return "linearized_sparse_factorf_t";
  }
};

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

  // Helpers to expose the correct LCM type for the scalar type
  using LinearizedDenseFactor = typename LinearizedDenseFactorTypeHelper<Scalar>::Type;
  using LinearizedSparseFactor = typename LinearizedSparseFactorTypeHelper<Scalar>::Type;

  // ----------------------------------------------------------------------------------------------
  // Residual function forms
  //
  // These are the functions the Factor operates on. In fact it stores only a HessianFunc (or
  // SparseHessianFunc) as any JacobianFunc specification is used to compute a HessianFunc. However,
  // it is not common for the user to specify these directly as usually functions do not accept a
  // Values, rather the individual input arguments. There are helper constructors to automate the
  // task of extracting and forwarding the proper inputs from the Values using the Keys in the
  // Factor.
  // ----------------------------------------------------------------------------------------------

  template <typename JacobianMatrixType>
  using JacobianFunc = std::function<void(const Values<Scalar>&,              // Input storage
                                          const std::vector<index_entry_t>&,  // Keys
                                          VectorX<Scalar>*,                   // Mx1 residual
                                          JacobianMatrixType*                 // MxN jacobian
                                          )>;

  using DenseJacobianFunc = JacobianFunc<MatrixX<Scalar>>;
  using SparseJacobianFunc = JacobianFunc<Eigen::SparseMatrix<Scalar>>;

  template <typename MatrixType>
  using HessianFunc = std::function<void(const Values<Scalar>&,              // Input storage
                                         const std::vector<index_entry_t>&,  // Keys
                                         VectorX<Scalar>*,                   // Mx1 residual
                                         MatrixType*,                        // MxN jacobian
                                         MatrixType*,                        // NxN hessian
                                         VectorX<Scalar>*                    // Nx1 right-hand side
                                         )>;

  using DenseHessianFunc = HessianFunc<MatrixX<Scalar>>;
  using SparseHessianFunc = HessianFunc<Eigen::SparseMatrix<Scalar>>;

  // ----------------------------------------------------------------------------------------------
  // Constructors
  // ----------------------------------------------------------------------------------------------

  Factor() = default;

  /**
   * Create directly from a (dense/sparse) hessian functor. This is the lowest-level constructor.
   *
   * @param keys_to_func: The set of input arguments, in order, accepted by func.
   * @param keys_to_optimize: The set of input arguments that correspond to the derivative in func.
   *    Must be a subset of keys_to_func. If empty, then all keys_to_func are optimized.
   */
  Factor(DenseHessianFunc hessian_func, const std::vector<Key>& keys_to_func,
         const std::vector<Key>& keys_to_optimize = {});
  Factor(SparseHessianFunc hessian_func, const std::vector<Key>& keys_to_func,
         const std::vector<Key>& keys_to_optimize = {});

  /**
   * Create from a function that computes the (dense/sparse) jacobian. The hessian will be computed
   * using the Gauss Newton approximation:
   *
   *     H   = J.T * J
   *     rhs = J.T * b
   *
   * @param keys_to_func: The set of input arguments, in order, accepted by func.
   * @param keys_to_optimize: The set of input arguments that correspond to the derivative in func.
   *    Must be a subset of keys_to_func. If empty, then all keys_to_func are optimized.
   */
  Factor(const DenseJacobianFunc& jacobian_func, const std::vector<Key>& keys_to_func,
         const std::vector<Key>& keys_to_optimize = {});
  Factor(const SparseJacobianFunc& jacobian_func, const std::vector<Key>& keys_to_func,
         const std::vector<Key>& keys_to_optimize = {});

  /**
   * Create from a function that computes the jacobian. The hessian will be computed using the
   * Gauss Newton approximation:
   *
   *     H   = J.T * J
   *     rhs = J.T * b
   *
   * This version handles a variety of functors that take in individual input arguments
   * rather than a Values object - the last two arguments to `func` should be outputs for the
   * residual and jacobian; arguments before that should be inputs to `func`.
   *
   * If generating this factor from a single Python function, you probably want to use the
   * Factor::Hessian constructor instead (it'll likely result in faster linearization).  If you
   * really want to generate a factor and use this constructor, `func` can be generated easily by
   * creating a Codegen object from a Python function which returns the residual, then calling
   * with_linearization with linearization_mode=STACKED_JACOBIAN
   *
   * See `symforce_factor_test.cc` for many examples.
   *
   * @param keys_to_func: The set of input arguments, in order, accepted by func.
   * @param keys_to_optimize: The set of input arguments that correspond to the derivative in func.
   *    Must be a subset of keys_to_func. If empty, then all keys_to_func are optimized.
   */
  template <typename Functor>
  static Factor Jacobian(Functor&& func, const std::vector<Key>& keys_to_func,
                         const std::vector<Key>& keys_to_optimize = {});

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
   * returns the residual, then calling with_linearization with
   * linearization_mode=FULL_LINEARIZATION (the default)
   *
   * See `symforce_factor_test.cc` for many examples.
   *
   * @param keys_to_func: The set of input arguments, in order, accepted by func.
   * @param keys_to_optimize: The set of input arguments that correspond to the derivative in func.
   *    Must be a subset of keys_to_func. If empty, then all keys_to_func are optimized.
   */
  template <typename Functor>
  static Factor Hessian(Functor&& func, const std::vector<Key>& keys_to_func,
                        const std::vector<Key>& keys_to_optimize = {});

  // ----------------------------------------------------------------------------------------------
  // Linearization
  // ----------------------------------------------------------------------------------------------

  /**
   * Evaluate the factor at the given linearization point and output just the
   * numerical values of the residual.
   *
   * @param maybe_index_entry_cache: Optional.  If provided, should be the index entries for each of
   *    the inputs to the factor in the given Values.  For repeated linearization, caching this
   *    prevents repeated hash lookups.  Can be computed as
   *    `values.CreateIndex(factor.AllKeys()).entries`.
   */
  void Linearize(const Values<Scalar>& values, VectorX<Scalar>* residual,
                 const std::vector<index_entry_t>* maybe_index_entry_cache = nullptr) const;

  /**
   * Evaluate the factor at the given linearization point and output just the
   * numerical values of the residual and jacobian.
   *
   * This overload can only be called if IsSparse() is false; otherwise, it will throw
   *
   * @param maybe_index_entry_cache: Optional.  If provided, should be the index entries for each of
   *    the inputs to the factor in the given Values.  For repeated linearization, caching this
   *    prevents repeated hash lookups.  Can be computed as
   *    `values.CreateIndex(factor.AllKeys()).entries`.
   */
  void Linearize(const Values<Scalar>& values, VectorX<Scalar>* residual, MatrixX<Scalar>* jacobian,
                 const std::vector<index_entry_t>* maybe_index_entry_cache = nullptr) const;

  /**
   * Evaluate the factor at the given linearization point and output just the
   * numerical values of the residual and jacobian.
   *
   * This overload can only be called if IsSparse() is true; otherwise, it will throw
   *
   * @param maybe_index_entry_cache: Optional.  If provided, should be the index entries for each of
   *    the inputs to the factor in the given Values.  For repeated linearization, caching this
   *    prevents repeated hash lookups.  Can be computed as
   *    `values.CreateIndex(factor.AllKeys()).entries`.
   */
  void Linearize(const Values<Scalar>& values, VectorX<Scalar>* residual,
                 Eigen::SparseMatrix<Scalar>* jacobian,
                 const std::vector<index_entry_t>* maybe_index_entry_cache = nullptr) const;

  /**
   * Evaluate the factor at the given linearization point and output a LinearizedDenseFactor that
   * contains the numerical values of the residual, jacobian, hessian, and right-hand-side.
   *
   * This overload can only be called if IsSparse() is false; otherwise, it will throw
   *
   * @param maybe_index_entry_cache: Optional.  If provided, should be the index entries for each of
   *    the inputs to the factor in the given Values.  For repeated linearization, caching this
   *    prevents repeated hash lookups.  Can be computed as
   *    `values.CreateIndex(factor.AllKeys()).entries`.
   */
  void Linearize(const Values<Scalar>& values, LinearizedDenseFactor& linearized_factor,
                 const std::vector<index_entry_t>* maybe_index_entry_cache = nullptr) const;

  /**
   * Evaluate the factor at the given linearization point and output a LinearizedDenseFactor that
   * contains the numerical values of the residual, jacobian, hessian, and right-hand-side.
   *
   * This overload can only be called if IsSparse() is false; otherwise, it will throw
   *
   * @param maybe_index_entry_cache: Optional.  If provided, should be the index entries for each of
   *    the inputs to the factor in the given Values.  For repeated linearization, caching this
   *    prevents repeated hash lookups.  Can be computed as
   *    `values.CreateIndex(factor.AllKeys()).entries`.
   */
  LinearizedDenseFactor Linearize(
      const Values<Scalar>& values,
      const std::vector<index_entry_t>* maybe_index_entry_cache = nullptr) const;

  /**
   * Evaluate the factor at the given linearization point and output a LinearizedDenseFactor that
   * contains the numerical values of the residual, jacobian, hessian, and right-hand-side.
   *
   * This overload can only be called if IsSparse is true; otherwise, it will throw
   *
   * @param maybe_index_entry_cache: Optional.  If provided, should be the index entries for each of
   *    the inputs to the factor in the given Values.  For repeated linearization, caching this
   *    prevents repeated hash lookups.  Can be computed as
   *    `values.CreateIndex(factor.AllKeys()).entries`.
   */
  void Linearize(const Values<Scalar>& values, LinearizedSparseFactor& linearized_factor,
                 const std::vector<index_entry_t>* maybe_index_entry_cache = nullptr) const;

  // ----------------------------------------------------------------------------------------------
  // Helpers
  // ----------------------------------------------------------------------------------------------

  /**
   * Does this factor use a sparse jacobian/hessian matrix?
   */
  bool IsSparse() const {
    return static_cast<bool>(sparse_hessian_func_);  // operator bool
  }

  /**
   * Get the optimized keys for this factor
   */
  const std::vector<Key>& OptimizedKeys() const;

  /**
   * Get all keys required to evaluate this factor
   */
  const std::vector<Key>& AllKeys() const;

 private:
  DenseHessianFunc hessian_func_;
  SparseHessianFunc sparse_hessian_func_;

  // Keys to be optimized in this factor, which must match the column order of the jacobian.
  std::vector<Key> keys_to_optimize_;

  // All keys required to evaluate the factor
  std::vector<Key> keys_;
};

// Shorthand instantiations
using Factord = Factor<double>;
using Factorf = Factor<float>;

template <typename Scalar>
std::ostream& operator<<(std::ostream& os, const sym::Factor<Scalar>& factor);

// TODO(hayk): Why doesn't this work instead of splitting out the types?
// template <typename Scalar>
// std::ostream& operator<<(std::ostream& os,
//                          const typename sym::Factor<Scalar>::LinearizedDenseFactor& factor);
std::ostream& operator<<(std::ostream& os, const sym::linearized_dense_factor_t& factor);
std::ostream& operator<<(std::ostream& os, const sym::linearized_dense_factorf_t& factor);
std::ostream& operator<<(std::ostream& os, const sym::linearized_sparse_factor_t& factor);
std::ostream& operator<<(std::ostream& os, const sym::linearized_sparse_factorf_t& factor);

/**
 * Compute the combined set of keys to optimize from the given factors. Order using the given
 * comparison function.
 */
template <typename Scalar, typename Compare>
std::vector<Key> ComputeKeysToOptimize(const std::vector<Factor<Scalar>>& factors,
                                       Compare key_compare) {
  // Some thoughts on efficiency at
  // https://stackoverflow.com/questions/1041620/whats-the-most-efficient-way-to-erase-duplicates-and-sort-a-vector

  // Aggregate uniques
  std::unordered_set<Key> key_set;
  for (const Factor<Scalar>& factor : factors) {
    key_set.insert(factor.OptimizedKeys().begin(), factor.OptimizedKeys().end());
  }

  // Copy to vector
  std::vector<Key> keys;
  keys.insert(keys.end(), key_set.begin(), key_set.end());

  // Order
  std::sort(keys.begin(), keys.end(), key_compare);

  return keys;
}

// If no comparator is specified, use sym::Key::LexicalLessThan.
template <typename Scalar>
std::vector<Key> ComputeKeysToOptimize(const std::vector<Factor<Scalar>>& factors) {
  return ComputeKeysToOptimize(factors, &sym::Key::LexicalLessThan);
}

}  // namespace sym

// Template method implementations
#include "./factor.tcc"

// Explicit instantiation declarations
extern template class sym::Factor<double>;
extern template class sym::Factor<float>;
