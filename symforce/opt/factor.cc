/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#include "./factor.h"

#include <cassert>

#include <fmt/format.h>
#include <fmt/ostream.h>
#include <fmt/ranges.h>

#include "./assert.h"
#include "./internal/factor_utils.h"

namespace sym {

template <typename Scalar, typename Matrix>
Factor<Scalar> FactorFromJacobianFunc(
    const typename Factor<Scalar>::template JacobianFunc<Matrix>& jacobian_func,
    const std::vector<Key>& keys_to_func, const std::vector<Key>& keys_to_optimize) {
  return Factor<Scalar>(
      [jacobian_func](const Values<Scalar>& values, const std::vector<index_entry_t>& keys_to_func,
                      VectorX<Scalar>* residual, Matrix* jacobian, Matrix* hessian,
                      VectorX<Scalar>* rhs) {
        jacobian_func(values, keys_to_func, residual, jacobian);

        SYM_ASSERT(residual != nullptr);
        if (jacobian == nullptr) {
          SYM_ASSERT(hessian == nullptr);
          SYM_ASSERT(rhs == nullptr);
        } else {
          SYM_ASSERT(residual->rows() == jacobian->rows());
          internal::CalculateHessianRhs(*residual, *jacobian, hessian, rhs);
        }
      },
      keys_to_func, keys_to_optimize);
}

template <typename Scalar>
Factor<Scalar> Factor<Scalar>::Jacobian(const DenseJacobianFunc& jacobian_func,
                                        const std::vector<Key>& keys_to_func,
                                        const std::vector<Key>& keys_to_optimize) {
  return FactorFromJacobianFunc<Scalar>(jacobian_func, keys_to_func, keys_to_optimize);
}

template <typename Scalar>
Factor<Scalar> Factor<Scalar>::Jacobian(const DenseJacobianFunc& jacobian_func,
                                        const std::vector<Key>& keys) {
  return Jacobian(jacobian_func, keys, keys);
}

template <typename Scalar>
Factor<Scalar> Factor<Scalar>::Jacobian(const SparseJacobianFunc& jacobian_func,
                                        const std::vector<Key>& keys_to_func,
                                        const std::vector<Key>& keys_to_optimize) {
  return FactorFromJacobianFunc<Scalar>(jacobian_func, keys_to_func, keys_to_optimize);
}

template <typename Scalar>
Factor<Scalar> Factor<Scalar>::Jacobian(const SparseJacobianFunc& jacobian_func,
                                        const std::vector<Key>& keys) {
  return Jacobian(jacobian_func, keys, keys);
}

template <typename Scalar>
void Factor<Scalar>::Linearize(const Values<Scalar>& values, VectorX<Scalar>* residual) const {
  EnsureIndexEntriesExist(values);
  if (IsSparse()) {
    sparse_hessian_func_(values, values_id_and_index_entries_.second, residual, nullptr, nullptr,
                         nullptr);
  } else {
    hessian_func_(values, values_id_and_index_entries_.second, residual, nullptr, nullptr, nullptr);
  }
}

template <typename Scalar>
void Factor<Scalar>::Linearize(const Values<Scalar>& values, VectorX<Scalar>* residual,
                               MatrixX<Scalar>* jacobian) const {
  SYM_ASSERT(!IsSparse());
  EnsureIndexEntriesExist(values);
  hessian_func_(values, values_id_and_index_entries_.second, residual, jacobian, nullptr, nullptr);
}

template <typename Scalar>
void Factor<Scalar>::Linearize(const Values<Scalar>& values, VectorX<Scalar>* residual,
                               Eigen::SparseMatrix<Scalar>* jacobian) const {
  SYM_ASSERT(IsSparse());
  EnsureIndexEntriesExist(values);
  sparse_hessian_func_(values, values_id_and_index_entries_.second, residual, jacobian, nullptr,
                       nullptr);
}

template <typename Scalar>
void Factor<Scalar>::Linearize(const Values<Scalar>& values,
                               LinearizedDenseFactor* linearized_factor) const {
  assert(linearized_factor != nullptr);
  SYM_ASSERT(!IsSparse());

  FillLinearizedFactorIndex(values, *linearized_factor);

  // TODO(hayk): Maybe the function should just accept a LinearizedDenseFactor*
  EnsureIndexEntriesExist(values);
  hessian_func_(values, values_id_and_index_entries_.second, &linearized_factor->residual,
                &linearized_factor->jacobian, &linearized_factor->hessian, &linearized_factor->rhs);

  // Sanity check dimensions
  SYM_ASSERT(linearized_factor->index.tangent_dim == linearized_factor->jacobian.cols());
  SYM_ASSERT(linearized_factor->index.tangent_dim == linearized_factor->hessian.rows());
  SYM_ASSERT(linearized_factor->index.tangent_dim == linearized_factor->rhs.rows());
}

template <typename Scalar>
void Factor<Scalar>::Linearize(const Values<Scalar>& values,
                               LinearizedSparseFactor* linearized_factor) const {
  assert(linearized_factor != nullptr);
  SYM_ASSERT(IsSparse());

  FillLinearizedFactorIndex(values, *linearized_factor);

  // TODO(hayk): Maybe the function should just accept a LinearizedSparseFactor*
  EnsureIndexEntriesExist(values);
  sparse_hessian_func_(values, values_id_and_index_entries_.second, &linearized_factor->residual,
                       &linearized_factor->jacobian, &linearized_factor->hessian,
                       &linearized_factor->rhs);

  // Sanity check dimensions
  SYM_ASSERT(linearized_factor->index.tangent_dim == linearized_factor->jacobian.cols());
  SYM_ASSERT(linearized_factor->index.tangent_dim == linearized_factor->hessian.rows());
  SYM_ASSERT(linearized_factor->index.tangent_dim == linearized_factor->rhs.rows());
}

template <typename Scalar>
typename Factor<Scalar>::LinearizedDenseFactor Factor<Scalar>::Linearize(
    const Values<Scalar>& values) const {
  LinearizedDenseFactor linearized_factor{};
  Linearize(values, &linearized_factor);
  return linearized_factor;
}

template <typename Scalar>
const std::vector<Key>& Factor<Scalar>::OptimizedKeys() const {
  return keys_to_optimize_;
}

template <typename Scalar>
const std::vector<Key>& Factor<Scalar>::AllKeys() const {
  return keys_;
}

template <typename ScalarType>
void Factor<ScalarType>::EnsureIndexEntriesExist(const Values<Scalar>& values) const {
  const auto& cached_values_id = values_id_and_index_entries_.first;
  if (values.Id() == cached_values_id) {
    // If index is cached from the same Values
    return;
  }

  // clear the cached value if a different Values object is used.
  auto& cached_index_entries_ = values_id_and_index_entries_.second;
  cached_index_entries_.clear();
  cached_index_entries_.reserve(keys_.size());
  for (const auto& key : keys_) {
    cached_index_entries_.push_back(values.IndexEntryAt(key));
  }
  values_id_and_index_entries_.first = values.Id();
}

template <typename Scalar>
template <typename LinearizedFactorT>
void Factor<Scalar>::FillLinearizedFactorIndex(const Values<Scalar>& values,
                                               LinearizedFactorT& linearized_factor) const {
  if (linearized_factor.index.storage_dim == 0) {
    // Set the types and everything from the index
    linearized_factor.index = values.CreateIndex(keys_to_optimize_);

    // But the offset we want is within the factor
    int32_t offset = 0;
    for (index_entry_t& entry : linearized_factor.index.entries) {
      entry.offset = offset;
      offset += entry.tangent_dim;
    }
  }
}

// ----------------------------------------------------------------------------
// Printing
// ----------------------------------------------------------------------------

template <typename Scalar>
std::ostream& operator<<(std::ostream& os, const sym::Factor<Scalar>& factor) {
  fmt::print(os, "<Factor optimized keys: {{{}}}, all_keys: {{{}}}>", factor.OptimizedKeys(),
             factor.AllKeys());
  return os;
}

template std::ostream& operator<< <float>(std::ostream& os, const sym::Factor<float>& factor);
template std::ostream& operator<< <double>(std::ostream& os, const sym::Factor<double>& factor);

// TODO(hayk): Why is this needed instead of being able to template operator<<?
template <typename LinearizedFactorT>
std::ostream& PrintLinearizedFactor(std::ostream& os, const LinearizedFactorT& factor) {
  std::vector<key_t> factor_keys;
  std::transform(factor.index.entries.begin(), factor.index.entries.end(),
                 std::back_inserter(factor_keys), [](const auto& entry) { return entry.key; });
  fmt::print(os,
             "<{}\n  keys: {{{}}}\n  storage_dim: {}\n  tangent_dim: {}\n  "
             "residual: ({})\n  jacobian: ({})\n  error: "
             "{}\n>\n",
             factor.getTypeName(), factor_keys, factor.index.storage_dim, factor.index.tangent_dim,
             factor.residual.transpose(), factor.jacobian, 0.5 * factor.residual.squaredNorm());
  return os;
}

std::ostream& operator<<(std::ostream& os, const sym::linearized_dense_factor_t& factor) {
  return PrintLinearizedFactor(os, factor);
}

std::ostream& operator<<(std::ostream& os, const sym::linearized_dense_factorf_t& factor) {
  return PrintLinearizedFactor(os, factor);
}

std::ostream& operator<<(std::ostream& os, const sym::linearized_sparse_factor_t& factor) {
  return PrintLinearizedFactor(os, factor);
}

std::ostream& operator<<(std::ostream& os, const sym::linearized_sparse_factorf_t& factor) {
  return PrintLinearizedFactor(os, factor);
}

}  // namespace sym

// Explicit instantiation
template class sym::Factor<double>;
template class sym::Factor<float>;
