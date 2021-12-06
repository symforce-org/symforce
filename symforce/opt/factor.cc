#include "./factor.h"

#include <fmt/format.h>
#include <fmt/ostream.h>
#include <fmt/ranges.h>

#include "./assert.h"

namespace sym {

template <typename Scalar>
Factor<Scalar> Factor<Scalar>::Jacobian(const JacobianFunc& jacobian_func,
                                        const std::vector<Key>& keys) {
  return Factor<Scalar>(
      [jacobian_func](const Values<Scalar>& values, VectorX<Scalar>* residual,
                      MatrixX<Scalar>* jacobian, MatrixX<Scalar>* hessian, VectorX<Scalar>* rhs) {
        SYM_ASSERT(residual != nullptr);
        jacobian_func(values, residual, jacobian);
        SYM_ASSERT(jacobian == nullptr || residual->rows() == jacobian->rows());

        // Compute the lower triangle of the hessian if needed
        if (hessian != nullptr) {
          SYM_ASSERT(jacobian != nullptr);
          hessian->resize(jacobian->cols(), jacobian->cols());
          hessian->template triangularView<Eigen::Lower>().setZero();
          hessian->template selfadjointView<Eigen::Lower>().rankUpdate(jacobian->transpose());
        }

        // Compute RHS if needed
        if (rhs != nullptr) {
          SYM_ASSERT(jacobian != nullptr);
          (*rhs) = jacobian->transpose() * (*residual);
        }
      },
      keys);
}

template <typename Scalar>
void Factor<Scalar>::Linearize(const Values<Scalar>& values, VectorX<Scalar>* residual) const {
  if (IsSparse()) {
    sparse_hessian_func_(values, residual, nullptr, nullptr, nullptr);
  } else {
    hessian_func_(values, residual, nullptr, nullptr, nullptr);
  }
}

template <typename Scalar>
void Factor<Scalar>::Linearize(const Values<Scalar>& values, VectorX<Scalar>* residual,
                               MatrixX<Scalar>* jacobian) const {
  SYM_ASSERT(!IsSparse());
  hessian_func_(values, residual, jacobian, nullptr, nullptr);
}

template <typename Scalar>
void Factor<Scalar>::Linearize(const Values<Scalar>& values, VectorX<Scalar>* residual,
                               Eigen::SparseMatrix<Scalar>* jacobian) const {
  SYM_ASSERT(IsSparse());
  sparse_hessian_func_(values, residual, jacobian, nullptr, nullptr);
}

template <typename Scalar>
void Factor<Scalar>::Linearize(const Values<Scalar>& values,
                               LinearizedDenseFactor* linearized_factor) const {
  assert(linearized_factor != nullptr);
  SYM_ASSERT(!IsSparse());

  FillLinearizedFactorIndex(values, *linearized_factor);

  // TODO(hayk): Maybe the function should just accept a LinearizedDenseFactor*
  hessian_func_(values, &linearized_factor->residual, &linearized_factor->jacobian,
                &linearized_factor->hessian, &linearized_factor->rhs);

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
  sparse_hessian_func_(values, &linearized_factor->residual, &linearized_factor->jacobian,
                       &linearized_factor->hessian, &linearized_factor->rhs);

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
const std::vector<Key>& Factor<Scalar>::Keys() const {
  return keys_;
}

template <typename Scalar>
template <typename LinearizedFactorT>
void Factor<Scalar>::FillLinearizedFactorIndex(const Values<Scalar>& values,
                                               LinearizedFactorT& linearized_factor) const {
  if (linearized_factor.index.storage_dim == 0) {
    // Set the types and everything from the index
    linearized_factor.index = values.CreateIndex(keys_);

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
  fmt::print(os, "<Factor keys: {{{}}}", factor.Keys());
  return os;
}

template std::ostream& operator<<<float>(std::ostream& os, const sym::Factor<float>& factor);
template std::ostream& operator<<<double>(std::ostream& os, const sym::Factor<double>& factor);

// TODO(hayk): Why is this needed instead of being able to template operator<<?
template <typename Scalar>
std::ostream& PrintLinearizedFactor(
    std::ostream& os, const typename sym::Factor<Scalar>::LinearizedDenseFactor& factor) {
  std::vector<key_t> factor_keys;
  std::transform(factor.index.entries.begin(), factor.index.entries.end(),
                 std::back_inserter(factor_keys), [](const auto& entry) { return entry.key; });
  fmt::print(os,
             "<LinearizedDenseFactor\n  keys: {{{}}}}\n  storage_dim: {}\n  tangent_dim: {}\n  "
             "residual: ({})\n  jacobian: ({})\n  error: "
             "{}\n>\n",
             factor_keys, factor.index.storage_dim, factor.index.tangent_dim,
             factor.residual.transpose(), factor.jacobian, 0.5 * factor.residual.squaredNorm());
  return os;
}

std::ostream& operator<<(std::ostream& os, const sym::linearized_dense_factor_t& factor) {
  return PrintLinearizedFactor<double>(os, factor);
}

std::ostream& operator<<(std::ostream& os, const sym::linearized_dense_factorf_t& factor) {
  return PrintLinearizedFactor<float>(os, factor);
}

}  // namespace sym

// Explicit instantiation
template class sym::Factor<double>;
template class sym::Factor<float>;
