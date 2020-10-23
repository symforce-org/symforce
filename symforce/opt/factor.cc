#include "./factor.h"

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
void Factor<Scalar>::Linearize(const Values<Scalar>& values, VectorX<Scalar>* residual,
                               MatrixX<Scalar>* jacobian) const {
  hessian_func_(values, residual, jacobian, nullptr, nullptr);
}

template <typename Scalar>
void Factor<Scalar>::Linearize(const Values<Scalar>& values,
                               LinearizedFactor* linearized_factor) const {
  assert(linearized_factor != nullptr);

  if (linearized_factor->index.storage_dim == 0) {
    // Set the types and everything from the index
    linearized_factor->index = values.CreateIndex(keys_);

    // But the offset we want is within the factor
    int32_t offset = 0;
    for (index_entry_t& entry : linearized_factor->index.entries) {
      entry.offset = offset;
      offset += entry.tangent_dim;
    }
  }

  // TODO(hayk): Maybe the function should just accept a LinearizedFactor*
  hessian_func_(values, &linearized_factor->residual, &linearized_factor->jacobian,
                &linearized_factor->hessian, &linearized_factor->rhs);

  // Sanity check dimensions
  SYM_ASSERT(linearized_factor->index.tangent_dim == linearized_factor->jacobian.cols());
  SYM_ASSERT(linearized_factor->index.tangent_dim == linearized_factor->hessian.rows());
  SYM_ASSERT(linearized_factor->index.tangent_dim == linearized_factor->rhs.rows());
}

template <typename Scalar>
typename Factor<Scalar>::LinearizedFactor Factor<Scalar>::Linearize(
    const Values<Scalar>& values) const {
  LinearizedFactor linearized_factor{};
  Linearize(values, &linearized_factor);
  return linearized_factor;
}

template <typename Scalar>
const std::vector<Key>& Factor<Scalar>::Keys() const {
  return keys_;
}

}  // namespace sym

// Explicit instantiation
template class sym::Factor<double>;
template class sym::Factor<float>;

// ----------------------------------------------------------------------------
// Printing
// ----------------------------------------------------------------------------

template <typename Scalar>
std::ostream& operator<<(std::ostream& os, const sym::Factor<Scalar>& factor) {
  os << "<Factor keys: {";
  const auto& keys = factor.Keys();
  for (int i = 0; i < keys.size(); ++i) {
    os << keys[i] << (i < keys.size() - 1 ? ", " : "");
  }
  os << "}>";
  return os;
}

template std::ostream& operator<<<float>(std::ostream& os, const sym::Factor<float>& factor);
template std::ostream& operator<<<double>(std::ostream& os, const sym::Factor<double>& factor);

// TODO(hayk): Why is this needed instead of being able to template operator<<?
template <typename Scalar>
std::ostream& PrintLinearizedFactor(std::ostream& os,
                                    const typename sym::Factor<Scalar>::LinearizedFactor& factor) {
  os << "<LinearizedFactor\n";
  os << "  keys: {";
  for (int i = 0; i < factor.index.entries.size(); ++i) {
    os << factor.index.entries[i].key << (i < factor.index.entries.size() - 1 ? ", " : "");
  }
  os << "}\n";
  os << "  storage_dim: " << factor.index.storage_dim << "\n";
  os << "  tangent_dim: " << factor.index.tangent_dim << "\n";
  os << "  residual: (" << factor.residual.transpose() << ")\n";
  os << "  jacobian: (" << factor.jacobian << ")\n";
  os << "  error: " << 0.5 * factor.residual.squaredNorm() << "\n";
  os << ">\n";
  return os;
}

// template std::ostream& operator<<<float>(
//     std::ostream& os, const typename sym::Factor<float>::LinearizedFactor& factor);
// template std::ostream& operator<<<double>(
//     std::ostream& os, const typename sym::Factor<double>::LinearizedFactor& factor);

std::ostream& operator<<(std::ostream& os, const sym::linearized_factor_t& factor) {
  return PrintLinearizedFactor<double>(os, factor);
}

std::ostream& operator<<(std::ostream& os, const sym::linearized_factorf_t& factor) {
  return PrintLinearizedFactor<float>(os, factor);
}
