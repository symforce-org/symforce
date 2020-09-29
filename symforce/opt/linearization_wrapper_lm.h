#pragma once

#include "../../../../util/levenberg_marquardt/levenberg_marquardt.h"
#include "./linearization.h"

namespace sym {

/**
 * Shim class to connect the symforce Linearization class to the LM optimizer equivalent.
 */
template <typename Scalar>
class LinearizationWrapperLM : public levenberg_marquardt::LinearizationBase {
 public:
  LinearizationWrapperLM() {}

  Eigen::Map<const Eigen::VectorXd> Residual() const override {
    AC_ASSERT(IsInitialized());
    return Eigen::Map<const Eigen::VectorXd>(lin_.Residual().data(), lin_.Residual().size());
  }

  Eigen::Map<const Eigen::SparseMatrix<Scalar>> Hessian() const override {
    AC_ASSERT(IsInitialized());
    const Eigen::SparseMatrix<Scalar>& H_sparse = lin_.HessianLowerSparse();
    return Eigen::Map<const Eigen::SparseMatrix<Scalar>>(
        H_sparse.rows(), H_sparse.cols(), H_sparse.nonZeros(), H_sparse.outerIndexPtr(),
        H_sparse.innerIndexPtr(), H_sparse.valuePtr(), H_sparse.innerNonZeroPtr());
  }

  Eigen::Map<const Eigen::SparseMatrix<Scalar>> Jacobian() const override {
    AC_ASSERT(IsInitialized());
    const Eigen::SparseMatrix<Scalar>& J_sparse = lin_.JacobianSparse();
    return Eigen::Map<const Eigen::SparseMatrix<Scalar>>(
        J_sparse.rows(), J_sparse.cols(), J_sparse.nonZeros(), J_sparse.outerIndexPtr(),
        J_sparse.innerIndexPtr(), J_sparse.valuePtr(), J_sparse.innerNonZeroPtr());
  }

  Eigen::Map<const Eigen::VectorXd> JacobianValues() const override {
    AC_ASSERT(IsInitialized());
    return Eigen::Map<const Eigen::VectorXd>(lin_.JacobianSparse().valuePtr(),
                                             lin_.JacobianSparse().nonZeros());
  }

  Eigen::Map<const Eigen::VectorXd> Jtb() const override {
    AC_ASSERT(IsInitialized());
    return Eigen::Map<const Eigen::VectorXd>(lin_.Rhs().data(), lin_.Rhs().size());
  }

  /**
   * Linearization function to pass to the LM optimizer.
   */
  static auto LinearizeFunc(const std::vector<Factor<Scalar>>& factors) {
    return [&factors](const Values<Scalar>& values, LinearizationWrapperLM<Scalar>* linearization) {
      if (!linearization->IsInitialized()) {
        linearization->lin_ = Linearization<Scalar>(factors, values);
      } else {
        linearization->lin_.Relinearize(values);
      }
      linearization->SetInitialized(true);
    };
  }

  /**
   * Update / retraction function to pass to the LM optimizer.
   */
  static auto UpdateFunc(const index_t& index, const Scalar epsilon) {
    return [&index, &epsilon](const Values<Scalar>& v, const VectorX<Scalar>& update,
                              Values<Scalar>* updated_inputs) {
      SYM_ASSERT(update.rows() == index.tangent_dim);

      if (updated_inputs->NumEntries() == 0) {
        // If the state blocks are empty the first time, copy in the full structure
        (*updated_inputs) = v;
      } else {
        // Otherwise just copy the keys being optimized
        updated_inputs->Update(index, v);
      }

      // Apply the update
      updated_inputs->Retract(index, update.data(), epsilon);
    };
  }

 private:
  Linearization<Scalar> lin_;
};

}  // namespace sym
