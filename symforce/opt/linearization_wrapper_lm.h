#pragma once

#include "../../../../util/levenberg_marquardt/levenberg_marquardt.h"
#include "./assert.h"
#include "./linearization.h"
#include "./util.h"

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

  Eigen::Map<const Eigen::SparseMatrix<double>> Hessian() const override {
    AC_ASSERT(IsInitialized());
    const Eigen::SparseMatrix<Scalar>& H_sparse = lin_.HessianLowerSparse();
    return Eigen::Map<const Eigen::SparseMatrix<Scalar>>(
        H_sparse.rows(), H_sparse.cols(), H_sparse.nonZeros(), H_sparse.outerIndexPtr(),
        H_sparse.innerIndexPtr(), H_sparse.valuePtr(), H_sparse.innerNonZeroPtr());
  }

  Eigen::Map<const Eigen::SparseMatrix<double>> Jacobian() const override {
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
  static std::function<void(const sym::Values<Scalar>&, LinearizationWrapperLM*)> LinearizeFunc(
      const index_t& index,
      std::function<const sym::Linearization<Scalar>&(const sym::Values<Scalar>&)>&&
          linearization_getter,
      const Scalar epsilon, const bool check_derivatives = false) {
    auto linearize_func = [linearization_getter{std::move(linearization_getter)}](
                              const Values<Scalar>& values,
                              LinearizationWrapperLM<Scalar>* linearization) {
      if (!linearization->IsInitialized()) {
        linearization->lin_ = linearization_getter(values);
      } else {
        linearization->lin_.Relinearize(values);
      }
      linearization->SetInitialized(true);
    };

    if (check_derivatives) {
      return WrapLinearizeFuncWithDerivativeChecker(index, std::move(linearize_func), epsilon);
    } else {
      return linearize_func;
    }
  }

  /**
   * Update / retraction function to pass to the LM optimizer.
   */
  static auto UpdateFunc(const index_t& index, const Scalar epsilon) {
    return [&index, epsilon](const Values<Scalar>& v, const VectorX<Scalar>& update,
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
  /**
   * Helper to wrap linearize_func in a functor that linearizes and also checks the result.  The
   * jacobian is checked against the numerical derivative of the residual, and the Hessian
   * is checked against the actual product J^T J
   */
  template <typename LinearizeFuncLambda>
  static std::function<void(const sym::Values<Scalar>&, LinearizationWrapperLM*)>
  WrapLinearizeFuncWithDerivativeChecker(const index_t& index, LinearizeFuncLambda&& linearize_func,
                                         const Scalar epsilon) {
    return [&index, epsilon, linearize_func{std::move(linearize_func)}](
               const Values<Scalar>& values, LinearizationWrapperLM<Scalar>* linearization) {
      linearize_func(values, linearization);

      // Save symbolic results for comparison
      const Eigen::VectorXd symbolic_residual = linearization->Residual();
      const Eigen::VectorXd symbolic_Jtb = linearization->Jtb();
      const Eigen::MatrixXd symbolic_jacobian = linearization->Jacobian();
      const Eigen::MatrixXd symbolic_hessian = linearization->Hessian();

      // Check numerical jacobian
      {
        const auto wrapped_residual =
            [&linearization, &values, &index,
             epsilon](const Eigen::VectorXd& values_perturbation) -> Eigen::VectorXd {
          sym::Valuesd perturbed_values = values;
          perturbed_values.Retract(index, values_perturbation.data(), epsilon);

          linearization->lin_.Relinearize(perturbed_values);
          return linearization->Residual();
        };

        const Eigen::MatrixXd numerical_jacobian = sym::NumericalDerivative(
            wrapped_residual, Eigen::VectorXd::Zero(linearization->Jacobian().cols()).eval(),
            epsilon, std::sqrt(epsilon));

        const bool jacobian_matches =
            numerical_jacobian.isApprox(symbolic_jacobian, std::sqrt(epsilon));

        if (!jacobian_matches) {
          std::ostringstream ss;
          ss << "Symbolic and numerical jacobians don't match" << std::endl;
          ss << "Symbolic Jacobian: \n" << symbolic_jacobian << std::endl;
          ss << "Numerical Jacobian: \n" << numerical_jacobian;
          std::cout << ss.str() << std::endl;
          SYM_ASSERT(jacobian_matches);
        }
      }

      // Check hessian
      {
        Eigen::MatrixXd full_hessian = symbolic_hessian + symbolic_hessian.transpose();
        full_hessian.diagonal() = symbolic_hessian.diagonal();

        const Eigen::MatrixXd numerical_hessian = symbolic_jacobian.transpose() * symbolic_jacobian;
        const bool hessian_matches = full_hessian.isApprox(numerical_hessian, std::sqrt(epsilon));
        if (!hessian_matches) {
          std::ostringstream ss;
          ss << "Hessian does not match J^T J" << std::endl;
          ss << "Symbolic (sym::Linearization) Hessian:\n" << full_hessian << std::endl;
          ss << "Numerical (J^T * J) Hessian:\n" << numerical_hessian << std::endl;
          std::cout << ss.str() << std::endl;
          SYM_ASSERT(hessian_matches);
        }
      }

      // Check Jtb
      {
        const Eigen::VectorXd numerical_Jtb = symbolic_jacobian.transpose() * symbolic_residual;
        const bool Jtb_matches = symbolic_Jtb.isApprox(numerical_Jtb, std::sqrt(epsilon));
        if (!Jtb_matches) {
          std::ostringstream ss;
          ss << "Generated Jtb does not match J^T * b" << std::endl;
          ss << "Symbolic (sym::Linearization) Jtb:\n" << symbolic_Jtb.transpose() << std::endl;
          ss << "Numerical (J^T * b) Jtb:\n" << numerical_Jtb.transpose() << std::endl;
          std::cout << ss.str() << std::endl;
          SYM_ASSERT(Jtb_matches);
        }
      }

      // NOTE(aaron): lin_ was relinearized multiple times to compute the numerical jacobian, so we
      // relinearize at the correct point before returning
      linearization->lin_.Relinearize(values);
    };
  }

  Linearization<Scalar> lin_;
};

}  // namespace sym
