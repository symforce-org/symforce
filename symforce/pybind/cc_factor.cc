/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#include "./cc_factor.h"

#include <functional>

#include <Eigen/Dense>
#include <fmt/format.h>
#include <fmt/ostream.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <symforce/opt/factor.h>
#include <symforce/opt/key.h>
#include <symforce/opt/values.h>

#include "./lcm_type_casters.h"
#include "./sym_type_casters.h"

namespace py = pybind11;

namespace sym {

//================================================================================================//
//---------------------------- Helpers for wrapping sym::Factord ---------------------------------//
//================================================================================================//

namespace {

using PyHessianFunc =
    std::function<py::tuple(const sym::Valuesd&, const std::vector<index_entry_t>&)>;

auto WrapPyHessianFunc(PyHessianFunc&& hessian_func) {
  using Vec = Eigen::VectorXd;
  using Mat = Eigen::MatrixXd;
  return [hessian_func = std::move(hessian_func)](
             const sym::Valuesd& values, const std::vector<index_entry_t>& keys,
             Vec* const residual, Mat* const jacobian, Mat* const hessian, Vec* const rhs) {
    const py::tuple out_tuple = hessian_func(values, keys);
    if (residual != nullptr) {
      *residual = py::cast<Vec>(out_tuple[0]);
    }
    if (jacobian != nullptr) {
      *jacobian = py::cast<Mat>(out_tuple[1]);
    }
    if (hessian != nullptr) {
      *hessian = py::cast<Mat>(out_tuple[2]);
    }
    if (rhs != nullptr) {
      *rhs = py::cast<Vec>(out_tuple[3]);
    }
  };
}

sym::Factord MakeHessianFactorSeparateKeys(PyHessianFunc hessian_func,
                                           const std::vector<sym::Key>& keys_to_func,
                                           const std::vector<sym::Key>& keys_to_optimize) {
  return sym::Factord(WrapPyHessianFunc(std::move(hessian_func)), keys_to_func, keys_to_optimize);
}

sym::Factord MakeHessianFactorCommonKeys(PyHessianFunc hessian_func,
                                         const std::vector<sym::Key>& keys) {
  return sym::Factord(WrapPyHessianFunc(std::move(hessian_func)), keys);
}

using PyJacobianFunc =
    std::function<py::tuple(const sym::Valuesd&, const std::vector<index_entry_t>&)>;

sym::Factord::DenseJacobianFunc WrapPyJacobianFunc(PyJacobianFunc&& jacobian_func) {
  return sym::Factord::DenseJacobianFunc(
      [jacobian_func = std::move(jacobian_func)](
          const sym::Valuesd& values, const std::vector<index_entry_t>& keys,
          Eigen::VectorXd* const residual, Eigen::MatrixXd* const jacobian) {
        const py::tuple out_tuple = jacobian_func(values, keys);
        if (residual != nullptr) {
          *residual = py::cast<Eigen::VectorXd>(out_tuple[0]);
        }
        if (jacobian != nullptr) {
          *jacobian = py::cast<Eigen::MatrixXd>(out_tuple[1]);
        }
      });
}

sym::Factord MakeJacobianFactorSeparateKeys(PyJacobianFunc jacobian_func,
                                            const std::vector<sym::Key>& keys_to_func,
                                            const std::vector<sym::Key>& keys_to_optimize) {
  return sym::Factord::Jacobian(WrapPyJacobianFunc(std::move(jacobian_func)), keys_to_func,
                                keys_to_optimize);
}

sym::Factord MakeJacobianFactorCommonKeys(PyJacobianFunc jacobian_func,
                                          const std::vector<sym::Key>& keys) {
  return sym::Factord::Jacobian(WrapPyJacobianFunc(std::move(jacobian_func)), keys);
}

}  // namespace

//================================================================================================//
//-------------------------------- The Public Factor Wrapper -------------------------------------//
//================================================================================================//

void AddFactorWrapper(pybind11::module_ module) {
  py::class_<sym::Factord>(module, "Factor", R"(
      A residual term for optimization.

      Created from a function and a set of Keys that act as inputs. Given a Values as an evaluation
      point, generates a linear approximation to the residual function.
  )")
      // TODO(brad): Add wrapper of the constructor from SparseHessianFunc
      .def(py::init(&MakeHessianFactorCommonKeys), py::arg("hessian_func"), py::arg("keys"), R"(
              Create directly from a (dense) hessian functor. This is the lowest-level constructor.

              Args:
                keys: The set of input arguments, in order, accepted by func.
           )")
      .def(py::init(&MakeHessianFactorSeparateKeys), py::arg("hessian_func"),
           py::arg("keys_to_func"), py::arg("keys_to_optimize"),
           R"(
              Create directly from a (sparse) hessian functor. This is the lowest-level constructor.

              Args:
                keys_to_func: The set of input arguments, in order, accepted by func.
                keys_to_optimize: The set of input arguments that correspond to the derivative in func. Must be a subset of keys_to_func.
           )")
      .def("is_sparse", &sym::Factord::IsSparse,
           "Does this factor use a sparse jacobian/hessian matrix?")
      .def_static("jacobian", &MakeJacobianFactorCommonKeys, py::arg("jacobian_func"),
                  py::arg("keys"), R"(
                    Create from a function that computes the jacobian. The hessian will be computed using the
                    Gauss Newton approximation:
                        H   = J.T * J
                        rhs = J.T * b

                    Args:
                      keys: The set of input arguments, in order, accepted by func.
                  )")
      .def_static("jacobian", &MakeJacobianFactorSeparateKeys, py::arg("jacobian_func"),
                  py::arg("keys_to_func"), py::arg("keys_to_optimize"), R"(
            Create from a function that computes the jacobian. The hessian will be computed using the
            Gauss Newton approximation:
                H   = J.T * J
                rhs = J.T * b

            Args:
              keys_to_func: The set of input arguments, in order, accepted by func.
              keys_to_optimize: The set of input arguments that correspond to the derivative in func. Must be a subset of keys_to_func.
          )")
      .def(
          "linearize",
          [](const sym::Factord& factor, const sym::Valuesd& values) {
            if (factor.IsSparse()) {
              Eigen::VectorXd residual;
              Eigen::SparseMatrix<double> jacobian;
              factor.Linearize(values, &residual, &jacobian);
              return py::make_tuple(residual, jacobian);
            } else {
              Eigen::VectorXd residual;
              Eigen::MatrixXd jacobian;
              factor.Linearize(values, &residual, &jacobian);
              return py::make_tuple(residual, jacobian);
            }
          },
          "Evaluate the factor at the given linearization point and output just the numerical "
          "values of the residual and jacobian.")
      .def("linearized_factor",
           py::overload_cast<const sym::Valuesd&>(&sym::Factord::Linearize, py::const_),
           py::arg("values"), R"(
             Evaluate the factor at the given linearization point and output a LinearizedDenseFactor that
             contains the numerical values of the residual, jacobian, hessian, and right-hand-side.

             This can only be called if is_sparse is false; otherwise, it will throw.
           )")
      .def("optimized_keys", &sym::Factord::OptimizedKeys,
           "Get the optimized keys for this factor.")
      .def("all_keys", &sym::Factord::AllKeys, "Get all keys required to evaluate this factor.")
      .def("__repr__", [](const sym::Factord& factor) { return fmt::format("{}", factor); });
}

}  // namespace sym
