/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#include "./cc_factor.h"

#include <cstring>
#include <functional>

#include <Eigen/Core>
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

/**
 * If Matrix is Eigen::SparseMatrix<double> and matrix is not a scipy.sparse.csc_matrix, or
 * if Matrix is any other type and matrix is a scipy.sparse.csc_matrix, throws a value error.
 */
template <typename Matrix>
void ThrowIfSparsityMismatch(const py::object& matrix) {
  if (std::strcmp(Py_TYPE(matrix.ptr())->tp_name, "csc_matrix") == 0) {
    throw py::value_error("Non-sparse matrix expected, scipy.sparse.csc_matrix found instead.");
  }
}
template <>
void ThrowIfSparsityMismatch<Eigen::SparseMatrix<double>>(const py::object& matrix) {
  if (!py::isinstance(matrix, py::module_::import("scipy.sparse").attr("csc_matrix"))) {
    throw py::value_error(
        fmt::format("scipy.sparse.csc_matrix expected, found {} instead.", py::type::of(matrix)));
  }
}

template <typename Matrix>
auto WrapPyHessianFunc(PyHessianFunc&& hessian_func) {
  using Vec = Eigen::VectorXd;
  return [hessian_func = std::move(hessian_func)](
             const sym::Valuesd& values, const std::vector<index_entry_t>& keys,
             Vec* const residual, Matrix* const jacobian, Matrix* const hessian, Vec* const rhs) {
    const py::tuple out_tuple = hessian_func(values, keys);
    if (residual != nullptr) {
      *residual = py::cast<Vec>(out_tuple[0]);
    }
    if (jacobian != nullptr) {
      ThrowIfSparsityMismatch<Matrix>(out_tuple[1]);
      *jacobian = py::cast<Matrix>(out_tuple[1]);
    }
    if (hessian != nullptr) {
      ThrowIfSparsityMismatch<Matrix>(out_tuple[2]);
      *hessian = py::cast<Matrix>(out_tuple[2]);
    }
    if (rhs != nullptr) {
      *rhs = py::cast<Vec>(out_tuple[3]);
    }
  };
}

template <typename... Keys>
sym::Factord MakeHessianFactor(PyHessianFunc hessian_func, const std::vector<Keys>&... keys,
                               bool sparse) {
  if (sparse) {
    return sym::Factord(WrapPyHessianFunc<Eigen::SparseMatrix<double>>(std::move(hessian_func)),
                        keys...);
  } else {
    return sym::Factord(WrapPyHessianFunc<Eigen::MatrixXd>(std::move(hessian_func)), keys...);
  }
}

using PyJacobianFunc =
    std::function<py::tuple(const sym::Valuesd&, const std::vector<index_entry_t>&)>;

template <typename Matrix>
sym::Factord::JacobianFunc<Matrix> WrapPyJacobianFunc(PyJacobianFunc&& jacobian_func) {
  return sym::Factord::JacobianFunc<Matrix>(
      [jacobian_func = std::move(jacobian_func)](
          const sym::Valuesd& values, const std::vector<index_entry_t>& keys,
          Eigen::VectorXd* const residual, Matrix* const jacobian) {
        const py::tuple out_tuple = jacobian_func(values, keys);
        if (residual != nullptr) {
          *residual = py::cast<Eigen::VectorXd>(out_tuple[0]);
        }
        if (jacobian != nullptr) {
          ThrowIfSparsityMismatch<Matrix>(out_tuple[1]);
          *jacobian = py::cast<Matrix>(out_tuple[1]);
        }
      });
}

template <typename... Keys>
sym::Factord MakeJacobianFactor(PyJacobianFunc jacobian_func, const std::vector<Keys>&... keys,
                                bool sparse) {
  if (sparse) {
    return sym::Factord(WrapPyJacobianFunc<Eigen::SparseMatrix<double>>(std::move(jacobian_func)),
                        keys...);
  } else {
    return sym::Factord(WrapPyJacobianFunc<Eigen::MatrixXd>(std::move(jacobian_func)), keys...);
  }
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
      .def(py::init(&MakeHessianFactor<sym::Key>), py::arg("hessian_func"), py::arg("keys"),
           py::arg("sparse") = false, R"(
           Create directly from a hessian functor. This is the lowest-level constructor.

           Args:
             keys: The set of input arguments, in order, accepted by func.
             sparse: Create a sparse factor if True, dense factor if false. Defaults to dense.

           Precondition:
             The jacobian and hessian returned by hessian_func have type scipy.sparse.csc_matrix if and only if sparse = True.
           )")
      .def(py::init(&MakeHessianFactor<sym::Key, sym::Key>), py::arg("hessian_func"),
           py::arg("keys_to_func"), py::arg("keys_to_optimize"), py::arg("sparse") = false,
           R"(
           Create directly from a hessian functor. This is the lowest-level constructor.

           Args:
             keys_to_func: The set of input arguments, in order, accepted by func.
             keys_to_optimize: The set of input arguments that correspond to the derivative in func. Must be a subset of keys_to_func.
             sparse: Create a sparse factor if True, dense factor if false. Defaults to dense.

           Precondition:
             The jacobian and hessian returned by hessian_func have type scipy.sparse.csc_matrix if and only if sparse = True.
           )")
      .def("is_sparse", &sym::Factord::IsSparse,
           "Does this factor use a sparse jacobian/hessian matrix?")
      .def_static("jacobian", &MakeJacobianFactor<sym::Key>, py::arg("jacobian_func"),
                  py::arg("keys"), py::arg("sparse") = false, R"(
           Create from a function that computes the jacobian. The hessian will be computed using the
           Gauss Newton approximation::

               H   = J.T * J
               rhs = J.T * b

           Args:
             keys: The set of input arguments, in order, accepted by func.
             sparse: Create a sparse factor if True, dense factor if false. Defaults to dense.

           Precondition:
             The jacobian returned by jacobian_func has type scipy.sparse.csc_matrix if and only if sparse = True.
           )")
      .def_static("jacobian", &MakeJacobianFactor<sym::Key, sym::Key>, py::arg("jacobian_func"),
                  py::arg("keys_to_func"), py::arg("keys_to_optimize"), py::arg("sparse") = false,
                  R"(
           Create from a function that computes the jacobian. The hessian will be computed using the
           Gauss Newton approximation::

               H   = J.T * J
               rhs = J.T * b

           Args:
             keys_to_func: The set of input arguments, in order, accepted by func.
             keys_to_optimize: The set of input arguments that correspond to the derivative in func. Must be a subset of keys_to_func.
             sparse: Create a sparse factor if True, dense factor if false. Defaults to dense.

             Precondition:
               The jacobian returned by jacobian_func has type scipy.sparse.csc_matrix if and only if sparse = True.
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
      .def(
          "linearized_factor",
          [](const sym::Factord& factor, const sym::Valuesd& values) {
            return factor.Linearize(values);
          },
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
