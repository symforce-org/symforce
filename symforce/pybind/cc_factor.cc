
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

sym::Factord MakeHessianFactor(PyHessianFunc hessian_func,
                               const std::vector<sym::Key>& keys_to_func,
                               const std::vector<sym::Key>& keys_to_optimize) {
  using Vec = Eigen::VectorXd;
  using Mat = Eigen::MatrixXd;
  return sym::Factord(
      [hessian_func = std::move(hessian_func)](
          const sym::Valuesd& values, const std::vector<index_entry_t>& keys, Vec* const residual,
          Mat* const jacobian, Mat* const hessian, Vec* const rhs) {
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
      },
      keys_to_func, keys_to_optimize);
}

sym::Factord MakeHessianFactor(PyHessianFunc hessian_func, const std::vector<sym::Key>& keys) {
  return MakeHessianFactor(std::move(hessian_func), keys, keys);
}

using PyJacobianFunc =
    std::function<py::tuple(const sym::Valuesd&, const std::vector<index_entry_t>&)>;

sym::Factord MakeJacobianFactor(PyJacobianFunc jacobian_func,
                                const std::vector<sym::Key>& keys_to_func,
                                const std::vector<sym::Key>& keys_to_optimize) {
  return sym::Factord::Jacobian(
      sym::Factord::JacobianFunc(
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
          }),
      keys_to_func, keys_to_optimize);
}

sym::Factord MakeJacobianFactor(PyJacobianFunc jacobian_func, const std::vector<sym::Key>& keys) {
  return MakeJacobianFactor(std::move(jacobian_func), keys, keys);
}

}  // namespace

//================================================================================================//
//-------------------------------- The Public Factor Wrapper -------------------------------------//
//================================================================================================//

void AddFactorWrapper(pybind11::module_ module) {
  py::class_<sym::Factord>(module, "Factor")
      .def(py::init(py::overload_cast<PyHessianFunc, const std::vector<Key>&>(&MakeHessianFactor)),
           py::arg("hessian_func"), py::arg("keys"))
      .def(py::init(
               py::overload_cast<PyHessianFunc, const std::vector<Key>&, const std::vector<Key>&>(
                   &MakeHessianFactor)),
           py::arg("hessian_func"), py::arg("keys_to_func"), py::arg("keys_to_optimize"))
      .def_static("jacobian",
                  py::overload_cast<PyJacobianFunc, const std::vector<Key>&>(&MakeJacobianFactor),
                  py::arg("jacobian_func"), py::arg("keys"))
      .def_static(
          "jacobian",
          py::overload_cast<PyJacobianFunc, const std::vector<Key>&, const std::vector<Key>&>(
              &MakeJacobianFactor),
          py::arg("jacobian_func"), py::arg("keys_to_func"), py::arg("keys_to_optimize"))
      .def("linearize",
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
           })
      .def("linearized_factor",
           py::overload_cast<const sym::Valuesd&>(&sym::Factord::Linearize, py::const_),
           py::arg("values"))
      .def("optimized_keys", &sym::Factord::OptimizedKeys)
      .def("all_keys", &sym::Factord::AllKeys)
      .def("__repr__", [](const sym::Factord& factor) { return fmt::format("{}", factor); });
}

}  // namespace sym
