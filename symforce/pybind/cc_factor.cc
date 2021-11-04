
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

sym::Factord MakeHessianFactor(std::function<py::tuple(const sym::Valuesd&)> hessian_func,
                               const std::vector<sym::Key>& keys) {
  using Vec = Eigen::VectorXd;
  using Mat = Eigen::MatrixXd;
  return sym::Factord(
      [hessian_func = std::move(hessian_func)](const sym::Valuesd& values, Vec* const residual,
                                               Mat* const jacobian, Mat* const hessian,
                                               Vec* const rhs) {
        const py::tuple out_tuple = hessian_func(values);
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
      keys);
}

sym::Factord MakeJacobianFactor(std::function<py::tuple(const sym::Valuesd&)> jacobian_func,
                                const std::vector<sym::Key>& keys) {
  return sym::Factord::Jacobian(
      sym::Factord::JacobianFunc([jacobian_func = std::move(jacobian_func)](
                                     const sym::Valuesd& values, Eigen::VectorXd* const residual,
                                     Eigen::MatrixXd* const jacobian) {
        const py::tuple out_tuple = jacobian_func(values);
        if (residual != nullptr) {
          *residual = py::cast<Eigen::VectorXd>(out_tuple[0]);
        }
        if (jacobian != nullptr) {
          *jacobian = py::cast<Eigen::MatrixXd>(out_tuple[1]);
        }
      }),
      keys);
}

//================================================================================================//
//-------------------------------- The Public Factor Wrapper -------------------------------------//
//================================================================================================//

void AddFactorWrapper(pybind11::module_ module) {
  py::class_<sym::Factord>(module, "Factor")
      .def(py::init(&MakeHessianFactor), py::arg("hessian_func"), py::arg("keys"))
      .def_static("jacobian", &MakeJacobianFactor, py::arg("jacobian_func"), py::arg("keys"))
      .def("linearize",
           [](const sym::Factord& factor, const sym::Valuesd& values) {
             Eigen::VectorXd residual;
             Eigen::MatrixXd jacobian;
             factor.Linearize(values, &residual, &jacobian);
             return py::make_tuple(residual, jacobian);
           })
      .def("linearized_factor",
           py::overload_cast<const sym::Valuesd&>(&sym::Factord::Linearize, py::const_),
           py::arg("values"))
      .def("keys", &sym::Factord::Keys)
      .def("__repr__", [](const sym::Factord& factor) { return fmt::format("{}", factor); });
}

}  // namespace sym
