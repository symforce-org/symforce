/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#include "./cc_key.h"

#include <fmt/format.h>
#include <fmt/ostream.h>

#include <symforce/opt/key.h>

#include "./lcm_type_casters.h"
#include "./sym_type_casters.h"

namespace py = pybind11;

namespace sym {

void AddKeyWrapper(py::module_ module) {
  py::class_<sym::Key>(module, "Key")
      .def(py::init<char>(), py::arg("letter"))
      .def(py::init<char, sym::Key::subscript_t>(), py::arg("letter"), py::arg("sub"))
      .def(py::init<char, sym::Key::subscript_t, sym::Key::superscript_t>(), py::arg("letter"),
           py::arg("sub"), py::arg("super"))
      .def_property_readonly("letter", &sym::Key::Letter)
      .def_property_readonly("sub", &sym::Key::Sub)
      .def_property_readonly("super", &sym::Key::Super)
      .def_static("with_super", &sym::Key::WithSuper, py::arg("key"), py::arg("super"))
      .def("get_lcm_type", &sym::Key::GetLcmType)
      .def("__eq__",
           [](const sym::Key& self, const py::object& other) {
             return py::isinstance<sym::Key>(other) && self == other.cast<sym::Key>();
           })
      .def("lexical_less_than", &sym::Key::LexicalLessThan)
      .def("__hash__", std::hash<sym::Key>{})
      .def("__repr__", [](const sym::Key& key) { return fmt::format("{}", key); });
}

}  // namespace sym
