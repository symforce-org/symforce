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
  py::class_<sym::Key>(
      module, "Key",
      "Key type for Values. Contains a letter plus an integral subscript and superscript. Can "
      "construct with a letter, a letter + sub, or a letter + sub + super, but not a letter + "
      "super.")
      .def(py::init<char>(), py::arg("letter"))
      .def(py::init<char, sym::Key::subscript_t>(), py::arg("letter"), py::arg("sub"))
      .def(py::init<char, sym::Key::subscript_t, sym::Key::superscript_t>(), py::arg("letter"),
           py::arg("sub"), py::arg("super"))
      .def_readonly_static("INVALID_LETTER", &sym::Key::kInvalidLetter)
      .def_readonly_static("INVALID_SUB", &sym::Key::kInvalidSub)
      .def_readonly_static("INVALID_SUPER", &sym::Key::kInvalidSuper)
      .def_property_readonly("letter", &sym::Key::Letter, "The letter value of the key.")
      .def_property_readonly("sub", &sym::Key::Sub,
                             "The subscript value of the key (INVALID_SUB if not set).")
      .def_property_readonly("super", &sym::Key::Super,
                             "The superscript value of the key (INVALID_SUPER if not set).")
      .def_static("with_super", &sym::Key::WithSuper, py::arg("key"), py::arg("super"),
                  "Create a new Key from an existing Key and a superscript. The superscript on the "
                  "existing Key must be empty.")
      .def("get_lcm_type", &sym::Key::GetLcmType)
      .def("__eq__",
           [](const sym::Key& self, const py::object& other) {
             return py::isinstance<sym::Key>(other) && self == other.cast<sym::Key>();
           })
      .def(
          "lexical_less_than", &sym::Key::LexicalLessThan,
          "Return true if a is LESS than b, in dictionary order of the tuple (letter, sub, super).")
      .def("__hash__", std::hash<sym::Key>{})
      .def("__repr__", [](const sym::Key& key) { return fmt::format("{}", key); });
}

}  // namespace sym
