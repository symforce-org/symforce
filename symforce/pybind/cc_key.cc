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
      .def(py::init<sym::Key::letter_t>(), py::arg("letter"))
      .def(py::init<sym::Key::letter_t, sym::Key::subscript_t>(), py::arg("letter"), py::arg("sub"))
      .def(py::init<sym::Key::letter_t, sym::Key::subscript_t, sym::Key::superscript_t>(),
           py::arg("letter"), py::arg("sub"), py::arg("super"))
      .def_readonly_static("INVALID_LETTER", &sym::Key::kInvalidLetter)
      .def_readonly_static("INVALID_SUB", &sym::Key::kInvalidSub)
      .def_readonly_static("INVALID_SUPER", &sym::Key::kInvalidSuper)
      .def_property_readonly("letter", &sym::Key::Letter, "The letter value of the key.")
      .def_property_readonly("sub", &sym::Key::Sub,
                             "The subscript value of the key (INVALID_SUB if not set).")
      .def_property_readonly("super", &sym::Key::Super,
                             "The superscript value of the key (INVALID_SUPER if not set).")
      .def("with_letter", &sym::Key::WithLetter, py::arg("letter"),
           "Creates a new key with a modified letter from an existing one.")
      .def("with_sub", &sym::Key::WithSub, py::arg("sub"),
           "Creates a new key with a modified subscript from an existing one.")
      .def("with_super", &sym::Key::WithSuper, py::arg("super"),
           "Creates a new key with a modified superscript from an existing one.")
      .def("get_lcm_type", &sym::Key::GetLcmType)
      .def("__eq__",
           [](const sym::Key& self, const py::object& other) {
             return py::isinstance<sym::Key>(other) && self == other.cast<sym::Key>();
           })
      .def(
          "lexical_less_than", &sym::Key::LexicalLessThan,
          "Return true if a is LESS than b, in dictionary order of the tuple (letter, sub, super).")
      .def("__hash__", std::hash<sym::Key>{})
      .def("__repr__", [](const sym::Key& key) { return fmt::format("{}", key); })
      .def(py::pickle(
          [](const sym::Key& key) {  //  __getstate__
            return py::make_tuple(key.Letter(), key.Sub(), key.Super());
          },
          [](py::tuple state) {  // __setstate__
            if (state.size() != 3) {
              throw py::value_error("Key.__setstate__ expected tuple of size 3.");
            }
            return sym::Key(state[0].cast<sym::Key::letter_t>(),
                            state[1].cast<sym::Key::subscript_t>(),
                            state[2].cast<sym::Key::superscript_t>());
          }));
}

}  // namespace sym
