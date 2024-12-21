# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import typing as T

import pybind11_stubgen


class FixMissingImports(pybind11_stubgen.parser.mixins.fix.FixMissingImports):
    def _add_import(self, name: pybind11_stubgen.structs.QualifiedName) -> None:
        if len(name) == 0:
            return
        if name[0] == pybind11_stubgen.structs.Identifier("lcmtypes"):
            self.__extra_imports.add(pybind11_stubgen.structs.Import(name=None, origin=name.parent))
            return
        super()._add_import(name)


def patch_lcmtype_imports() -> None:
    pybind11_stubgen.parser.mixins.fix.FixMissingImports = FixMissingImports  # type: ignore[misc]
    pybind11_stubgen.FixMissingImports = FixMissingImports  # type: ignore[misc]


def patch_handle_docstring() -> None:
    def handle_docstring(
        self: pybind11_stubgen.IParser, path: pybind11_stubgen.structs.QualifiedName, value: T.Any
    ) -> T.Optional[pybind11_stubgen.structs.Docstring]:
        if isinstance(value, str):
            assert isinstance(
                self, pybind11_stubgen.parser.mixins.parse.ExtractSignaturesFromPybind11Docstrings
            )
            return self._strip_empty_lines(value.splitlines())
        return None

    pybind11_stubgen.parser.mixins.parse.BaseParser.handle_docstring = handle_docstring  # type: ignore[method-assign]


def patch_remove_parameters() -> None:
    def handle_class(
        self: pybind11_stubgen.parser.mixins.fix.FixNumpyArrayRemoveParameters,
        path: pybind11_stubgen.structs.QualifiedName,
        class_: type,
    ) -> T.Optional[pybind11_stubgen.structs.Class]:
        maybe_class = super(  # type: ignore[safe-super]
            pybind11_stubgen.parser.mixins.fix.FixNumpyArrayRemoveParameters, self
        ).handle_class(path, class_)
        if maybe_class is None:
            return maybe_class

        methods = []
        for method in maybe_class.methods:
            if method not in methods:
                methods.append(method)
        maybe_class.methods = methods

        return maybe_class

    pybind11_stubgen.parser.mixins.fix.FixNumpyArrayRemoveParameters.handle_class = handle_class  # type: ignore[method-assign,assignment]


def patch_fix_missing_none_hash_field_annotation() -> None:
    """
    See https://github.com/sizmailov/pybind11-stubgen/pull/236
    """

    def handle_field(
        self: pybind11_stubgen.parser.mixins.fix.FixMissingNoneHashFieldAnnotation,
        path: pybind11_stubgen.structs.QualifiedName,
        field: T.Any,
    ) -> T.Optional[pybind11_stubgen.structs.Field]:
        result = super(  # type: ignore[safe-super]
            pybind11_stubgen.parser.mixins.fix.FixMissingNoneHashFieldAnnotation, self
        ).handle_field(path, field)
        if result is None:
            return None
        if field is None and path[-1] == "__hash__":
            result.attribute.annotation = self.parse_annotation_str("typing.ClassVar[typing.Any]")
        return result

    pybind11_stubgen.parser.mixins.fix.FixMissingNoneHashFieldAnnotation.handle_field = handle_field  # type: ignore[method-assign]
