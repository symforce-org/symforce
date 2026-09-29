# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

from __future__ import annotations

import typing as T

import pybind11_stubgen
import pybind11_stubgen.parser.mixins.fix as fix_mixins
import pybind11_stubgen.parser.mixins.parse as parse_mixins
from pybind11_stubgen import ExtractSignaturesFromPybind11Docstrings
from pybind11_stubgen import FixMissingImports as PybindFixMissingImports
from pybind11_stubgen import (
    FixMissingNoneHashFieldAnnotation as PybindFixMissingNoneHashFieldAnnotation,
)
from pybind11_stubgen import IParser
from pybind11_stubgen.structs import Class
from pybind11_stubgen.structs import Docstring
from pybind11_stubgen.structs import Field
from pybind11_stubgen.structs import Identifier
from pybind11_stubgen.structs import Import
from pybind11_stubgen.structs import InvalidExpression
from pybind11_stubgen.structs import QualifiedName
from pybind11_stubgen.structs import ResolvedType
from pybind11_stubgen.structs import Value


class FixMissingImports(PybindFixMissingImports):
    def _add_import(self, name: QualifiedName) -> None:
        if len(name) == 0:
            return
        if name[0] == Identifier("lcmtypes"):
            self.__extra_imports.add(Import(name=None, origin=name.parent))
            return
        super()._add_import(name)


def patch_lcmtype_imports() -> None:
    fix_mixins.FixMissingImports = FixMissingImports  # type: ignore[misc]
    pybind11_stubgen.FixMissingImports = FixMissingImports  # type: ignore[misc]


def patch_handle_docstring() -> None:
    """
    Patch BaseParser.handle_docstring to always strip empty lines from the start or end of
    docstrings
    """

    def handle_docstring(self: IParser, path: QualifiedName, value: T.Any) -> T.Optional[Docstring]:
        if isinstance(value, str):
            assert isinstance(self, ExtractSignaturesFromPybind11Docstrings)
            return self._strip_empty_lines(value.splitlines())
        return None

    parse_mixins.BaseParser.handle_docstring = handle_docstring  # type: ignore[method-assign]


def patch_fix_missing_none_hash_field_annotation() -> None:
    """
    See https://github.com/sizmailov/pybind11-stubgen/pull/236
    """

    def handle_field(
        self: PybindFixMissingNoneHashFieldAnnotation,
        path: QualifiedName,
        field: T.Any,
    ) -> T.Optional[Field]:
        result = super(PybindFixMissingNoneHashFieldAnnotation, self).handle_field(path, field)  # type: ignore[safe-super]
        if result is None:
            return None
        if field is None and path[-1] == "__hash__":
            result.attribute.annotation = self.parse_annotation_str("typing.ClassVar[typing.Any]")
        return result

    fix_mixins.FixMissingNoneHashFieldAnnotation.handle_field = handle_field  # type: ignore[method-assign]


class FixNumpyArrayRemoveParameters(IParser):
    __ndarray_name = QualifiedName.from_str("numpy.typing.ArrayLike")

    def handle_class(self, path: QualifiedName, class_: type) -> T.Optional[Class]:
        maybe_class = super().handle_class(path, class_)  # type: ignore[safe-super]
        if maybe_class is None:
            return maybe_class

        methods = []
        for method in maybe_class.methods:
            if method not in methods:
                methods.append(method)
        maybe_class.methods = methods

        return maybe_class

    def parse_annotation_str(self, annotation_str: str) -> ResolvedType | InvalidExpression | Value:
        result = super().parse_annotation_str(annotation_str)  # type: ignore[safe-super]
        if isinstance(result, ResolvedType) and result.name == QualifiedName.from_str(
            "typing.Annotated"
        ):
            assert (
                result.parameters is not None
                and len(result.parameters) >= 1
                and isinstance(result.parameters[0], ResolvedType)
            )
            if result.parameters[0].name == self.__ndarray_name:
                return result.parameters[0]
        return result


def patch_remove_parameters() -> None:
    """
    Fix NumpyArrayRemoveParameters to work with pybind 3.x and deduplicate overloads
    """

    fix_mixins.FixNumpyArrayRemoveParameters = FixNumpyArrayRemoveParameters  # type: ignore[misc,assignment]
    pybind11_stubgen.FixNumpyArrayRemoveParameters = FixNumpyArrayRemoveParameters  # type: ignore[misc,assignment]
