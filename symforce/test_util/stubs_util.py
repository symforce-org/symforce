# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

from __future__ import annotations

import typing as T

import pybind11_stubgen
from pybind11_stubgen.structs import Class
from pybind11_stubgen.structs import Docstring
from pybind11_stubgen.structs import Field
from pybind11_stubgen.structs import Identifier
from pybind11_stubgen.structs import Import
from pybind11_stubgen.structs import InvalidExpression
from pybind11_stubgen.structs import QualifiedName
from pybind11_stubgen.structs import ResolvedType
from pybind11_stubgen.structs import Value


class FixMissingImports(pybind11_stubgen.parser.mixins.fix.FixMissingImports):
    def _add_import(self, name: QualifiedName) -> None:
        if len(name) == 0:
            return
        if name[0] == Identifier("lcmtypes"):
            self.__extra_imports.add(Import(name=None, origin=name.parent))
            return
        super()._add_import(name)

    # NOTE(aaron): Fixed in https://github.com/sizmailov/pybind11-stubgen/pull/263
    def parse_annotation_str(self, annotation_str: str) -> ResolvedType | InvalidExpression | Value:
        result = super().parse_annotation_str(annotation_str)

        def handle_annotation(annotation: ResolvedType | InvalidExpression | Value) -> None:
            if isinstance(annotation, ResolvedType):
                self._add_import(annotation.name)
                if annotation.parameters is not None:
                    for p in annotation.parameters:
                        handle_annotation(p)

        handle_annotation(result)
        return result


def patch_lcmtype_imports() -> None:
    pybind11_stubgen.parser.mixins.fix.FixMissingImports = FixMissingImports  # type: ignore[misc]
    pybind11_stubgen.FixMissingImports = FixMissingImports  # type: ignore[misc]


def patch_current_module_prefix() -> None:
    """
    Fix use of the current module in nested types

    Could upstream
    """

    def parse_annotation_str(
        self: pybind11_stubgen.parser.mixins.fix.FixCurrentModulePrefixInTypeNames,
        annotation_str: str,
    ) -> ResolvedType | InvalidExpression | Value:
        result = super(  # type: ignore[safe-super]
            pybind11_stubgen.parser.mixins.fix.FixCurrentModulePrefixInTypeNames, self
        ).parse_annotation_str(annotation_str)

        def handle_annotation(annotation: ResolvedType | InvalidExpression | Value) -> None:
            if isinstance(annotation, ResolvedType):
                annotation.name = self._strip_current_module(annotation.name)
                if annotation.parameters is not None:
                    for p in annotation.parameters:
                        handle_annotation(p)

        handle_annotation(result)
        return result

    pybind11_stubgen.parser.mixins.fix.FixCurrentModulePrefixInTypeNames.parse_annotation_str = (  # type: ignore[method-assign]
        parse_annotation_str
    )


def patch_handle_docstring() -> None:
    """
    Patch BaseParser.handle_docstring to always strip empty lines from the start or end of
    docstrings
    """

    def handle_docstring(
        self: pybind11_stubgen.IParser, path: QualifiedName, value: T.Any
    ) -> T.Optional[Docstring]:
        if isinstance(value, str):
            assert isinstance(
                self, pybind11_stubgen.parser.mixins.parse.ExtractSignaturesFromPybind11Docstrings
            )
            return self._strip_empty_lines(value.splitlines())
        return None

    pybind11_stubgen.parser.mixins.parse.BaseParser.handle_docstring = handle_docstring  # type: ignore[method-assign]


def patch_fix_missing_none_hash_field_annotation() -> None:
    """
    See https://github.com/sizmailov/pybind11-stubgen/pull/236
    """

    def handle_field(
        self: pybind11_stubgen.parser.mixins.fix.FixMissingNoneHashFieldAnnotation,
        path: QualifiedName,
        field: T.Any,
    ) -> T.Optional[Field]:
        result = super(  # type: ignore[safe-super]
            pybind11_stubgen.parser.mixins.fix.FixMissingNoneHashFieldAnnotation, self
        ).handle_field(path, field)
        if result is None:
            return None
        if field is None and path[-1] == "__hash__":
            result.attribute.annotation = self.parse_annotation_str("typing.ClassVar[typing.Any]")
        return result

    pybind11_stubgen.parser.mixins.fix.FixMissingNoneHashFieldAnnotation.handle_field = handle_field  # type: ignore[method-assign]


def patch_numpy_annotations() -> None:
    class FixTypingTypeNames(pybind11_stubgen.parser.mixins.fix.FixTypingTypeNames):
        def _parse_annotation_str(
            self,
            result: ResolvedType | InvalidExpression | Value,
        ) -> ResolvedType | InvalidExpression | Value:
            if not isinstance(result, ResolvedType):
                return result

            result.parameters = (
                [self._parse_annotation_str(p) for p in result.parameters]
                if result.parameters is not None
                else None
            )

            if len(result.name) != 1:
                if result.name[0] == "typing" and result.name[1] in self.__typing_extensions_names:
                    result.name = QualifiedName.from_str(f"typing_extensions.{result.name[1]}")
                return result

            word = result.name[0]
            if word in self.__typing_names:
                package = "typing"
                if word in self.__typing_extensions_names:
                    package = "typing_extensions"
                result.name = QualifiedName.from_str(f"{package}.{word[0].upper()}{word[1:]}")
            if word == "function" and result.parameters is None:
                result.name = QualifiedName.from_str("typing.Callable")
            if word in {"object", "handle"} and result.parameters is None:
                result.name = QualifiedName.from_str("typing.Any")

            return result

    pybind11_stubgen.parser.mixins.fix.FixTypingTypeNames = FixTypingTypeNames  # type: ignore[misc]
    pybind11_stubgen.FixTypingTypeNames = FixTypingTypeNames  # type: ignore[misc]


class FixNumpyArrayRemoveParameters(pybind11_stubgen.IParser):
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
        print(result)
        if isinstance(result, ResolvedType) and result.name == QualifiedName.from_str(
            "typing.Annotated"
        ):
            assert (
                result.parameters is not None
                and len(result.parameters) >= 1
                and isinstance(result.parameters[0], ResolvedType)
            )
            print(f"Annotated: {result.parameters[0].name}")
            if result.parameters[0].name == self.__ndarray_name:
                print("Found ndarray")
                return result.parameters[0]
        return result


def patch_remove_parameters() -> None:
    """
    Fix NumpyArrayRemoveParameters to work with pybind 3.x and deduplicate overloads
    """

    pybind11_stubgen.parser.mixins.fix.FixNumpyArrayRemoveParameters = FixNumpyArrayRemoveParameters  # type: ignore[misc,assignment]
    pybind11_stubgen.FixNumpyArrayRemoveParameters = FixNumpyArrayRemoveParameters  # type: ignore[misc,assignment]
