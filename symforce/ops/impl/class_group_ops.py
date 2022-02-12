# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

from symforce import typing as T

from .class_storage_ops import ClassStorageOps


class ClassGroupOps(ClassStorageOps):
    @staticmethod
    def identity(a: T.ElementOrType) -> T.Element:
        return a.identity()

    @staticmethod
    def compose(a: T.Element, b: T.Element) -> T.Element:
        return a.compose(b)

    @staticmethod
    def inverse(a: T.Element) -> T.Element:
        return a.inverse()
