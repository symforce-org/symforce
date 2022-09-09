# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import numpy as np

import symforce.internal.symbolic as sf
from symforce import typing as T

from .class_lie_group_ops import ClassLieGroupOps


class SymClassLieGroupOps(ClassLieGroupOps):
    @staticmethod
    def from_tangent(a: T.ElementOrType, vec: T.List[T.Scalar], epsilon: T.Scalar) -> T.Element:
        return a.from_tangent(np.array(vec), epsilon)

    @staticmethod
    def to_tangent(a: T.Element, epsilon: T.Scalar) -> T.List[T.Scalar]:
        return a.to_tangent(epsilon).flatten().tolist()

    @staticmethod
    def retract(a: T.Element, vec: T.Sequence[T.Scalar], epsilon: T.Scalar) -> T.Element:
        return a.retract(np.array(vec), epsilon)

    @staticmethod
    def local_coordinates(
        a: T.Element, b: T.Element, epsilon: T.Scalar = sf.epsilon()
    ) -> T.List[T.Scalar]:
        return a.local_coordinates(b, epsilon).flatten().tolist()
