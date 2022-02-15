# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import numpy as np

from symforce.ops import GroupOps
from symforce import typing as T

from .array_storage_ops import ArrayStorageOps


class ArrayGroupOps(ArrayStorageOps):
    """
    Implements Group operations for numpy ndarrays.
    """

    @staticmethod
    def identity(a: T.ArrayElement) -> T.ArrayElement:
        return np.array(([GroupOps.identity(v) for v in a])).reshape(a.shape)

    @staticmethod
    def compose(a: T.ArrayElement, b: T.ArrayElement) -> T.ArrayElement:
        return np.array(([GroupOps.compose(v_a, v_b) for v_a, v_b in zip(a, b)])).reshape(a.shape)

    @staticmethod
    def inverse(a: T.ArrayElement) -> T.ArrayElement:
        return np.array(([GroupOps.inverse(v) for v in a])).reshape(a.shape)
