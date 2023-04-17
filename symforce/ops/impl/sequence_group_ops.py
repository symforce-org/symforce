# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

from symforce import typing as T
from symforce.ops import GroupOps
from symforce.typing_util import get_type

from .sequence_storage_ops import SequenceStorageOps


class SequenceGroupOps(SequenceStorageOps):
    @staticmethod
    def identity(a: T.SequenceElement) -> T.SequenceElement:
        return get_type(a)([GroupOps.identity(v) for v in a])

    @staticmethod
    def compose(a: T.SequenceElement, b: T.SequenceElement) -> T.SequenceElement:
        return get_type(a)([GroupOps.compose(v_a, v_b) for v_a, v_b in zip(a, b)])

    @staticmethod
    def inverse(a: T.SequenceElement) -> T.SequenceElement:
        return get_type(a)([GroupOps.inverse(v) for v in a])
