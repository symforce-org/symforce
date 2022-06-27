# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import symforce.symbolic as sf
from symforce import typing as T
from symforce.test_util import TestCase
from symforce.test_util.storage_ops_test_mixin import StorageOpsTestMixin


class GeoSequenceTest(StorageOpsTestMixin, TestCase):
    """
    Test a scalar as a geometric class.
    Note the mixin that tests all storage, group and lie group ops.
    """

    @classmethod
    def element(cls) -> T.List[sf.Rot3]:
        element = []
        element.append(sf.Rot3())
        element.append(sf.Rot3.from_yaw_pitch_roll(1.0, 0, 0))
        return element


if __name__ == "__main__":
    TestCase.main()
