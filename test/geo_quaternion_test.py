# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import symforce.symbolic as sf
from symforce.test_util import TestCase
from symforce.test_util.group_ops_test_mixin import GroupOpsTestMixin


class GeoQuaternionTest(GroupOpsTestMixin, TestCase):
    """
    Test the Quaternion geometric class.
    Note the mixin that tests all storage and group ops.
    """

    @classmethod
    def element(cls) -> sf.Quaternion:
        return sf.Quaternion(xyz=sf.V3(0.1, -0.3, 1.3), w=3.2)


if __name__ == "__main__":
    TestCase.main()
