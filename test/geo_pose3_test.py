# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import numpy as np

import symforce.symbolic as sf
from symforce.ops import LieGroupOps
from symforce.test_util import TestCase
from symforce.test_util.lie_group_ops_test_mixin import LieGroupOpsTestMixin


class GeoPose3Test(LieGroupOpsTestMixin, TestCase):
    """
    Test the Pose3 geometric class.
    Note the mixin that tests all storage, group and lie group ops.
    """

    MANIFOLD_IS_DEFINED_IN_TERMS_OF_GROUP_OPS = False

    @classmethod
    def element(cls) -> sf.Pose3:
        return sf.Pose3.from_tangent([1.3, 0.2, 1.1, -0.2, 5.3, 1.2])

    def test_pose3_accessors(self) -> None:
        """
        Tests additional accessors
        """
        element = self.element()

        self.assertEqual(element.R, element.rotation())
        self.assertEqual(element.t, element.position())

    def pose3_operations(self, a: sf.Pose3, b: sf.Pose3) -> None:
        """
        Tests Pose3 operations
        """
        self.assertEqual(a * b, a.compose(b))
        self.assertEqual(a * b.t, a.R * b.t + a.t)

    def test_pose3_operations_numeric(self) -> None:
        """
        Tests (numeric):
            Pose3.__mul__
        """
        R_a = sf.Rot3.random()
        t_a = sf.V3(np.random.rand(3))
        a = sf.Pose3(R_a, t_a)

        R_b = sf.Rot3.random()
        t_b = sf.V3(np.random.rand(3))
        b = sf.Pose3(R_b, t_b)

        self.pose3_operations(a, b)

    def test_pose3_operations_symbolic(self) -> None:
        """
        Tests (symbolic):
            Pose3.__mul__
        """
        a = sf.Pose3.symbolic("a")
        b = sf.Pose3.symbolic("b")
        self.pose3_operations(a, b)

    def test_translation_rotation_independence(self) -> None:
        """
        Tests that the rotation component of the tangent does not change translation
        """
        element = self.element()
        tangent_vec = [1.0] * 6

        value = LieGroupOps.from_tangent(element, tangent_vec, epsilon=self.EPSILON)
        self.assertStorageNear(value.t, tangent_vec[3:], places=7)


if __name__ == "__main__":
    TestCase.main()
