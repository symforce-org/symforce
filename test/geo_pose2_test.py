# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import numpy as np

from symforce import geo
from symforce import sympy as sm
from symforce.ops import LieGroupOps
from symforce.test_util import TestCase
from symforce.test_util.lie_group_ops_test_mixin import LieGroupOpsTestMixin


class GeoPose2Test(LieGroupOpsTestMixin, TestCase):
    """
    Test the Pose2 geometric class.
    Note the mixin that tests all storage, group and lie group ops.
    """

    MANIFOLD_IS_DEFINED_IN_TERMS_OF_GROUP_OPS = False

    @classmethod
    def element(cls) -> geo.Pose2:
        return geo.Pose2.from_tangent([-0.2, 5.3, 1.2])

    def pose2_operations(self, a: geo.Pose2, b: geo.Pose2) -> None:
        """
        Tests Pose2 operations
        """
        self.assertEqual(a * b, a.compose(b))
        self.assertEqual(a * b.t, a.R * b.t + a.t)

    def test_pose2_operations_numeric(self) -> None:
        """
        Tests:
            Pose2.__mul__
        """
        R_a = geo.Rot2.random()
        t_a = geo.V2(np.random.rand(2))
        a = geo.Pose2(R_a, t_a)

        R_b = geo.Rot2.random()
        t_b = geo.V2(np.random.rand(2))
        b = geo.Pose2(R_b, t_b)

        self.pose2_operations(a, b)

    def test_pose2_operations_symbolic(self) -> None:
        """
        Tests:
            Pose2.__mul__
        """
        a = geo.Pose2.symbolic("a")
        b = geo.Pose2.symbolic("b")
        self.pose2_operations(a, b)

    def test_translation_rotation_independence(self) -> None:
        """
        Tests that the rotation component of the tangent does not change translation
        """
        element = self.element()
        tangent_vec = [1.0] * 3

        value = LieGroupOps.from_tangent(element, tangent_vec, epsilon=self.EPSILON)
        self.assertStorageNear(value.t, tangent_vec[:2], places=7)


if __name__ == "__main__":
    TestCase.main()
