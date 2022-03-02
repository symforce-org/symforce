# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

from symforce.ops import GroupOps

from .storage_ops_test_mixin import StorageOpsTestMixin


class GroupOpsTestMixin(StorageOpsTestMixin):
    """
    Test helper for the GroupOps concept. Inherit a test case from this.
    """

    def test_group_ops(self) -> None:
        """
        Tests:
            identity
            inverse
            compose
            between
        """
        # Create an identity and non-identity element
        element = self.element()
        identity = GroupOps.identity(element)
        self.assertNotEqual(identity, element, ".element() must be non-identity type")

        # Basic equality
        self.assertEqual(identity, identity)
        self.assertEqual(element, element)

        # Inverse of identity is identity
        self.assertStorageNear(identity, GroupOps.inverse(identity))

        # Composition with identity
        self.assertStorageNear(element, GroupOps.compose(element, identity))
        self.assertStorageNear(element, GroupOps.compose(identity, element))

        # Composition with inverse
        self.assertStorageNear(identity, GroupOps.compose(GroupOps.inverse(element), element))
        self.assertStorageNear(identity, GroupOps.compose(element, GroupOps.inverse(element)))

        # Between for differencing
        self.assertStorageNear(identity, GroupOps.between(element, element))
        self.assertStorageNear(element, GroupOps.between(identity, element))
        self.assertStorageNear(GroupOps.inverse(element), GroupOps.between(element, identity))
