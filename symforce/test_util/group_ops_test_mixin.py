from symforce import typing as T
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
        self.assertNear(identity, GroupOps.inverse(identity))

        # Composition with identity
        self.assertNear(element, GroupOps.compose(element, identity))
        self.assertNear(element, GroupOps.compose(identity, element))

        # Composition with inverse
        self.assertNear(identity, GroupOps.compose(GroupOps.inverse(element), element))
        self.assertNear(identity, GroupOps.compose(element, GroupOps.inverse(element)))

        # Between for differencing
        self.assertNear(identity, GroupOps.between(element, element))
        self.assertNear(element, GroupOps.between(identity, element))
        self.assertNear(GroupOps.inverse(element), GroupOps.between(element, identity))
