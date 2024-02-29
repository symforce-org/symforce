# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

from symforce import typing as T
from symforce.ops.ops import OpNotImplementedError
from symforce.ops.ops import Ops
from symforce.test_util import TestCase


class SymforceOpsTest(TestCase):
    """
    Test the Ops class.
    """

    def test_implementation_no_register(self) -> None:
        """
        Tests:
            Ops.implementation
        Check that Ops.implementation raises a NotImplementedError if
        argument has not been registered.
        """
        with self.assertRaises(OpNotImplementedError):
            Ops.implementation(type("UnregisteredType", (object,), {}))

    @staticmethod
    def get_implementation_type() -> T.Type:
        # A helper to make testing more concise
        return type("ImplementationType", (object,), {})

    def test_implementation_after_register(self) -> None:
        """
        Tests:
            Ops.register
            Ops.implementation
        Check that Ops.implementation returns the implementation type it was
        registered with.
        """
        RegisteredType = type("RegisteredType", (object,), {})
        ImplementationType = self.get_implementation_type()

        Ops.register(RegisteredType, ImplementationType)
        self.assertEqual(ImplementationType, Ops.implementation(RegisteredType))

    def assert_implementation_raises(self, reg_cls: T.Type, impl_type: T.Type) -> None:
        # A helper to make testing more concise
        with self.assertRaises(OpNotImplementedError):
            reg_cls.implementation(impl_type)

    def assert_implementation_returns(self, reg_cls: T.Type, impl_type: T.Type) -> None:
        # A helper to make testing more concise
        try:
            reg_cls.implementation(impl_type)
        except NotImplementedError:
            self.fail("Ops.implementation raised NotImplementedError unexpectedly")

    @staticmethod
    def get_type_parent_child() -> T.Tuple[T.Type, T.Type]:
        # A helper to make testing more concise
        TypeParent = type("TypeParent", (object,), {})
        return TypeParent, type("TypeChild", (TypeParent,), {})

    @staticmethod
    def get_ops_parent_child() -> T.Tuple[T.Type, T.Type]:
        # A helper to make testing more concise
        OpsParent = type("OpsParent", (Ops,), {})
        return OpsParent, type("OpsChild", (OpsParent,), {})

    def test_implementation_exception_parents(self) -> None:
        """
        Tests:
            Ops.implementation
        Check that implementation raises correctly when parent type is registered to parent Ops
        """
        TypeParent, TypeChild = self.get_type_parent_child()
        OpsParent, OpsChild = self.get_ops_parent_child()
        ImplementationType = self.get_implementation_type()

        OpsParent.register(TypeParent, ImplementationType)

        self.assert_implementation_returns(OpsParent, TypeParent)
        self.assert_implementation_raises(OpsChild, TypeParent)
        self.assert_implementation_returns(OpsParent, TypeChild)
        self.assert_implementation_raises(OpsChild, TypeChild)

    def test_implementation_exception_children(self) -> None:
        """
        Tests:
            Ops.implementation
        Check that implementation raises correctly when child type is registered to child Ops
        """
        TypeParent, TypeChild = self.get_type_parent_child()
        OpsParent, OpsChild = self.get_ops_parent_child()
        ImplementationType = self.get_implementation_type()

        OpsChild.register(TypeChild, ImplementationType)

        self.assert_implementation_raises(OpsParent, TypeParent)
        self.assert_implementation_raises(OpsChild, TypeParent)
        self.assert_implementation_returns(OpsParent, TypeChild)
        self.assert_implementation_returns(OpsChild, TypeChild)

    def test_implementation_exception_child_parent(self) -> None:
        """
        Tests:
            Ops.implementation
        Check that implementation raises correctly when child type is registered to parent Ops
        """
        TypeParent, TypeChild = self.get_type_parent_child()
        OpsParent, OpsChild = self.get_ops_parent_child()
        ImplementationType = self.get_implementation_type()

        OpsParent.register(TypeChild, ImplementationType)

        self.assert_implementation_raises(OpsParent, TypeParent)
        self.assert_implementation_raises(OpsChild, TypeParent)
        self.assert_implementation_returns(OpsParent, TypeChild)
        self.assert_implementation_raises(OpsChild, TypeChild)

    def test_implementation_exception_parent_child(self) -> None:
        """
        Tests:
            Ops.implemenation
        Check that implementation raises correctly when child type is registered to child Ops
        """
        TypeParent, TypeChild = self.get_type_parent_child()
        OpsParent, OpsChild = self.get_ops_parent_child()
        ImplementationType = self.get_implementation_type()

        OpsChild.register(TypeParent, ImplementationType)

        self.assert_implementation_returns(OpsParent, TypeParent)
        self.assert_implementation_returns(OpsChild, TypeParent)
        self.assert_implementation_returns(OpsParent, TypeChild)
        self.assert_implementation_returns(OpsChild, TypeChild)

    def test_implementation_overregister_inherited_implementation(self) -> None:
        """
        Tests:
            Ops.Implementation
        Check that when an inherited implementation type is over registered, that
        the correct implementation is returned
        """
        TypeParent, TypeChild = self.get_type_parent_child()
        TypeGrandChild = type("TypeGrandChild", (TypeChild,), {})
        OpsParent, OpsChild = self.get_ops_parent_child()
        ImplementationType1 = self.get_implementation_type()
        ImplementationType2 = self.get_implementation_type()

        OpsParent.register(TypeParent, ImplementationType1)
        OpsChild.register(TypeChild, ImplementationType2)

        # Checking TypeParent gets the correct implementations
        self.assertEqual(OpsParent.implementation(TypeParent), ImplementationType1)
        with self.assertRaises(OpNotImplementedError):
            OpsChild.implementation(TypeParent)

        # Checking TypeChild gets the correct implementations
        self.assertEqual(OpsParent.implementation(TypeChild), ImplementationType2)
        self.assertEqual(OpsChild.implementation(TypeChild), ImplementationType2)

        # Checking that TypeGrandChild gets the correct implementation
        self.assertEqual(OpsParent.implementation(TypeGrandChild), ImplementationType2)
        self.assertEqual(OpsChild.implementation(TypeGrandChild), ImplementationType2)

    def test_implementation_multiple_registration_inheritance(self) -> None:
        """
        Tests:
            Ops.Implementation
        Check that when a type inherites from two classes, each registered to a different
        implementation, that the correct implementation is returned
        """
        # Setup
        TypeParent1 = type("TypeParent1", (object,), {})
        TypeParent2 = type("TypeParent2", (object,), {})
        TypeChild = type("TypeChild", (TypeParent1, TypeParent2), {})

        OpsParent: T.Type = type("OpsParent", (Ops,), {})
        OpsChild1: T.Type = type("OpsChild1", (OpsParent,), {})
        OpsChild2: T.Type = type("OpsChild2", (OpsParent,), {})

        ImplementationType1 = self.get_implementation_type()
        ImplementationType2 = self.get_implementation_type()

        OpsChild1.register(TypeParent1, ImplementationType1)
        OpsChild2.register(TypeParent2, ImplementationType2)

        self.assertEqual(ImplementationType1, OpsChild1.implementation(TypeChild))
        self.assertEqual(ImplementationType2, OpsChild2.implementation(TypeChild))

        # Note, TypeParent1 is the first class after TypeChild in TypeChild's mro
        self.assertEqual(ImplementationType1, OpsParent.implementation(TypeChild))


if __name__ == "__main__":
    TestCase.main()
