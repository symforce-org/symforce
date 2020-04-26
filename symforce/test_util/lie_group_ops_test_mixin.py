import numpy as np

from symforce import geo
from symforce.ops import LieGroupOps

from .group_ops_test_mixin import GroupOpsTestMixin


class LieGroupOpsTestMixin(GroupOpsTestMixin):
    """
    Test helper for the LieGroupOps concept. Inherit a test case from this.
    """

    # Small number to avoid singularities
    EPSILON = 1e-8

    def test_lie_group_ops(self):
        """
        Tests:
            tangent_dim
            from_tangent
            to_tangent
            retract
            local_coordinates
        """
        # Create an identity and non-identity element
        element = self.element()
        identity = LieGroupOps.identity(element)

        # Check manifold dimension
        dim = LieGroupOps.tangent_dim(element)
        self.assertEqual(dim, LieGroupOps.tangent_dim(identity))
        self.assertGreater(dim, 0)

        # Manifold dimension must be less than or equal to storage dim
        self.assertLessEqual(dim, LieGroupOps.storage_dim(identity))

        # Construct from a tangent space pertubation around identity
        pertubation = list(np.random.normal(scale=0.1, size=(dim,)))
        value = LieGroupOps.from_tangent(element, pertubation, epsilon=self.EPSILON)
        self.assertEqual(type(value), self.element_type())

        # Map back to the tangent space
        recovered_pertubation = LieGroupOps.to_tangent(value, epsilon=self.EPSILON)
        self.assertEqual(type(recovered_pertubation), list)

        # Assert we are close (near epsilon) to the original
        self.assertNear(pertubation, recovered_pertubation, places=6)

        # Element from zero tangent vector is identity
        identity_actual = LieGroupOps.from_tangent(
            element, geo.M.zeros(dim, 1), epsilon=self.EPSILON
        )
        self.assertNear(identity, identity_actual, places=7)

        # Tangent vector of identity element is zero
        tangent_zero_actual = LieGroupOps.to_tangent(identity, epsilon=self.EPSILON)
        self.assertNear(tangent_zero_actual, geo.M.zeros(dim, 1), places=7)

        # Test retraction behaves as expected (compose and from_tangent)
        retracted_element = LieGroupOps.retract(element, pertubation, epsilon=self.EPSILON)
        self.assertNear(retracted_element, LieGroupOps.compose(element, value), places=7)

        # Test zero retraction
        element_actual = LieGroupOps.retract(element, geo.M.zeros(dim, 1), epsilon=self.EPSILON)
        self.assertNear(element_actual, element, places=7)

        # Test local_coordinates behaves as expected (between and to_tangent)
        pertubation_recovered = LieGroupOps.local_coordinates(
            element, retracted_element, epsilon=self.EPSILON
        )
        diff_element = LieGroupOps.between(element, retracted_element)
        self.assertNear(
            LieGroupOps.to_tangent(diff_element, epsilon=self.EPSILON),
            pertubation_recovered,
            places=7,
        )

        # Test that it recovers the original pertubation
        self.assertNear(pertubation, pertubation_recovered, places=6)

        # Test an identity local coordinates
        self.assertNear(
            LieGroupOps.local_coordinates(element, element, epsilon=self.EPSILON),
            geo.M.zeros(dim, 1),
            places=7,
        )
