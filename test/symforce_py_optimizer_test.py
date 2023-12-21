# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import symforce

symforce.set_epsilon_to_symbol()

from lcmtypes.sym._index_entry_t import index_entry_t
from lcmtypes.sym._key_t import key_t
from lcmtypes.sym._type_t import type_t

import symforce.symbolic as sf
from symforce import logger
from symforce import typing as T
from symforce.opt._internal.generated_residual_cache import GeneratedResidualCache
from symforce.opt.factor import Factor
from symforce.opt.optimizer import Optimizer
from symforce.test_util import TestCase
from symforce.values import Values


class SymforcePyOptimizerTest(TestCase):
    """
    Test the symforce optimizer in Python.
    """

    def setUp(self) -> None:
        super().setUp()

        # Clear the residual cache before each test
        Factor._generated_residual_cache = (  # pylint: disable=protected-access
            GeneratedResidualCache()
        )

    def test_rotation_smoothing(self) -> None:
        """
        Optimize a chain of 3D orientations with prior and between factors.
        """
        num_samples = 10
        xs = [f"x{i}" for i in range(num_samples)]
        x_priors = [f"x_prior{i}" for i in range(num_samples)]

        factors = []

        ### Between factors

        def between(x: sf.Rot3, y: sf.Rot3, epsilon: sf.Scalar) -> sf.V3:
            return sf.V3(x.local_coordinates(y, epsilon=epsilon))

        for i in range(num_samples - 1):
            factors.append(Factor(keys=[xs[i], xs[i + 1], "epsilon"], residual=between))

        ### Prior factors

        def prior_residual(x: sf.Rot3, epsilon: sf.Scalar, x_prior: sf.Rot3) -> sf.V3:
            return sf.V3(x.local_coordinates(x_prior, epsilon=epsilon))

        for i in range(num_samples):
            factors.append(
                Factor(keys=[xs[i], "epsilon", x_priors[i]], name="prior", residual=prior_residual)
            )

        # Create the optimizer
        optimizer = Optimizer(factors=factors, optimized_keys=xs)

        # Create initial values
        initial_values = Values(epsilon=sf.numeric_epsilon)
        for i in range(num_samples):
            initial_values[xs[i]] = sf.Rot3.from_yaw_pitch_roll(yaw=0.0, pitch=0.1 * i, roll=0.0)
        for i in range(num_samples):
            initial_values[x_priors[i]] = sf.Rot3.from_yaw_pitch_roll(roll=0.1 * i)

        result = optimizer.optimize(initial_values)

        logger.debug(f"Initial values: {result.initial_values}")
        logger.debug(f"Optimized values: {result.optimized_values}")
        logger.debug(f"Num iterations: {len(result.iterations)}")
        logger.debug(f"Final error: {result.error()}")
        logger.debug(f"Status: {result.status}")

        # Check values
        self.assertEqual(len(result.iterations), 7)
        self.assertAlmostEqual(result.error(), 0.039, places=3)
        self.assertEqual(result.status, Optimizer.Status.SUCCESS)
        self.assertEqual(result.failure_reason, Optimizer.FailureReason.INVALID)

        # Check that we can pull out the variable blocks
        index_entry = optimizer.linearization_index()["x1"]
        self.assertEqual(
            index_entry,
            index_entry_t(
                # The key here is an implementation detail, so just copy the key that we got, and
                # check below that it's not empty
                key=index_entry.key,
                type=type_t.ROT3,
                offset=3,
                storage_dim=4,
                tangent_dim=3,
            ),
        )
        self.assertNotEqual(index_entry.key, key_t())

        # Check that the factor cache has the expected number of entries
        self.assertEqual(
            len(Factor._generated_residual_cache),  # pylint: disable=protected-access
            2,
        )

        index_entry2 = optimizer.linearization_index_entry("x1")
        self.assertEqual(index_entry, index_entry2)

    def test_unoptimized_factor_exception(self) -> None:
        """
        Tests that a ValueError is raised if none of the factor keys match the optimizer keys.
        """

        def residual(x: T.Scalar) -> sf.V1:
            return sf.V1((x - 2.71828) ** 2)

        factor1 = Factor(["present_key"], residual)
        factor2 = Factor(["absent_key"], residual)

        optimized_keys = ["present_key", "other_present_key"]
        with self.assertRaises(ValueError):
            try:
                Optimizer([factor1, factor2], optimized_keys)
            except ValueError as err:
                self.assertIn(str(factor2.keys), str(err))
                self.assertIn(str(optimized_keys), str(err))
                raise err from None

    def test_factor_keys_ordering(self) -> None:
        """
        Adding the same residual with different key orderings should work correctly

        https://github.com/symforce-org/symforce/issues/300
        """

        def prior(x: T.Scalar) -> sf.V1:
            return sf.V1(x)

        def between(x: T.Scalar, y: T.Scalar, b: T.Scalar) -> sf.V1:
            return sf.V1(y - x - b)

        initial_values = Values(x0=0.0, x1=0.0, x2=0.0, b01=1.0, b12=1.0, b20=-2.0)

        factors = []

        factors.append(Factor(keys=["x0"], residual=prior))
        factors.append(Factor(keys=["x0", "x1", "b01"], residual=between))
        factors.append(Factor(keys=["x1", "x2", "b12"], residual=between))
        factors.append(Factor(keys=["x2", "x0", "b20"], residual=between))

        optimizer = Optimizer(factors=factors, optimized_keys=["x0", "x1", "x2"])

        result = optimizer.optimize(initial_values)

        self.assertEqual(result.status, Optimizer.Status.SUCCESS)
        self.assertLess(result.error(), 1e-6)

        # Check that the factor cache has the expected number of entries
        self.assertEqual(
            len(Factor._generated_residual_cache),  # pylint: disable=protected-access
            2,
        )


if __name__ == "__main__":
    TestCase.main()
