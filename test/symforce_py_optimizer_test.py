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
from symforce.opt.factor import Factor
from symforce.opt.optimizer import Optimizer
from symforce.test_util import TestCase
from symforce.values import Values


class SymforcePyOptimizerTest(TestCase):
    """
    Test the symforce optimizer in Python.
    """

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
        logger.debug(f"Num iterations: {len(result.iteration_stats)}")
        logger.debug(f"Final error: {result.error()}")

        # Check values
        self.assertEqual(len(result.iteration_stats), 7)
        self.assertAlmostEqual(result.error(), 0.039, places=3)

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
            len(Factor._generated_residual_cache), 2  # pylint: disable=protected-access
        )

        index_entry2 = optimizer.linearization_index_entry("x1")
        self.assertEqual(index_entry, index_entry2)


if __name__ == "__main__":
    TestCase.main()
