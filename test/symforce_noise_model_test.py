# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import symforce.opt.noise_models as nm
import symforce.symbolic as sf
from symforce.ops import StorageOps
from symforce.test_util import TestCase
from symforce.test_util import epsilon_handling
from symforce.test_util import sympy_only


class NoiseModelTest(TestCase):
    def test_isotropic_noise_model(self) -> None:
        weight = 42
        noise_model = nm.IsotropicNoiseModel(weight)

        # Check whitening function
        unwhitened_residual = sf.V3(1, 2, 3)
        whitened_residual = StorageOps.evalf(noise_model.whiten(unwhitened_residual))
        self.assertEqual(StorageOps.evalf(sf.sqrt(weight) * unwhitened_residual), whitened_residual)

        # Check overall cost
        self.assertEqual(
            weight * unwhitened_residual.dot(unwhitened_residual) / 2,
            noise_model.error(unwhitened_residual),
        )

        # Check ops used for IsotropicNoiseModel.from_sigma; we should not get something like
        # x / sqrt(sigma^2)
        x = sf.V1.symbolic("x")
        sigma = sf.Symbol("sigma")
        model = nm.IsotropicNoiseModel.from_sigma(sigma)
        self.assertEqual(model.whiten(x), x / sigma)

        # Test other constructors
        noise_model_from_variance = nm.IsotropicNoiseModel.from_variance(1 / weight)
        noise_model_from_sigma = nm.IsotropicNoiseModel.from_sigma(1 / sf.sqrt(weight))
        self.assertEqual(
            StorageOps.evalf(noise_model_from_variance.whiten(unwhitened_residual)),
            whitened_residual,
        )
        self.assertEqual(
            StorageOps.evalf(noise_model_from_sigma.whiten(unwhitened_residual)), whitened_residual
        )

    def test_diagonal_noise_model(self) -> None:
        weights = [1, 2, 3]
        noise_model = nm.DiagonalNoiseModel(weights)

        # Check whitening function
        unwhitened_residual = sf.V3(4, 5, 6)
        whitened_residual = StorageOps.evalf(noise_model.whiten(unwhitened_residual))
        expected_whitened_residual = sf.M(
            [sf.sqrt(w) * v for w, v in zip(weights, unwhitened_residual)]
        )
        self.assertEqual(StorageOps.evalf(expected_whitened_residual), whitened_residual)

        # Check overall cost
        self.assertEqual(
            expected_whitened_residual.dot(expected_whitened_residual) / 2,
            noise_model.error(unwhitened_residual),
        )

        # Check ops used for DiagonalNoiseModel.from_sigma; we should not get something like
        # x / sqrt(sigma^2)
        sigmas = sf.V3.symbolic("sigma").to_storage()
        x = sf.V3.symbolic("x")
        model = nm.DiagonalNoiseModel.from_sigmas(sigmas)
        for i in range(3):
            self.assertEqual(model.whiten(x)[i], x[i] / sigmas[i])

        # Test other constructors
        noise_model_from_variance = nm.DiagonalNoiseModel.from_variances([1 / w for w in weights])
        noise_model_from_sigma = nm.DiagonalNoiseModel.from_sigmas(
            [1 / sf.sqrt(w) for w in weights]
        )
        self.assertEqual(
            StorageOps.evalf(noise_model_from_variance.whiten(unwhitened_residual)),
            whitened_residual,
        )
        self.assertEqual(
            StorageOps.evalf(noise_model_from_sigma.whiten(unwhitened_residual)), whitened_residual
        )

    def test_pseudo_huber_noise_model(self) -> None:
        """
        Tests the residual and jacobian values of the pseudo-huber noise model
        """
        delta = 10.0
        scalar_information = sf.Symbol("scalar_information")
        scalar_information_num = 3.0
        epsilon = 1.0e-8
        noise_model = nm.PseudoHuberNoiseModel(
            delta=delta, scalar_information=scalar_information, epsilon=epsilon
        )

        # Check the jacobian at points we care about
        unwhitened_residual = sf.V3.symbolic("res")
        error = sf.V1(noise_model.error(unwhitened_residual)).subs(
            scalar_information, scalar_information_num
        )
        jacobian = error.jacobian(unwhitened_residual)

        # Jacobian should be zero when the residual is zero
        self.assertEqual(jacobian.subs(unwhitened_residual, sf.V3.zero()), sf.M13.zero())

        # Negating residual should flip the sign of the jacobian because the error function should
        # by symmetric
        unwhitened_residual_numeric = sf.V3(0.1, -0.2, 0.3)
        self.assertEqual(
            jacobian.subs(unwhitened_residual, unwhitened_residual_numeric),
            -jacobian.subs(unwhitened_residual, -unwhitened_residual_numeric),
        )

        # Jacobian should be constant for large values because the tails of the error function
        # should be linear
        large_unwhitened_residual = sf.V3(10000.0, 20000.0, 30000.0)
        self.assertStorageNear(
            jacobian.subs(unwhitened_residual, large_unwhitened_residual),
            jacobian.subs(unwhitened_residual, 2 * large_unwhitened_residual),
            places=4,
        )

        # Error should behave like the L2 loss for small values
        small_unwhitened_residual = sf.V3(0.01, -0.02, 0.03)
        self.assertStorageNear(
            error.subs(unwhitened_residual, 3.0 * small_unwhitened_residual),
            9.0 * error.subs(unwhitened_residual, small_unwhitened_residual),
            places=4,
        )

    @sympy_only
    def test_pseudo_huber_epsilon_handling(self) -> None:
        """
        Epsilon handling for PseudoHuberNoiseModel.whiten_norm should be correct (i.e. value and
        derivative at 0 should be correct as epsilon->0)
        """
        delta = 10
        scalar_information = 2

        def whiten_ratio(x: sf.Scalar, epsilon: sf.Scalar) -> sf.Scalar:
            noise_model = nm.PseudoHuberNoiseModel(
                delta=delta, scalar_information=scalar_information, epsilon=epsilon
            )
            return noise_model.whiten_norm(sf.V1(x), epsilon)[0, 0]

        self.assertTrue(epsilon_handling.is_epsilon_correct(whiten_ratio, expected_value=0))

    def test_barron_noise_model(self) -> None:
        """
        Some simple tests on the Barron noise model.
        """
        x = sf.Symbol("x")
        x_matrix = sf.V1(x)

        alpha = 1.0
        delta = 2.0
        scalar_information = 1.0
        epsilon = 1.0e-6

        noise_model = nm.BarronNoiseModel(
            alpha=alpha, delta=delta, scalar_information=scalar_information, x_epsilon=epsilon
        )
        error = sf.V1(noise_model.error(x_matrix))
        jac = error.jacobian(x_matrix)

        # Test 0: Derivative should be 0 at 0
        self.assertStorageNear(jac.subs(x, 0.0).evalf(), 0)

        # Test 1: Derivative should be symmetric.
        test1a = jac.subs(x, 1.0).evalf()
        test1b = jac.subs(x, -1.0).evalf()
        self.assertStorageNear(test1a, -test1b)

        # Test 2: Derivative should be constant for large values w/ alpha==1,
        # this should be pseudo-huber
        test2a = jac.subs(x, 1000.0).evalf()
        test2b = jac.subs(x, 2000.0).evalf()
        self.assertStorageNear(test2a, test2b, places=4)

        # Test 3: for alpha=-inf, it should asymptote at delta^2
        alpha = -1.0e10
        noise_model = nm.BarronNoiseModel(
            alpha=alpha, delta=delta, scalar_information=scalar_information, x_epsilon=epsilon
        )
        error = sf.V1(noise_model.error(x_matrix))
        jac = error.jacobian(x_matrix)

        test3a = error.subs(x, 1000.0).evalf()
        self.assertStorageNear(test3a, delta**2, places=2)

        test3b = jac.subs(x, 1000.0).evalf()
        self.assertStorageNear(test3b, 0.0)

        # Test 4: the residual gradient w/ 0 weight should be 0 (and finite!)
        # Make all the params symbolic so they don't get removed.
        alpha, delta, scalar_information, epsilon = sf.symbols("alpha, scale, weight, epsilon")
        noise_model = nm.BarronNoiseModel(
            alpha, delta=delta, scalar_information=scalar_information, x_epsilon=epsilon
        )
        whitened_residual = noise_model.whiten(x_matrix)
        jac = whitened_residual.jacobian(x_matrix)
        test4 = jac.subs(
            {alpha: 1.0, delta: 2.0, scalar_information: 0.0, epsilon: 1e-10, x: 0.0}
        ).evalf()
        self.assertStorageNear(test4, 0.0)

    @sympy_only
    def test_barron_noise_model_epsilon_handling(self) -> None:
        """
        Epsilon handling for BarronNoiseModel.whiten_norm should be correct (i.e. value and
        derivative at 0 should be correct as epsilon->0)
        """
        delta = 10
        scalar_information = 2

        def test_epsilon_at_alpha(alpha: sf.Scalar) -> bool:
            def whiten_ratio(x: sf.Scalar, epsilon: sf.Scalar) -> sf.Scalar:
                noise_model = nm.BarronNoiseModel(
                    alpha=alpha,
                    delta=delta,
                    scalar_information=scalar_information,
                    alpha_epsilon=epsilon,
                    x_epsilon=epsilon,
                )
                return noise_model.whiten_norm(sf.V1(x), epsilon)[0, 0]

            # SymPy fails to calculate the limits correctly here, so we provide the correct answers
            return epsilon_handling.is_epsilon_correct(
                whiten_ratio, expected_value=0, expected_derivative=sf.sqrt(scalar_information)
            )

        self.assertTrue(test_epsilon_at_alpha(1))
        self.assertTrue(test_epsilon_at_alpha(2))
        self.assertTrue(test_epsilon_at_alpha(int(-1e7)))


if __name__ == "__main__":
    TestCase.main()
