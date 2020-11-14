from symforce import geo
import symforce.opt.noise_models as nm
from symforce import sympy as sm
from symforce import types as T
from symforce.test_util import TestCase, requires_sympy, epsilon_handling


class NoiseModelTest(TestCase):
    def test_barron_noise_model(self):
        # type: () -> None
        """
        Some simple tests on the Barron noise model.
        """
        x = sm.Symbol("x")
        x_matrix = geo.V1(x)

        alpha = 1.0
        scale = 2.0
        weight = 1.0
        epsilon = 1.0e-6

        noise_model = nm.BarronNoiseModel(alpha, scale, weight, epsilon)
        error = geo.V1(noise_model.error(x_matrix))
        jac = error.jacobian(x_matrix)

        # Test 0: Derivative should be 0 at 0
        self.assertNear(jac.subs(x, 0.0).evalf(), 0)

        # Test 1: Derivative should be symmetric.
        test1a = jac.subs(x, 1.0).evalf()
        test1b = jac.subs(x, -1.0).evalf()
        self.assertNear(test1a, -test1b)

        # Test 2: Derivative should be constant for large values w/ alpha==1,
        # this should be pseudo-huber
        test2a = jac.subs(x, 1000.0).evalf()
        test2b = jac.subs(x, 2000.0).evalf()
        self.assertNear(test2a, test2b, places=4)

        # Test 3: for alpha=-inf, it should asymptote at 1
        alpha = -1.0e10
        noise_model = nm.BarronNoiseModel(alpha, scale, weight, epsilon)
        error = geo.V1(noise_model.error(x_matrix))
        jac = error.jacobian(x_matrix)

        test3a = error.subs(x, 1000.0).evalf()
        self.assertNear(test3a, 1.0, places=2)

        test3b = jac.subs(x, 1000.0).evalf()
        self.assertNear(test3b, 0.0)

        # Test 4: the residual gradient w/ 0 weight should be 0 (and finite!)
        # Make all the params symbolic so they don't get removed.
        alpha, scale, weight, epsilon = sm.symbols("alpha, scale, weight, epsilon")
        noise_model = nm.BarronNoiseModel(alpha, scale, weight, epsilon)
        whitened_residual = noise_model.whiten(x_matrix)
        jac = whitened_residual.jacobian(x_matrix)
        test4 = jac.subs({alpha: 1.0, scale: 2.0, weight: 0.0, epsilon: 1e-10, x: 0.0}).evalf()
        self.assertNear(test4, 0.0)

    @requires_sympy
    def test_barron_noise_model_epsilon_handling(self):
        # type: () -> None
        """
        Epsilon handling for BarronNoiseModel.whiten_norm should be correct (i.e. value and
        derivative at 0 should be correct as epsilon->0)
        """
        scale = 10
        weight = 2

        def test_epsilon_at_alpha(alpha):
            # type: (T.Scalar) -> bool
            def whiten_ratio(x, epsilon):
                # type: (T.Scalar, T.Scalar) -> T.Scalar
                noise_model = nm.BarronNoiseModel(
                    alpha=alpha,
                    scale=scale,
                    weight=weight,
                    alpha_epsilon=epsilon,
                    x_epsilon=epsilon,
                )
                return noise_model.whiten_norm(geo.V1(x))[0, 0]

            # SymPy fails to calculate the limits correctly here, so we provide the correct answers
            return epsilon_handling.is_epsilon_correct(
                whiten_ratio, expected_value=0, expected_derivative=sm.sqrt(weight) / scale
            )

        self.assertTrue(test_epsilon_at_alpha(1))
        self.assertTrue(test_epsilon_at_alpha(2))
        self.assertTrue(test_epsilon_at_alpha(int(-1e10)))


if __name__ == "__main__":
    TestCase.main()
