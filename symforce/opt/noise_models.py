from symforce import geo
from symforce import sympy as sm
from symforce import types as T


class NoiseModel:
    """
    Base class for computing a least squares error term from a residual vector.
    """

    def __init__(self, epsilon: T.Scalar) -> None:
        self.epsilon = epsilon

    def reduce(self, whitened_residual: geo.Matrix) -> T.Scalar:
        """
        Take the sum of squares of the residual.
        """
        return whitened_residual.squared_norm() / 2

    def _whiten(self, residual: geo.Matrix.MatrixT, epsilon: T.Scalar) -> geo.Matrix.MatrixT:
        """
        Whiten the residual vector with the given epsilon.
        """
        raise NotImplementedError()

    def whiten(self, residual: geo.Matrix.MatrixT) -> geo.Matrix.MatrixT:
        """
        Whiten the residual vector.
        """
        return self._whiten(residual, self.epsilon)

    def whiten_norm(self, residual: geo.Matrix.MatrixT) -> geo.Matrix.MatrixT:
        """
        Whiten the norm of the residual vector.

        Let f(x) be the whitening function here, and let x be vector of residuals.
        We compute the whitened residual vector as w(x) = f(||x||)/||x|| * x.
        Then, the overall residual is later computed as ||w(x)|| == f(||x||),
        and so we're minimizing the whitened norm of the full residual
        for each point.
        """
        norm = residual.norm(self.epsilon)

        # norm here is sqrt(epsilon) at 0, so this is safe to call with epsilon=0, and the result
        # is also proportional to sqrt(epsilon), so the division is safe
        whitened_norm = self._whiten(geo.V1(norm), epsilon=0)[0, 0]

        scale_factor = whitened_norm / norm
        return scale_factor * residual

    def error(self, residual: geo.Matrix) -> T.Scalar:
        """
        Return a scalar error.
        """
        return self.reduce(self.whiten(residual))


class BarronNoiseModel(NoiseModel):
    """
    Loss function adapted from:
    A General and Adaptive Robust Loss Function
    Jonathan T. Barron

    e(x) = (b/d) * (( (x/scale)^2 / b + 1)^(d/2) - 1)
    where
    b = |alpha - 2| + epsilon
    d = alpha + epsilon if alpha >= 0 else alpha - epsilon

    The above is the "practical implementation" from Appendix B of the paper.

    The parameter `scale` is the transition point from quadratic to robust, while `alpha`
    controls the shape and convexity. Notable values:
    alpha=2 -> L2 loss
    alpha=1 -> Pseudo-huber loss
    alpha=0 -> Cauchy loss
    alpha=-2 -> Geman-McClure loss
    alpha=-inf -> Welsch loss
    """

    def __init__(
        self,
        alpha: T.Scalar,
        scale: T.Scalar,
        weight: T.Scalar,
        x_epsilon: T.Scalar,
        alpha_epsilon: T.Scalar = None,
    ) -> None:
        """
        Args:
            alpha (Scalar): shape parameter
            scale (Scalar): scale parameter, the transition point
            x_epsilon (Scalar): small value used for handling the singularity at x == 0, defaults
                                to alpha_epsilon
            alpha_epsilon (Scalar): small value used for handling singularities around alpha
        """
        super().__init__(epsilon=x_epsilon)
        self.alpha = alpha
        self.scale = scale
        self.weight = weight
        self.alpha_epsilon = x_epsilon if alpha_epsilon is None else alpha_epsilon

    def barron_error(self, x: T.Scalar) -> T.Scalar:
        """
        Return the barron cost function error for the argument x.

        Args:
            x (Scalar): argument to return the cost for.

        Returns:
            (Scalar): Barron loss at value x
        """
        b = sm.Abs(self.alpha - 2) + self.alpha_epsilon
        d = self.alpha + (sm.sign_no_zero(self.alpha) * self.alpha_epsilon)
        return (b / d) * ((((x / self.scale) ** 2) / b + 1) ** (d / 2) - 1)

    def _whiten(self, residual: geo.Matrix.MatrixT, epsilon: T.Scalar) -> geo.Matrix.MatrixT:
        def whiten_scalar(r: T.Scalar) -> T.Scalar:
            return sm.sqrt(self.weight) * (
                sm.sqrt(2 * self.barron_error(r) + epsilon) - sm.sqrt(epsilon)
            )

        return residual.applyfunc(whiten_scalar)
