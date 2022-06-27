# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------
from __future__ import annotations

from abc import abstractmethod

import symforce.symbolic as sf
from symforce import typing as T


class NoiseModel:
    """
    Base class for whitening unwhitened residuals and/or computing their associated error in a
    least-squares problem.
    """

    @abstractmethod
    def whiten(self, unwhitened_residual: sf.Matrix.MatrixT) -> sf.Matrix.MatrixT:
        """
        Whiten the residual vector.
        """
        pass

    @staticmethod
    def reduce(whitened_residual: sf.Matrix.MatrixT) -> sf.Scalar:
        """
        Take the sum of squares of the residual.
        """
        return whitened_residual.squared_norm() / 2

    def error(self, unwhitened_residual: sf.Matrix.MatrixT) -> sf.Scalar:
        """
        Return a scalar error.
        """
        return self.reduce(self.whiten(unwhitened_residual))


class ScalarNoiseModel(NoiseModel):
    """
    Base class for noise models that apply a whitening function to each element of the unwhitened
    residual. I.e. if f() is the whiten_scalar function, each element of the whitened residual can be
    written as:
        whitened_residual[i] = f(unwhitened_residual[i])
    """

    @abstractmethod
    def whiten_scalar(self, x: sf.Scalar, bounded_away_from_zero: bool = False) -> sf.Scalar:
        """
        A scalar-valued whitening function which is applied to each element of the unwhitened
        residual.

        Args:
            x: A single element of the unwhitened residual
        """
        pass

    def whiten(self, unwhitened_residual: sf.Matrix.MatrixT) -> sf.Matrix.MatrixT:
        """
        Whiten the unwhitened residual vector by applying `whiten_scalar` to each element.
        """
        return unwhitened_residual.applyfunc(self.whiten_scalar)

    def whiten_norm(
        self, residual: sf.Matrix.MatrixT, epsilon: sf.Scalar = sf.epsilon()
    ) -> sf.Matrix.MatrixT:
        """
        Whiten the norm of the residual vector.

        Let f(x) be the whitening function here, and let x be vector of residuals.
        We compute the whitened residual vector as w(x) = f(||x||)/||x|| * x.
        Then, the overall residual is later computed as ||w(x)|| == f(||x||),
        and so we're minimizing the whitened norm of the full residual
        for each point.
        """
        norm = residual.norm(epsilon)

        # Because `norm` is sqrt(epsilon) at 0, we tell `whiten_scalar` the function is bounded
        # away from zero, allowing for the function to ignore epsilons used to avoid singularities
        # at x = 0 if they exist, reducing unneeded ops. The result is also proportional to
        # sqrt(epsilon), so the division is safe
        whitened_norm = self.whiten_scalar(norm, bounded_away_from_zero=True)

        scale_factor = whitened_norm / norm
        return scale_factor * residual


class IsotropicNoiseModel(ScalarNoiseModel):
    """
    Isotropic noise model; equivalent to multiplying the squared residual by a scalar. The cost
    used in the optimization is:

        cost = 0.5 * information * unwhitened_residual.T * unwhitened_residual

    such that:
        cost = 0.5 * whitened_residual.T * whitened_residual

    The whitened residual is:
        whitened_residual = sqrt(information) * unwhitened_residual

    Args:
        scalar_information: Scalar by which the least-squares error will be multiplied. In the
            context of probability theory, the information is the inverse of the variance of the
            unwhitened residual. The information represents the weight given to a specific
            unwhitened residual relative to other residuals used in the least-squares optimization.
        scalar_sqrt_information: Square-root of scalar_information. If scalar_sqrt_information is
            specified, we avoid needing to take the square root of scalar_information. Note that
            only one of scalar_information and scalar_sqrt_information needs to be specified.
    """

    def __init__(
        self,
        scalar_information: T.Optional[sf.Scalar] = None,
        scalar_sqrt_information: T.Optional[sf.Scalar] = None,
    ) -> None:
        if scalar_sqrt_information is not None:
            # User has given the square root information, so we can avoid taking the square root
            self.scalar_sqrt_information = scalar_sqrt_information
        else:
            assert (
                scalar_information is not None
            ), 'Either "scalar_information" or "scalar_sqrt_information" must be provided.'
            self.scalar_sqrt_information = sf.sqrt(scalar_information)

    @classmethod
    def from_variance(cls, variance: sf.Scalar) -> IsotropicNoiseModel:
        """
        Returns an IsotropicNoiseModel given a variance. Typically used when we treat the residual
        as a random variable with known variance, and wish to weight its cost according to the
        information gained by that measurement (i.e. the inverse of the variance).

        Args:
            variance: Typically the variance of the residual elements. Results in cost
                cost = 0.5 * (1 / variance) * unwhitened_residual.T * unwhitened_residual
        """
        return cls(scalar_information=1 / variance)

    @classmethod
    def from_sigma(cls, standard_deviation: sf.Scalar) -> IsotropicNoiseModel:
        """
        Returns an IsotropicNoiseModel given a standard deviation. Typically used when we treat the
        residual as a random variable with known standard deviation, and wish to weight its cost
        according to the information gained by that measurement (i.e. the inverse of the variance).

        Args:
            standard_deviation: The standard deviation of the residual elements. Results in
                cost = 0.5 * (1 / sigma^2) * unwhitened_residual.T * unwhitened_residual
        """
        return IsotropicNoiseModel(scalar_sqrt_information=1 / standard_deviation)

    def whiten_scalar(self, x: sf.Scalar, bounded_away_from_zero: bool = False) -> sf.Scalar:
        """
        Multiplies a single element of the unwhitened residual by `sqrt(information)` so
        that the least-squares cost associated with the element is scaled by `information`.

        Args:
            x: Single element of the unwhitened residual
            bounded_away_from_zero: True if x is guaranteed to not be zero. Typically used to avoid
                extra ops incurred by using epsilons to avoid singularities at x = 0 when it's
                known that x != 0. However, this argument is unused because there is no singularity
                at x = 0 for this whitening function.
        """
        return self.scalar_sqrt_information * x


class DiagonalNoiseModel(NoiseModel):
    """
    Noise model with diagonal weighting matrix. The cost used in the optimization is:

        cost = 0.5 * unwhitened_residual.T * sf.diag(information_diag) * unwhitened_residual
    where `information_diag` is a vector of scalars representing the relative importance of each
    element of the unwhitened residual.

    The total cost is then:
        cost = 0.5 * whitened_residual.T * whitened_residual

    Thus, the whitened residual is:
        whitened_residual = sf.diag(sqrt_information_diag) * unwhitened_residual
    where `sqrt_information_diag` is the element-wise square root of `information_diag`.

    Args:
        information_diag: List of elements of the diagonal of the information matrix. In the context
            of probability theory, this vector represents the inverse of the variance of each
            element of the unwhitened residual, assuming that each element is an independent
            random variable.
        sqrt_information_diag: Element-wise square-root of information_diag. If specified, we avoid
            needing to take the square root of each element of information_diag. Note that only one
            of information_diag and sqrt_information_diag needs to be specified.
    """

    def __init__(
        self,
        information_diag: T.Optional[T.Sequence[sf.Scalar]] = None,
        sqrt_information_diag: T.Optional[T.Sequence[sf.Scalar]] = None,
    ) -> None:
        if sqrt_information_diag is not None:
            # User has given the square root information, so we can avoid taking the square root
            self.sqrt_information_matrix = sf.Matrix.diag(sqrt_information_diag)
        else:
            assert (
                information_diag is not None
            ), 'Either "information_diag" or "sqrt_information_diag" must be provided.'
            self.sqrt_information_matrix = sf.Matrix.diag(information_diag).applyfunc(sf.sqrt)

    @classmethod
    def from_variances(cls, variances: T.Sequence[sf.Scalar]) -> DiagonalNoiseModel:
        """
        Returns an DiagonalNoiseModel given a list of variances of each element of the unwhitened
        residual. Typically used when we treat the unwhitened residual as a sequence of independent
        random variables with known variances.

        Args:
            variances: List of the variances of each element of the unwhitened residual
        """
        return cls(information_diag=[1 / v for v in variances])

    @classmethod
    def from_sigmas(cls, standard_deviations: T.Sequence[sf.Scalar]) -> DiagonalNoiseModel:
        """
        Returns an DiagonalNoiseModel given a list of standard deviations of each element of the
        unwhitened residual. Typically used when we treat the unwhitened residual as a sequence of
        independent random variables with known standard deviations.

        Args:
            standard_deviations: List of the standard deviations of each element of the unwhitened
                residual
        """
        return cls(sqrt_information_diag=[1 / s for s in standard_deviations])

    def whiten(self, unwhitened_residual: sf.Matrix.MatrixT) -> sf.Matrix.MatrixT:
        return T.cast(sf.Matrix.MatrixT, self.sqrt_information_matrix * unwhitened_residual)


class PseudoHuberNoiseModel(ScalarNoiseModel):
    """
    A smooth loss function that behaves like the L2 loss for small x and the L1 loss for large x.
    The cost used in the least-squares optimization will be:
        cost = sum( pseudo_huber_loss(unwhitened_residual[i]) )
        cost = 0.5 * whitened_residual.T * whitened_residual
    where the sum is taken over the elements of the unwhitened residual.

    This noise model applies the square-root of the pseudo-huber loss function to each element of
    the unwhitened residual such that the resulting cost used in the least-squares problem is the
    pseudo-huber loss. The pseudo-huber loss is defined as:
        pseudo_huber_loss(x) = delta^2 * ( sqrt( 1 + scalar_information * (x/delta)^2 ) - 1)

    The whitened residual is then:
        whitened_residual[i] = sqrt( 2 * pseudo_huber_loss(unwhitened_residual[i]) )

    Args:
        delta: Controls the point at which the loss function transitions from the L2 to L1 loss.
            Must be greater than zero.
        scalar_information: Constant scalar weight that changes the steepness of the loss function.
            Can be considered the inverse of the variance of an element of the unwhitened residual.
        epsilon: Small value used to handle singularity at x = 0.
    """

    def __init__(
        self,
        delta: sf.Scalar,
        scalar_information: sf.Scalar,
        epsilon: sf.Scalar = sf.epsilon(),
    ) -> None:
        self.delta = delta
        self.scalar_information = scalar_information
        self.epsilon = epsilon

    def pseudo_huber_error(self, x: sf.Scalar) -> sf.Scalar:
        """
        Return the pseudo-huber cost function error for the argument x.

        Args:
            x: argument to return the cost for.
        """
        return self.delta ** 2 * (sf.sqrt(1 + self.scalar_information * (x / self.delta) ** 2) - 1)

    def whiten_scalar(self, x: sf.Scalar, bounded_away_from_zero: bool = False) -> sf.Scalar:
        """
        Applies a whitening function to a single element of the unwhitened residual such that the
        cost associated with the element in the least-sqaures cost function is the Pseudo-Huber
        loss function.

        Args:
            x: Single element of the unwhitened residual
            bounded_away_from_zero: True if x is guaranteed to not be zero. Typically used to avoid
                extra ops incurred by using epsilons to avoid singularities at x = 0 when it's
                known that x != 0.
        """
        epsilon = self.epsilon
        if bounded_away_from_zero:
            epsilon = 0
        return sf.sqrt(2 * self.pseudo_huber_error(x) + epsilon) - sf.sqrt(epsilon)


class BarronNoiseModel(ScalarNoiseModel):
    """
    Noise model adapted from:
        Barron, Jonathan T. "A general and adaptive robust loss function." Proceedings of the
        IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2019.

    This noise model applies a modified version of the "practical implementation" from Appendix B
    of the paper to each scalar element of an unwhitened residual. The Barron loss function is
    defined as:
        barron_loss(x) = delta^2 * (b/d) * (( scalar_information * (x/delta)^2 / b + 1)^(d/2) - 1)
    where
        b = |alpha - 2| + epsilon
        d = alpha + epsilon if alpha >= 0 else alpha - epsilon
    Here delta controls the point at which the loss function transitions from quadratic to robust.
    This is different from the original Barron loss function, and is designed to match the pseudo-
    huber loss function.

    Thus, the cost used in the optimization will be:
        cost = sum( barron_loss(unwhitened_residual[i]) )
        cost = 0.5 * whitened_residual.T * whitened_residual
    where the sum is taken over the elements of the unwhitened residual.

    Thus, the whitened residual is:
        whitened_residual[i] = sqrt( 2 * barron_loss(unwhitened_residual[i]) )

    Args:
        alpha: Controls shape and convexity of the loss function. Notable values:
            alpha = 2 -> L2 loss
            alpha = 1 -> Pseudo-huber loss
            alpha = 0 -> Cauchy loss
            alpha = -2 -> Geman-McClure loss
            alpha = -inf -> Welsch loss
        delta: Determines the transition point from quadratic to robust. Similar to "delta" as used
            by the pseudo-huber loss function.
        scalar_information: Scalar representing the inverse of the variance of an element of the
            unwhitened residual. Conceptually, we use "scalar_information" to whiten (in a
            probabalistic sense) the unwhitened residual before passing it through the Barron loss.
        x_epsilon: Small value used for handling the singularity at x == 0.
        alpha_epsilon: Small value used for handling singularities around alpha.
    """

    def __init__(
        self,
        alpha: sf.Scalar,
        scalar_information: sf.Scalar,
        x_epsilon: sf.Scalar,
        delta: sf.Scalar = 1,
        alpha_epsilon: sf.Scalar = None,
    ) -> None:
        self.alpha = alpha
        self.delta = delta
        self.scalar_information = scalar_information
        self.x_epsilon = x_epsilon
        self.alpha_epsilon = x_epsilon if alpha_epsilon is None else alpha_epsilon

    @staticmethod
    def compute_alpha_from_mu(mu: sf.Scalar, epsilon: sf.Scalar) -> sf.Scalar:
        """
        Transform mu, which ranges from 0->1, to alpha by alpha=2-1/(1-mu). This transformation
        means alpha will range from 1 to -inf, so that the noise model starts as a pseudo-huber and
        goes to a robust Welsch cost.

        Args:
            mu: ranges from 0->1
            epsilon: small value to avoid numerical instability

        Returns:
            sf.Scalar: alpha for use in the BarronNoiseModel construction
        """
        alpha = 2 - 1 / (1 - mu + epsilon)
        return alpha

    def barron_error(self, x: sf.Scalar) -> sf.Scalar:
        """
        Return the barron cost function error for the argument x.

        Args:
            x: argument to return the cost for.

        Returns:
            Barron loss at value x
        """
        b = sf.Abs(self.alpha - 2) + self.alpha_epsilon
        d = self.alpha + (sf.sign_no_zero(self.alpha) * self.alpha_epsilon)
        return (
            self.delta ** 2
            * (b / d)
            * (((self.scalar_information * (x / self.delta) ** 2) / b + 1) ** (d / 2) - 1)
        )

    def whiten_scalar(self, x: sf.Scalar, bounded_away_from_zero: bool = False) -> sf.Scalar:
        """
        Applies a whitening function to a single element of the unwhitened residual such that the
        cost associated with the element in the least-sqaures cost function is the Barron loss
        function (weighted by self.weight).

        Args:
            x: Single element of the unwhitened residual
        """
        x_epsilon = self.x_epsilon
        if bounded_away_from_zero:
            x_epsilon = 0
        return sf.sqrt(2 * self.barron_error(x) + x_epsilon) - sf.sqrt(x_epsilon)
