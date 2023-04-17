# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

from __future__ import annotations

import numpy as np

import symforce.internal.symbolic as sf
from symforce import typing as T
from symforce.ops.interfaces import Group


class Complex(Group):
    """
    A complex number is a number that can be expressed in the form a + bi, where a and b are real
    numbers, and i is a solution of the equation x**2 = -1. Because no real number satisfies this
    equation, i is called an imaginary number. For the complex number a + bi, a is called the
    real part, and b is called the imaginary part. Despite the historical nomenclature "imaginary",
    complex numbers are regarded in the mathematical sciences as just as "real" as the real numbers,
    and are fundamental in many aspects of the scientific description of the natural world.

    A complex number is also a convenient way to store a two-dimensional rotation.

    References:

        https://en.wikipedia.org/wiki/Complex_number
    """

    def __init__(self, real: T.Scalar, imag: T.Scalar) -> None:
        """
        Construct from a real and imaginary scalar.

        Args:
            real (Scalar):
            imag (Scalar):
        """
        self.real = real
        self.imag = imag

    # -------------------------------------------------------------------------
    # Storage concept - see symforce.ops.storage_ops
    # -------------------------------------------------------------------------

    def __repr__(self) -> str:
        return "<C real={}, imag={}>".format(repr(self.real), repr(self.imag))

    @classmethod
    def storage_dim(cls) -> int:
        return 2

    def to_storage(self) -> T.List[T.Scalar]:
        return [self.real, self.imag]

    @classmethod
    def from_storage(cls, vec: T.Sequence[T.Scalar]) -> Complex:
        assert len(vec) == cls.storage_dim()
        return cls(real=vec[0], imag=vec[1])

    @classmethod
    def symbolic(cls, name: str, **kwargs: T.Any) -> Complex:
        return cls.from_storage([sf.Symbol(f"{name}_{v}", **kwargs) for v in ["re", "im"]])

    # -------------------------------------------------------------------------
    # Group concept - see symforce.ops.group_ops
    # -------------------------------------------------------------------------

    @classmethod
    def identity(cls) -> Complex:
        return Complex(1, 0)

    def compose(self, other: Complex) -> Complex:
        return self.__class__(
            real=self.real * other.real - self.imag * other.imag,
            imag=self.imag * other.real + self.real * other.imag,
        )

    def inverse(self) -> Complex:
        return self.conj() / self.squared_norm()

    # -------------------------------------------------------------------------
    # Helper methods
    # -------------------------------------------------------------------------

    @classmethod
    def zero(cls) -> Complex:
        """
        Zero value.

        Returns:
            Complex:
        """
        return cls(0, 0)

    def conj(self) -> Complex:
        """
        Complex conjugate (real, -imag).

        Returns:
            Complex:
        """
        return self.__class__(self.real, -self.imag)

    def squared_norm(self) -> T.Scalar:
        """
        Squared norm of the two-vector.

        Returns:
            Scalar: real**2 + imag**2
        """
        return self.real ** 2 + self.imag ** 2

    def __mul__(self, right: Complex) -> Complex:
        """
        Complex multiplication (composition).

        Args:
            right (Complex):

        Returns:
            Complex:
        """
        return self.compose(right)

    def __add__(self, right: Complex) -> Complex:
        """
        Element-wise addition.

        Args:
            right (Complex):

        Returns:
            Complex:
        """
        return self.__class__(self.real + right.real, self.imag + right.imag)

    def __neg__(self) -> Complex:
        """
        Element-wise negation.

        Returns:
            Complex:
        """
        return self.__class__(-self.real, -self.imag)

    def __div__(self, scalar: T.Scalar) -> Complex:
        """
        Scalar element-wise division.

        Args:
            scalar (Scalar):

        Returns:
            Complex:
        """
        return self.__class__(self.real / scalar, self.imag / scalar)

    __truediv__ = __div__

    @classmethod
    def random_uniform(cls, low: T.Scalar, high: T.Scalar) -> Complex:
        """
        Generate a random complex number with real and imaginary parts between the given bounds
        """
        re = np.random.uniform(low, high)
        im = np.random.uniform(low, high)
        return Complex(re, im)

    @classmethod
    def unit_random(cls) -> Complex:
        """
        Generate a unit-norm random complex number
        """
        u1 = np.random.uniform(low=0.0, high=1.0)
        return cls.unit_random_from_uniform_sample(u1, pi=np.pi)

    @classmethod
    def unit_random_from_uniform_sample(cls, u1: T.Scalar, pi: T.Scalar = sf.pi) -> Complex:
        """
        Generate a unit-norm random Complex number from a variable sampled uniformly on [0, 1]
        """
        theta = 2 * pi * u1
        return Complex(sf.cos(theta), sf.sin(theta))
