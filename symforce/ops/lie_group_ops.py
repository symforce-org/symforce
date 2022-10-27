# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

from __future__ import annotations

import symforce.internal.symbolic as sf
from symforce import logger
from symforce import typing as T
from symforce.typing_util import get_type

from .group_ops import GroupOps

if T.TYPE_CHECKING:
    from symforce import geo


class LieGroupOps(GroupOps):
    """
    API for Lie groups.

    A Lie group is a group that is also a differentiable manifold, with the property that the
    group operations are compatible with the smooth structure.

    A manifold is a topological space that locally resembles Euclidean space near
    each point. More precisely, each point of an n-dimensional manifold has a neighbourhood that
    is homeomorphic to the Euclidean space of dimension n.

    A differentiable manifold is a type of manifold that is locally similar enough to a linear
    space to allow one to do calculus. Any manifold can be described by a collection of charts,
    also known as an atlas. One may then apply ideas from calculus while working within the
    individual charts, since each chart lies within a linear space to which the usual rules of
    calculus apply. If the charts are suitably compatible (namely, the transition from one chart
    to another is differentiable), then computations done in one chart are valid in any other
    differentiable chart.

    References:

        * https://en.wikipedia.org/wiki/Manifold
        * https://en.wikipedia.org/wiki/Differentiable_manifold
        * https://en.wikipedia.org/wiki/Lie_group
    """

    @staticmethod
    def tangent_dim(a: T.ElementOrType) -> int:
        """
        Size of the element's tangent space, aka the degrees of freedom it represents. The
        storage_dim is the higher dimensional space in which this manifold is embedded. For
        example SO3 could be a tangent_dim of 3 with a storage_dim of 4 if storing quaternions,
        or 9 if storing rotation matrices. For vector spaces they are equal.
        """
        return LieGroupOps.implementation(get_type(a)).tangent_dim(a)

    @staticmethod
    def from_tangent(
        a: T.ElementOrType, vec: T.Sequence[T.Scalar], epsilon: T.Scalar = sf.epsilon()
    ) -> T.Element:
        """
        Mapping from the tangent space approximation at identity into a group element of type a.
        For most manifold types this is implemented as the exponential map.

        This method does not rely on the value of a, only the type.

        Args:
            a:
            vec: Tangent space perturbation
            epsilon: Small number to avoid singularity

        Returns:
            Element: Valid group element that approximates vec around identity.
        """
        return LieGroupOps.implementation(get_type(a)).from_tangent(a, vec, epsilon)

    @staticmethod
    def to_tangent(a: T.Element, epsilon: T.Scalar = sf.epsilon()) -> T.List[T.Scalar]:
        """
        Mapping from this element to the tangent space approximation at identity.

        Args:
            a:
            epsilon: Small number to avoid singularity

        Returns:
            list: Tangent space perturbation around identity that approximates a.
        """
        type_a = get_type(a)
        return LieGroupOps.implementation(type_a).to_tangent(a, epsilon)

    @staticmethod
    def retract(
        a: T.Element, vec: T.Sequence[T.Scalar], epsilon: T.Scalar = sf.epsilon()
    ) -> T.Element:
        """
        Apply a tangent space perturbation vec to the group element a. Often used in optimization
        to update nonlinear values from an update step in the tangent space.

        Implementation is simply `compose(a, from_tangent(vec))`.

        Args:
            a:
            vec:
            epsilon: Small number to avoid singularity

        Returns:
            Element: Group element that conceptually represents "a + vec"
        """
        return LieGroupOps.implementation(get_type(a)).retract(a, vec, epsilon)

    @staticmethod
    def local_coordinates(
        a: T.Element, b: T.Element, epsilon: T.Scalar = sf.epsilon()
    ) -> T.List[T.Scalar]:
        """
        Computes a tangent space perturbation around a to produce b. Often used in optimization
        to minimize the distance between two group elements.

        Implementation is simply `to_tangent(between(a, b))`.

        Args:
            a:
            b:
            epsilon: Small number to avoid singularity

        Returns:
            list: Tangent space perturbation that conceptually represents "b - a"
        """
        return LieGroupOps.implementation(get_type(a)).local_coordinates(a, b, epsilon)

    @staticmethod
    def interpolate(
        a: T.Element, b: T.Element, alpha: T.Scalar, epsilon: T.Scalar = sf.epsilon()
    ) -> T.List[T.Scalar]:
        """
        Interpolates between self and b.

        Implementation is to take the perturbation between self and b in tangent space
        (local_coordinates) and add a scaled version of that to self (retract).

        Args:
            a:
            b:
            alpha: ratio between a and b - 0 is a, 1 is b. Note that this variable
                   is not clamped between 0 and 1 in this function.
            epsilon: Small number to avoid singularity

        Returns:
            Element: Interpolated group element
        """
        return LieGroupOps.retract(
            a, [c * alpha for c in LieGroupOps.local_coordinates(a, b, epsilon)], epsilon
        )

    @staticmethod
    def storage_D_tangent(a: T.Element) -> geo.Matrix:
        """
        Computes the jacobian of the storage space of an element with respect to the tangent space around
        that element.
        """
        try:
            return LieGroupOps.implementation(get_type(a)).storage_D_tangent(a)
        except NotImplementedError:
            logger.error(
                "storage_D_tangent not implemented for {}; use storage_D_tangent.ipynb to compute".format(
                    get_type(a)
                )
            )
            raise

    @staticmethod
    def tangent_D_storage(a: T.Element) -> geo.Matrix:
        """
        Computes the jacobian of the tangent space around an element with respect to the storage space of
        that element.
        """
        try:
            return LieGroupOps.implementation(get_type(a)).tangent_D_storage(a)
        except NotImplementedError:
            logger.error(
                "tangent_D_storage not implemented for {}; use tangent_D_storage.ipynb to compute".format(
                    get_type(a)
                )
            )
            raise
