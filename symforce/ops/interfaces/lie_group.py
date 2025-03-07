# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

from __future__ import annotations

import symforce.internal.symbolic as sf
from symforce import ops
from symforce import typing as T

from .group import Group

if T.TYPE_CHECKING:
    from symforce import geo


class LieGroup(Group):
    """
    Interface for objects that implement the lie group concept. Because this class is registered
    using :class:`symforce.ops.impl.class_lie_group_ops.ClassLieGroupOps` (see bottom of this file),
    any object that inherits from ``LieGroup`` and that implements the functions defined in this
    class can be used with the LieGroupOps concept.

    Note that ``LieGroup`` is a subclass of :class:`.group.Group` which is a subclass of
    :class:`.storage.Storage`, meaning that a ``LieGroup`` object can be also be used with GroupOps
    and StorageOps (assuming the necessary functions are implemented).
    """

    # Type that represents this or any subclasses
    LieGroupT = T.TypeVar("LieGroupT", bound="LieGroup")

    @classmethod
    def tangent_dim(cls) -> int:
        """
        Dimension of the embedded manifold
        """
        raise NotImplementedError()

    @classmethod
    def from_tangent(
        cls: T.Type[LieGroupT], vec: T.Sequence[T.Scalar], epsilon: T.Scalar = sf.epsilon()
    ) -> LieGroupT:
        """
        Mapping from the tangent space vector about identity into a group element.
        """
        raise NotImplementedError()

    def to_tangent(self: LieGroupT, epsilon: T.Scalar = sf.epsilon()) -> T.List[T.Scalar]:
        """
        Mapping from this element to the tangent space vector about identity.
        """
        raise NotImplementedError()

    def retract(
        self: LieGroupT, vec: T.Sequence[T.Scalar], epsilon: T.Scalar = sf.epsilon()
    ) -> LieGroupT:
        """
        Applies a tangent space perturbation vec to self. Often used in optimization
        to update nonlinear values from an update step in the tangent space.

        Implementation is simply `compose(self, from_tangent(vec))`.
        Conceptually represents "self + vec" if self is a vector.
        """
        return self.compose(self.from_tangent(vec, epsilon=epsilon))

    def local_coordinates(
        self: LieGroupT, b: LieGroupT, epsilon: T.Scalar = sf.epsilon()
    ) -> T.List[T.Scalar]:
        """
        Computes a tangent space perturbation around self to produce b. Often used in optimization
        to minimize the distance between two group elements.

        Implementation is simply `to_tangent(between(self, b))`.
        Tangent space perturbation that conceptually represents "b - self" if self is a vector.
        """
        return self.between(b).to_tangent(epsilon=epsilon)

    def jacobian(self: LieGroupT, X: T.Any, tangent_space: bool = True) -> geo.Matrix:
        """
        Computes the jacobian of this LieGroup element with respect to the input X, where X is
        anything that supports LieGroupOps

        If tangent_space is True, the jacobian is computed in the local coordinates of the tangent
        spaces around self and X. If tangent_space is False, the jacobian is computed in the storage
        spaces of self and X.

        See Also:
            :func:`symforce.ops.lie_group_ops.LieGroupOps.jacobian`
            :func:`symforce.jacobian_helpers.tangent_jacobians`

        Returns: the jacobian matrix of shape MxN, where M is the dimension of the tangent (or
            storage) space of self and N is the dimension of the tangent (or storage) space of X.
        """
        return ops.LieGroupOps.jacobian(self, X, tangent_space)


from ..impl.class_lie_group_ops import ClassLieGroupOps

ops.LieGroupOps.register(LieGroup, ClassLieGroupOps)
